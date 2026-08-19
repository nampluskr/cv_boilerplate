import os
import sys

import torch
import yaml
from PIL import Image
from torch.utils.data.dataloader import default_collate

from src.bench.control import enforce_control
from src.bench.leaderboard import build_leaderboard as write_leaderboard
from src.bench.runner import run_benchmark as execute_benchmark
from src.core.builders import build_dataloader, build_optimizer, build_scheduler
from src.core.checkpoint import load_checkpoint
from src.core.config import resolve_config, validate_config
from src.core.context import RunContext
from src.core.engine import Trainer
from src.core.logger import MetricsCsvWriter, setup_logger
from src.core.offline import disable_offline_guard, enable_offline_guard
from src.core.paths import build_run_dir
from src.core.registry import ADAPTERS, DATASETS, LOSSES, METRICS, MODELS, TRANSFORMS
from src.utils.io import load_json, load_yaml, save_json, save_yaml


def check_assets(assets_path):
    assets = load_yaml(assets_path)
    rows = []
    all_ok = True

    for name, spec in assets.get("datasets", {}).items():
        root = spec["root"]
        for rel in spec.get("require", []):
            path = os.path.join(root, rel)
            exists = os.path.exists(path)
            size = describe_size(path) if exists else "-"
            rows.append(("dataset", f"{name}/{rel}", path, exists, size))
            all_ok = all_ok and exists

    for name, path in assets.get("weights", {}).items():
        exists = os.path.isfile(path)
        size = describe_size(path) if exists else "-"
        rows.append(("weight", name, path, exists, size))
        all_ok = all_ok and exists

    print(f"{'type':<8} {'name':<32} {'status':<8} {'size':<10} path")
    for kind, name, path, exists, size in rows:
        status = "OK" if exists else "MISSING"
        print(f"{kind:<8} {name:<32} {status:<8} {size:<10} {path}")

    if not all_ok:
        print("check-assets: one or more assets are missing.")
        sys.exit(1)
    print("check-assets: all assets present.")


def describe_size(path):
    if os.path.isdir(path):
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                total += os.path.getsize(os.path.join(dirpath, filename))
    else:
        total = os.path.getsize(path)
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024:
            return f"{total:.0f}{unit}"
        total /= 1024
    return f"{total:.0f}TB"


def build_transforms(config):
    image_size = config["data"]["image_size"]
    train_spec = config["data"]["transform"]["train"]
    eval_spec = config["data"]["transform"]["eval"]
    transform_train = TRANSFORMS.build(train_spec["name"], image_size, True, **train_spec.get("params", {}))
    transform_eval = TRANSFORMS.build(eval_spec["name"], image_size, False, **eval_spec.get("params", {}))
    return transform_train, transform_eval


def build_components(config):
    model = MODELS.build(config["model"]["name"], **config["model"].get("params", {}))
    loss_fn = LOSSES.build(config["loss"]["name"], **config["loss"].get("params", {}))
    metrics = {
        metric_spec["name"]: METRICS.build(metric_spec["name"], **metric_spec.get("params", {}))
        for metric_spec in config["metrics"]
    }
    adapter = ADAPTERS.build(config["adapter"]["name"], loss_fn, metrics, **config["adapter"].get("params", {}))
    return model, adapter


def build_dataset(config, split, transform):
    params = dict(config["data"].get("params", {}))
    split_cfg = config["data"].get("split", {})
    if split_cfg.get("mode") == "file" and "split_path" not in params:
        params["split_path"] = split_cfg["path"]
    return DATASETS.build(
        config["data"]["name"], root=config["data"]["root"], split=split, transform=transform,
        **params,
    )


def bind_class_names(adapter, dataset):
    if hasattr(dataset, "classes"):
        adapter.class_names = dataset.classes


def apply_network_policy(config):
    """The offline guard is armed unconditionally at process start (SS __main__.py); this only
    relaxes it when the resolved config explicitly opts in (PLAN-P1 SS9.1, Codex A1 finding #10)."""
    if config["runtime"]["allow_network"]:
        disable_offline_guard()
    else:
        enable_offline_guard()


def print_resolved_config(config_path, overrides):
    config = resolve_config(config_path, overrides)
    validate_config(config)
    print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))


def train(config_path, overrides, log_level, resume=None):
    config = resolve_config(config_path, overrides)
    validate_config(config)
    apply_network_policy(config)

    run_dir = build_run_dir(config["output"]["root"], config["meta"]["task_name"],
                             config["output"]["run_name"], config_path)
    os.makedirs(run_dir, exist_ok=True)
    save_yaml(config, os.path.join(run_dir, "config.resolved.yaml"))

    logger = setup_logger(run_dir, log_level)
    ctx = RunContext(config, run_dir, allow_test_split=False)
    ctx.start(" ".join(sys.argv))
    ctx.setup_seed()

    transform_train, transform_eval = build_transforms(config)
    model, adapter = build_components(config)
    adapter.to(ctx.device)
    train_ds = build_dataset(config, "train", transform_train)
    valid_ds = build_dataset(config, "valid", transform_eval)
    bind_class_names(adapter, train_ds)
    train_loader = build_dataloader(train_ds, config["data"], "train", adapter, ctx.seed,
                                     config["runtime"]["device"])
    valid_loader = build_dataloader(valid_ds, config["data"], "valid", adapter, ctx.seed,
                                     config["runtime"]["device"])
    optimizer = build_optimizer(config["optim"], model)
    scheduler = build_scheduler(config["optim"], optimizer)

    metric_names = [m["name"] for m in config["metrics"]]
    metrics_writer = MetricsCsvWriter(run_dir, metric_names)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    trainer = Trainer(logger=logger, metrics_writer=metrics_writer, checkpoint_dir=checkpoint_dir)

    start_epoch, best_metric = 1, None
    if resume:
        checkpoint = load_checkpoint(resume, model, optimizer, scheduler)
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint["best_metric"]

    best = trainer.fit(model, adapter, train_loader, valid_loader, optimizer, scheduler, ctx,
                        start_epoch=start_epoch, best_metric=best_metric)
    ctx.finish()

    # Re-evaluate valid with the reloaded-best + on_fit_end-calibrated model (fit()'s returned
    # `best` was captured mid-training, before on_fit_end could mutate scoring buffers on the
    # model, so it can silently go stale relative to the final checkpoint). A throwaway Trainer
    # avoids appending a spurious row to metrics_epoch.csv.
    final_valid = Trainer(logger=logger).evaluate(model, adapter, valid_loader, ctx, epoch=None, split="valid")
    monitor_metric = config["train"]["monitor"]["metric"]
    metrics_final = {"valid": final_valid}
    metrics_final.update(adapter.extra_final_metrics())
    save_json(metrics_final, os.path.join(run_dir, "metrics_final.json"))
    save_json(ctx.env_info(), os.path.join(run_dir, "env.json"))
    logger.info(f"training complete. run_dir={run_dir} best {monitor_metric}={final_valid.get(monitor_metric, best)}")
    return run_dir


def evaluate(config_path, overrides, log_level, checkpoint, split):
    config = resolve_config(config_path, overrides)
    validate_config(config)
    apply_network_policy(config)

    run_dir = build_run_dir(config["output"]["root"], config["meta"]["task_name"],
                             f"{config['output']['run_name'] or 'evaluate'}__{split}", config_path)
    os.makedirs(run_dir, exist_ok=True)
    save_yaml(config, os.path.join(run_dir, "config.resolved.yaml"))

    logger = setup_logger(run_dir, log_level)
    ctx = RunContext(config, run_dir, allow_test_split=True)
    ctx.start(" ".join(sys.argv))
    ctx.setup_seed()

    _, transform_eval = build_transforms(config)
    model, adapter = build_components(config)
    adapter.to(ctx.device)
    dataset = build_dataset(config, split, transform_eval)
    bind_class_names(adapter, dataset)
    loader = build_dataloader(dataset, config["data"], split, adapter, ctx.seed,
                               config["runtime"]["device"], allow_test_split=True)

    load_checkpoint(checkpoint, model, map_location="cpu", restore_rng=False)
    model.to(ctx.device)

    trainer = Trainer(logger=logger)
    results = trainer.evaluate(model, adapter, loader, ctx, epoch=None, split=split)
    save_json(results, os.path.join(run_dir, "metrics_final.json"))

    if config["output"].get("save_visualizations", False):
        max_items = config["output"].get("max_visualizations", 16)
        viz_dir = os.path.join(run_dir, "visualizations")
        model.eval()
        with torch.no_grad():
            for batch in loader:
                predictions = adapter.predict_step(model, batch, ctx.device)
                adapter.visualize(batch, predictions, viz_dir, max_items)
                break

    logger.info(f"evaluate complete. run_dir={run_dir} results={results}")
    return results


def load_input_images(input_path):
    if os.path.isdir(input_path):
        filenames = sorted(f for f in os.listdir(input_path)
                            if f.lower().endswith((".png", ".jpg", ".jpeg")))
        paths = [os.path.join(input_path, f) for f in filenames]
    else:
        paths = [input_path]

    # Raw PIL images only, no pre-resize: the task's own transform_eval owns resize/crop/
    # normalize so predict() sees the same preprocessing as the Dataset eval path (Codex A2
    # finding #2 - a pre-resize here distorted aspect ratio before transform_eval ran again).
    images = [Image.open(path).convert("RGB") for path in paths]
    stems = [os.path.splitext(os.path.basename(path))[0] for path in paths]
    return images, paths, stems


def predict(config_path, overrides, log_level, checkpoint, input_path, output_dir):
    config = resolve_config(config_path, overrides)
    validate_config(config)
    apply_network_policy(config)

    run_dir = output_dir or build_run_dir(config["output"]["root"], config["meta"]["task_name"],
                                           f"{config['output']['run_name'] or 'predict'}__predict",
                                           config_path)
    os.makedirs(run_dir, exist_ok=True)

    logger = setup_logger(run_dir, log_level)
    ctx = RunContext(config, run_dir, allow_test_split=False)
    ctx.setup_seed()

    _, transform_eval = build_transforms(config)
    model, adapter = build_components(config)
    adapter.bind_class_names_from_config(config["data"])
    adapter.to(ctx.device)
    load_checkpoint(checkpoint, model, map_location="cpu", restore_rng=False)
    model.to(ctx.device)
    model.eval()

    images, paths, stems = load_input_images(input_path)
    images = [transform_eval(image) for image in images]
    collate = adapter.collate_fn() or default_collate
    batch = collate(list(zip(images, stems)))

    with torch.no_grad():
        predictions = adapter.predict_step(model, batch, ctx.device)

    predictions_dir = os.path.join(run_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    save_json({"paths": paths, "predictions": predictions}, os.path.join(predictions_dir, "predict.json"))
    adapter.save_predictions(predictions, predictions_dir)
    logger.info(f"predict complete. run_dir={run_dir} num_predictions={len(predictions)}")
    return predictions


def run_benchmark(bench_path, overrides, log_level, only=None, overwrite=False):
    bench, bench_dir, rows, control_report = execute_benchmark(bench_path, overrides, only, overwrite)

    from src.core.config import load_and_merge_base
    base_config = load_and_merge_base(bench["base"])
    metric_names = [m["name"] for m in base_config["metrics"]]
    monitor_metric = base_config["train"]["monitor"]["metric"]
    monitor_mode = base_config["train"]["monitor"]["mode"]

    write_leaderboard(bench_dir, bench["name"], rows, metric_names, monitor_metric, monitor_mode,
                       control_report)
    print(f"benchmark complete. bench_dir={bench_dir}")
    for row in rows:
        print(f"  {row['split']}: status={row['status']}")
    return bench_dir, rows


def build_leaderboard(bench_output_dir):
    splits_dir = os.path.join(bench_output_dir, "splits")
    split_names = sorted(
        name for name in os.listdir(splits_dir) if os.path.isdir(os.path.join(splits_dir, name))
    )

    split_configs = {}
    for split_name in split_names:
        config_path = os.path.join(splits_dir, split_name, "config.resolved.yaml")
        if os.path.isfile(config_path):
            split_configs[split_name] = load_yaml(config_path)

    report_path = os.path.join(bench_output_dir, "control_report.json")
    previous_report = load_json(report_path) if os.path.isfile(report_path) else {"exceptions_applied": []}
    exceptions = previous_report.get("exceptions_applied", [])
    extra_fields = previous_report.get("extra_fields", [])

    expected_splits = previous_report.get("expected_splits")
    if expected_splits is not None:
        missing = sorted(set(expected_splits) - set(split_names))
        if missing:
            raise RuntimeError(
                f"leaderboard: split directories missing under {splits_dir}: {missing}. "
                f"The original benchmark run expected {expected_splits}. Refusing to build a "
                f"leaderboard that silently drops splits from the control comparison."
            )

    control_report = enforce_control(split_configs, exceptions, extra_fields)
    control_report["expected_splits"] = expected_splits if expected_splits is not None else split_names
    save_json(control_report, report_path)

    rows = []
    for split_name, config in split_configs.items():
        split_dir = os.path.join(splits_dir, split_name)
        metrics_final_path = os.path.join(split_dir, "metrics_final.json")
        if not os.path.isfile(metrics_final_path):
            rows.append({"split": split_name, "model": config["model"]["name"], "status": "failed",
                         "run_dir": split_dir})
            continue
        data = load_json(metrics_final_path)
        row = {
            "split": split_name,
            "model": config["model"]["name"],
            "status": "ok",
            "params_total": data["profile"]["params_total"],
            "flops_g": data["profile"]["flops_g"],
            "fps": data["profile"]["fps"],
            "best_epoch": None,
            "train_time_sec": None,
            "batch_size": config["data"]["batch_size"],
            "image_size": config["data"]["image_size"],
            "seed": config["runtime"]["seed"],
            "control_status": "ok",
            "exceptions": "",
            "run_dir": split_dir,
        }
        row.update(data.get("test", {}))
        rows.append(row)

    reference_config = next(iter(split_configs.values()))
    metric_names = [m["name"] for m in reference_config["metrics"]]
    monitor_metric = reference_config["train"]["monitor"]["metric"]
    monitor_mode = reference_config["train"]["monitor"]["mode"]
    bench_name = os.path.basename(bench_output_dir.rstrip("/"))

    write_leaderboard(bench_output_dir, bench_name, rows, metric_names, monitor_metric, monitor_mode,
                       control_report)
    print(f"leaderboard regenerated. bench_dir={bench_output_dir}")
    return bench_output_dir, rows
