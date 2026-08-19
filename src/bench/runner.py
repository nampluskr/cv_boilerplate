import os
import time

import torch

from src.bench.control import enforce_control
from src.bench.profile import profile_model
from src.core.builders import build_dataloader, build_optimizer, build_scheduler
from src.core.config import apply_overrides, deep_merge, load_and_merge_base, validate_config
from src.core.context import RunContext
from src.core.engine import Trainer
from src.core.logger import MetricsCsvWriter, setup_logger
from src.core.offline import disable_offline_guard, enable_offline_guard
from src.core.paths import build_benchmark_dir, build_benchmark_split_dir
from src.core.registry import ADAPTERS, DATASETS, LOSSES, METRICS, MODELS, TRANSFORMS
from src.utils.io import load_yaml, save_json, save_yaml


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


def resolve_split_configs(bench_path, cli_overrides):
    bench = load_yaml(bench_path)
    base_config = load_and_merge_base(bench["base"])
    split_configs = {}
    for split_spec in bench["splits"]:
        merged = deep_merge(base_config, split_spec.get("override", {}))
        merged = apply_overrides(merged, cli_overrides or [])
        split_configs[split_spec["name"]] = merged
    return bench, split_configs


def execute_split(bench_name, split_name, config, overwrite):
    split_dir = build_benchmark_split_dir(config["output"]["root"], bench_name, split_name)
    if os.path.exists(split_dir) and not overwrite:
        raise RuntimeError(f"split output already exists (use --overwrite): {split_dir}")
    os.makedirs(split_dir, exist_ok=True)
    save_yaml(config, os.path.join(split_dir, "config.resolved.yaml"))

    logger = setup_logger(split_dir, "INFO", name=f"bench.{split_name}")
    ctx = RunContext(config, split_dir, allow_test_split=True)
    ctx.setup_seed()

    transform_train, transform_eval = build_transforms(config)
    model, adapter = build_components(config)
    adapter.to(ctx.device)
    train_ds = build_dataset(config, "train", transform_train)
    valid_ds = build_dataset(config, "valid", transform_eval)
    test_ds = build_dataset(config, "test", transform_eval)
    bind_class_names(adapter, train_ds)

    train_loader = build_dataloader(train_ds, config["data"], "train", adapter, ctx.seed,
                                     config["runtime"]["device"])
    valid_loader = build_dataloader(valid_ds, config["data"], "valid", adapter, ctx.seed,
                                     config["runtime"]["device"])
    test_loader = build_dataloader(test_ds, config["data"], "test", adapter, ctx.seed,
                                    config["runtime"]["device"], allow_test_split=True)

    optimizer = build_optimizer(config["optim"], model)
    scheduler = build_scheduler(config["optim"], optimizer)
    metric_names = [m["name"] for m in config["metrics"]]
    metrics_writer = MetricsCsvWriter(split_dir, metric_names)
    checkpoint_dir = os.path.join(split_dir, "checkpoints")
    trainer = Trainer(logger=logger, metrics_writer=metrics_writer, checkpoint_dir=checkpoint_dir)

    start = time.perf_counter()
    best_metric = trainer.fit(model, adapter, train_loader, valid_loader, optimizer, scheduler, ctx)
    train_time_sec = time.perf_counter() - start

    # See cli/commands.py::train() for why valid is re-evaluated here instead of reusing
    # `best_metric`: on_fit_end can still mutate scoring buffers on the model after fit()'s
    # mid-training best_metric was captured. Throwaway Trainer (no metrics_writer) so this pass
    # doesn't append to metrics_epoch.csv.
    final_valid = Trainer(logger=logger).evaluate(model, adapter, valid_loader, ctx, epoch=None, split="valid")
    test_results = trainer.evaluate(model, adapter, test_loader, ctx, epoch=None, split="test")

    if config["output"].get("save_visualizations", False):
        max_items = config["output"].get("max_visualizations", 16)
        viz_dir = os.path.join(split_dir, "visualizations")
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                predictions = adapter.predict_step(model, batch, ctx.device)
                adapter.visualize(batch, predictions, viz_dir, max_items)
                break

    profile_result = profile_model(model, adapter, config["data"]["image_size"], ctx.device)

    metrics_final = {"valid": final_valid, "test": test_results, "profile": profile_result}
    metrics_final.update(adapter.extra_final_metrics())
    save_json(metrics_final, os.path.join(split_dir, "metrics_final.json"))
    save_json(ctx.env_info(), os.path.join(split_dir, "env.json"))

    row = {
        "split": split_name,
        "model": config["model"]["name"],
        "status": "ok",
        "params_total": profile_result["params_total"],
        "flops_g": profile_result["flops_g"],
        "fps": profile_result["fps"],
        "best_epoch": None,
        "train_time_sec": round(train_time_sec, 2),
        "batch_size": config["data"]["batch_size"],
        "image_size": config["data"]["image_size"],
        "seed": config["runtime"]["seed"],
        "control_status": "ok",
        "exceptions": "",
        "run_dir": split_dir,
    }
    row.update(test_results)
    return row


def run_benchmark(bench_path, cli_overrides, only=None, overwrite=False):
    bench, split_configs = resolve_split_configs(bench_path, cli_overrides)
    for config in split_configs.values():
        validate_config(config)

    exceptions = bench.get("control", {}).get("exceptions", [])
    extra_fields = bench.get("control", {}).get("extra_fields", [])
    control_report = enforce_control(split_configs, exceptions, extra_fields)

    # runtime.allow_network is a controlled field, so every split agrees on it; apply once.
    if next(iter(split_configs.values()))["runtime"]["allow_network"]:
        disable_offline_guard()
    else:
        enable_offline_guard()
    # Recorded so `leaderboard` can detect a split directory silently missing from a later
    # rebuild instead of comparing only whatever directories happen to exist (Codex A1 finding #2).
    control_report["expected_splits"] = [split_spec["name"] for split_spec in bench["splits"]]

    bench_dir = build_benchmark_dir(next(iter(split_configs.values()))["output"]["root"], bench["name"])
    os.makedirs(bench_dir, exist_ok=True)
    save_json(control_report, os.path.join(bench_dir, "control_report.json"))

    requested = set(only.split(",")) if only else None
    rows = []
    for split_spec in bench["splits"]:
        split_name = split_spec["name"]
        if requested is not None and split_name not in requested:
            continue
        config = split_configs[split_name]
        try:
            row = execute_split(bench["name"], split_name, config, overwrite)
        except Exception as exc:  # noqa: BLE001 - a failed split must not stop the benchmark
            row = {
                "split": split_name, "model": config["model"]["name"], "status": "failed",
                "run_dir": build_benchmark_split_dir(config["output"]["root"], bench["name"], split_name),
                "error": str(exc),
            }
        rows.append(row)

    return bench, bench_dir, rows, control_report
