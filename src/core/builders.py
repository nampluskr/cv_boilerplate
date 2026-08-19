import torch
from torch.utils.data import DataLoader

from src.core.context import make_worker_init_fn
from src.core.registry import BUILDERS

BUILDERS.register("adamw")(lambda target, **params: torch.optim.AdamW(target, **params))
BUILDERS.register("sgd")(lambda target, **params: torch.optim.SGD(target, **params))


@BUILDERS.register("cosine")
def build_cosine_scheduler(target, t_max, eta_min=0.0, **params):
    return torch.optim.lr_scheduler.CosineAnnealingLR(target, T_max=t_max, eta_min=eta_min)


@BUILDERS.register("step")
def build_step_scheduler(target, step_size, gamma=0.1, **params):
    return torch.optim.lr_scheduler.StepLR(target, step_size=step_size, gamma=gamma)


def build_optimizer(config_optim, model):
    spec = config_optim["optimizer"]
    trainable = (p for p in model.parameters() if p.requires_grad)
    return BUILDERS.build(spec["name"], trainable, **spec.get("params", {}))


def build_scheduler(config_optim, optimizer):
    spec = config_optim.get("scheduler")
    if spec is None:
        return None
    return BUILDERS.build(spec["name"], optimizer, **spec.get("params", {}))


def build_dataloader(dataset, config_data, split, adapter, seed, device, allow_test_split=False):
    if split == "test" and not allow_test_split:
        raise RuntimeError(
            "test split DataLoader can only be built when allow_test_split=True "
            "(evaluate/benchmark commands only)."
        )
    return DataLoader(
        dataset,
        batch_size=config_data["batch_size"],
        shuffle=(split == "train"),
        num_workers=config_data["num_workers"],
        collate_fn=adapter.collate_fn(),
        worker_init_fn=make_worker_init_fn(seed),
        generator=torch.Generator().manual_seed(seed),
        drop_last=(config_data["drop_last"] and split == "train"),
        pin_memory=(device == "cuda"),
        persistent_workers=(config_data["num_workers"] > 0),
    )
