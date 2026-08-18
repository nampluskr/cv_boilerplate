import copy
import os

import torch
import yaml

from src.core.errors import ConfigError
from src.core.registry import ADAPTERS, DATASETS, LOSSES, METRICS, MODELS, TRANSFORMS

TOP_LEVEL_KEYS = {
    "meta", "runtime", "data", "model", "loss", "metrics", "adapter", "optim", "train", "output",
}

MAX_INHERIT_DEPTH = 3


def deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return copy.deepcopy(override)
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_raw_yaml(path):
    if not os.path.isfile(path):
        raise ConfigError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_and_merge_base(path, depth=0, visited=None):
    if visited is None:
        visited = set()
    if depth > MAX_INHERIT_DEPTH:
        raise ConfigError(f"_base inheritance depth exceeds {MAX_INHERIT_DEPTH} at '{path}'.")
    real_path = os.path.realpath(path)
    if real_path in visited:
        raise ConfigError(f"Circular _base reference detected at '{path}'.")
    visited = visited | {real_path}

    raw = load_raw_yaml(path)
    base_spec = raw.pop("_base", None)

    merged = {}
    if base_spec is not None:
        base_paths = base_spec if isinstance(base_spec, list) else [base_spec]
        for base_path in base_paths:
            resolved_base_path = os.path.join(os.path.dirname(path), base_path)
            if not os.path.isfile(resolved_base_path):
                resolved_base_path = base_path
            merged = deep_merge(merged, load_and_merge_base(resolved_base_path, depth + 1, visited))

    merged = deep_merge(merged, raw)
    return merged


def parse_set_arg(arg):
    if "=" not in arg:
        raise ConfigError(f"--set argument must be of form <dotted.key>=<yaml_value>, got '{arg}'")
    key, _, value_str = arg.partition("=")
    value = yaml.safe_load(value_str)
    return key.strip(), value


def apply_set_override(config, key, value):
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        node = descend(node, part, key)
    set_leaf(node, parts[-1], value, key)


def descend(node, part, full_key):
    if isinstance(node, list):
        if not part.lstrip("-").isdigit() or int(part) >= len(node):
            raise ConfigError(f"--set key '{full_key}' references nonexistent index '{part}'.")
        return node[int(part)]
    if isinstance(node, dict):
        if part not in node:
            raise ConfigError(f"--set key '{full_key}' references nonexistent key '{part}'.")
        return node[part]
    raise ConfigError(f"--set key '{full_key}' cannot descend into a scalar at '{part}'.")


def set_leaf(node, part, value, full_key):
    if isinstance(node, list):
        if not part.lstrip("-").isdigit() or int(part) >= len(node):
            raise ConfigError(f"--set key '{full_key}' references nonexistent index '{part}'.")
        node[int(part)] = value
        return
    if isinstance(node, dict):
        if part not in node:
            raise ConfigError(f"--set key '{full_key}' references nonexistent key '{part}'.")
        node[part] = value
        return
    raise ConfigError(f"--set key '{full_key}' cannot set leaf on a scalar at '{part}'.")


def apply_overrides(config, override_args):
    config = copy.deepcopy(config)
    for arg in override_args or []:
        key, value = parse_set_arg(arg)
        apply_set_override(config, key, value)
    return config


def resolve_config(path, override_args=None):
    config = load_and_merge_base(path)
    config = apply_overrides(config, override_args)
    return config


def require(condition, message):
    if not condition:
        raise ConfigError(message)


def validate_config(config, check_paths=True, check_registry=True, check_cuda=True):
    # 1. top-level keys
    keys = set(config.keys())
    missing = TOP_LEVEL_KEYS - keys
    unknown = keys - TOP_LEVEL_KEYS
    require(not missing, f"Config is missing required top-level keys: {sorted(missing)}")
    require(not unknown, f"Config has undefined top-level keys: {sorted(unknown)}")

    # 2. required subkeys and basic types
    data = config["data"]
    require(
        isinstance(data.get("image_size"), list) and len(data["image_size"]) == 2
        and all(isinstance(v, int) for v in data["image_size"]),
        "data.image_size must be a list of two integers [H, W].",
    )
    require(isinstance(data.get("batch_size"), int), "data.batch_size must be an integer.")
    require("split" in data and "mode" in data["split"], "data.split.mode is required.")
    require(data["split"]["mode"] in ("file", "ratio"), "data.split.mode must be 'file' or 'ratio'.")

    train = config["train"]
    require(isinstance(train.get("epochs"), int), "train.epochs must be an integer.")
    require("monitor" in train and "metric" in train["monitor"] and "mode" in train["monitor"],
            "train.monitor.metric and train.monitor.mode are required.")
    require(train["monitor"]["mode"] in ("max", "min"), "train.monitor.mode must be 'max' or 'min'.")

    # 3. {name, params} registry membership
    if check_registry:
        require_named(config["data"].get("name"), DATASETS, "data.name")
        require_named(config["model"].get("name"), MODELS, "model.name")
        require_named(config["loss"].get("name"), LOSSES, "loss.name")
        require_named(config["adapter"].get("name"), ADAPTERS, "adapter.name")
        for i, metric in enumerate(config.get("metrics", [])):
            require_named(metric.get("name"), METRICS, f"metrics.{i}.name")
        transform = config["data"].get("transform", {})
        for split_name in ("train", "eval"):
            if split_name in transform:
                require_named(transform[split_name].get("name"), TRANSFORMS,
                               f"data.transform.{split_name}.name")

    # 4. path existence
    if check_paths:
        require(os.path.exists(data["root"]), f"data.root does not exist: {data['root']}")
        if data["split"]["mode"] == "file":
            require(os.path.isfile(data["split"]["path"]),
                    f"data.split.path does not exist: {data['split']['path']}")
        weights_path = config["model"].get("params", {}).get("weights_path")
        if weights_path is not None:
            require(os.path.isfile(weights_path),
                    f"model.params.weights_path does not exist: {weights_path}")

    # 5. monitor metric must be declared
    metric_names = [m["name"] for m in config.get("metrics", [])]
    require(train["monitor"]["metric"] in metric_names,
            f"train.monitor.metric '{train['monitor']['metric']}' is not in metrics {metric_names}.")

    # 6. device availability
    if check_cuda and config["runtime"].get("device") == "cuda":
        require(torch.cuda.is_available(), "runtime.device is 'cuda' but torch.cuda.is_available() is False.")

    return config


def require_named(name, registry, field_label):
    require(name is not None, f"{field_label} is required.")
    require(name in registry.entries,
            f"{field_label} '{name}' is not registered in namespace '{registry.namespace}'. "
            f"Available: {registry.keys()}")
