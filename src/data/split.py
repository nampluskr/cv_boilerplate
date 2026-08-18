import json

import numpy as np


class SplitError(Exception):
    pass


def assert_disjoint(split_dict):
    sets = {name: set(split_dict[name]) for name in ("train", "valid", "test")}
    pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]
    for name1, name2 in pairs:
        overlap = sets[name1] & sets[name2]
        if overlap:
            examples = sorted(overlap)[:5]
            raise SplitError(
                f"split '{name1}' and '{name2}' overlap on {len(overlap)} ids, "
                f"examples={examples}"
            )


def assert_subset(split_dict, all_ids):
    union = set(split_dict["train"]) | set(split_dict["valid"]) | set(split_dict["test"])
    unknown = union - set(all_ids)
    if unknown:
        raise SplitError(f"split references unknown ids: {sorted(unknown)[:5]}")


def load_split_file(path):
    with open(path, "r", encoding="utf-8") as f:
        split_dict = json.load(f)
    assert_disjoint(split_dict)
    return split_dict


def generate_ratio_split(all_ids, ratio, seed, stratify_by=None):
    ids = list(all_ids)
    rng = np.random.default_rng(seed)

    if stratify_by is None:
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * ratio["train"])
        n_valid = int(n * ratio["valid"])
        split_dict = {
            "train": ids[:n_train],
            "valid": ids[n_train:n_train + n_valid],
            "test": ids[n_train + n_valid:],
        }
    else:
        groups = {}
        for sample_id, group_key in zip(ids, stratify_by):
            groups.setdefault(group_key, []).append(sample_id)
        split_dict = {"train": [], "valid": [], "test": []}
        for group_ids in groups.values():
            group_ids = list(group_ids)
            rng.shuffle(group_ids)
            n = len(group_ids)
            n_train = int(n * ratio["train"])
            n_valid = int(n * ratio["valid"])
            split_dict["train"].extend(group_ids[:n_train])
            split_dict["valid"].extend(group_ids[n_train:n_train + n_valid])
            split_dict["test"].extend(group_ids[n_train + n_valid:])

    assert_disjoint(split_dict)
    return split_dict
