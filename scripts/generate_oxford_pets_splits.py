"""Generate the canonical and subset split files for oxford_pets classification/segmentation.

Usage (from repo root, pytorch_env activated):
    python scripts/generate_oxford_pets_splits.py

Produces:
    configs/splits/oxford_pets.json             -- full split (PLAN-P2 SS3.2)
    configs/splits/oxford_pets_subset_cls.json   -- deterministic subset (PLAN-P2 SS3.3)

Sample index basis is annotations/list.txt (PLAN.md SS6.4), not the images/ directory listing.
"""

import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.data.split import assert_disjoint, assert_subset  # noqa: E402

DATASET_ROOT = "/mnt/d/datasets/oxford_pets"
SEED = 42
TRAIN_RATIO = 0.85
CREATED_AT = "2026-08-19"

SUBSET_PER_CLASS = {"train": 10, "valid": 2, "test": 2}


def read_list_txt(path):
    """Return {stem: class_id} skipping comment lines, list.txt is 1-based CLASS-ID."""
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            mapping[parts[0]] = int(parts[1])
    return mapping


def read_id_list(path):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line.split()[0])
    return ids


def stratified_train_valid_split(trainval_ids, stem_to_class, seed, train_ratio):
    """Stratify by CLASS-ID so every class contributes train_ratio to train, rest to valid.
    Uses a per-class RNG stream seeded from a single generator so the result depends only on
    (seed, class contents), not on dict iteration order.
    """
    groups = {}
    for stem in trainval_ids:
        groups.setdefault(stem_to_class[stem], []).append(stem)

    rng = np.random.default_rng(seed)
    train_ids, valid_ids = [], []
    for class_id in sorted(groups):
        ids = sorted(groups[class_id])
        shuffled = list(ids)
        rng.shuffle(shuffled)
        n_train = int(round(len(shuffled) * train_ratio))
        train_ids.extend(shuffled[:n_train])
        valid_ids.extend(shuffled[n_train:])
    return sorted(train_ids), sorted(valid_ids)


def take_deterministic_subset(ids_pool, stem_to_class, per_class_n):
    """Sort IDs lexicographically then take the first per_class_n per class (PLAN-P2 SS3.3).
    No randomness, so the result is stable regardless of seed or library version. Raises if any
    class has fewer than per_class_n samples instead of silently emitting a smaller subset
    (Codex A2 finding #1).
    """
    by_class = {}
    for stem in ids_pool:
        by_class.setdefault(stem_to_class[stem], []).append(stem)
    subset = []
    for class_id in sorted(by_class):
        names = sorted(by_class[class_id])
        if len(names) < per_class_n:
            raise RuntimeError(
                f"class {class_id} has only {len(names)} samples in this pool, "
                f"need {per_class_n} for the subset split"
            )
        subset.extend(names[:per_class_n])
    return sorted(subset)


def main():
    list_path = os.path.join(DATASET_ROOT, "annotations", "list.txt")
    trainval_path = os.path.join(DATASET_ROOT, "annotations", "trainval.txt")
    test_path = os.path.join(DATASET_ROOT, "annotations", "test.txt")

    stem_to_class = read_list_txt(list_path)
    trainval_ids = read_id_list(trainval_path)
    test_ids = sorted(read_id_list(test_path))

    train_ids, valid_ids = stratified_train_valid_split(trainval_ids, stem_to_class, SEED, TRAIN_RATIO)

    # Completeness check (Codex A2 finding #1): assert_disjoint/assert_subset only prove the
    # produced IDs are pairwise disjoint and a subset of list.txt, not that every source ID made
    # it into some split. A silently-dropped ID from a mismatched trainval.txt/test.txt would
    # otherwise materialize an under-sized split file that still passes those two checks.
    if len(train_ids) + len(valid_ids) != len(trainval_ids):
        raise RuntimeError(
            f"train+valid={len(train_ids) + len(valid_ids)} does not match "
            f"trainval.txt count={len(trainval_ids)}"
        )
    all_ids = set(stem_to_class.keys())
    covered = set(train_ids) | set(valid_ids) | set(test_ids)
    if covered != all_ids:
        missing = sorted(all_ids - covered)[:5]
        raise RuntimeError(
            f"split covers {len(covered)}/{len(all_ids)} list.txt IDs; example missing: {missing}"
        )

    full_split = {
        "dataset": "oxford_pets",
        "created_at": CREATED_AT,
        "source": "annotations/list.txt + trainval.txt/test.txt, class-stratified 85/15 (PLAN-P2 SS3.2)",
        "seed": SEED,
        "train": train_ids,
        "valid": valid_ids,
        "test": test_ids,
    }
    assert_disjoint(full_split)
    assert_subset(full_split, all_ids=list(stem_to_class.keys()))

    full_path = os.path.join(REPO_ROOT, "configs", "splits", "oxford_pets.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(full_split, f, indent=2)
    print(f"wrote {full_path}: train={len(train_ids)} valid={len(valid_ids)} test={len(test_ids)}")

    subset_split = {
        "dataset": "oxford_pets",
        "created_at": CREATED_AT,
        "source": "deterministic subset of configs/splits/oxford_pets.json (PLAN-P2 SS3.3)",
        "seed": SEED,
        "train": take_deterministic_subset(train_ids, stem_to_class, SUBSET_PER_CLASS["train"]),
        "valid": take_deterministic_subset(valid_ids, stem_to_class, SUBSET_PER_CLASS["valid"]),
        "test": take_deterministic_subset(test_ids, stem_to_class, SUBSET_PER_CLASS["test"]),
    }
    assert_disjoint(subset_split)
    assert_subset(subset_split, all_ids=list(stem_to_class.keys()))

    subset_path = os.path.join(REPO_ROOT, "configs", "splits", "oxford_pets_subset_cls.json")
    with open(subset_path, "w", encoding="utf-8") as f:
        json.dump(subset_split, f, indent=2)
    print(
        f"wrote {subset_path}: train={len(subset_split['train'])} "
        f"valid={len(subset_split['valid'])} test={len(subset_split['test'])}"
    )


if __name__ == "__main__":
    main()
