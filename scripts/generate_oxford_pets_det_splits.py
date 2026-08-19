"""Generate the detection-specific split files for oxford_pets (PLAN-P4 SS3.2, SS3.3).

Detection cannot reuse the canonical oxford_pets.json split (PLAN-P2 SS3.2): test.txt has zero
XML annotations, so a detection test split built from the canonical file would have no boxes to
evaluate against. This script instead builds a self-contained split over the population that
actually has Pascal VOC XML annotations.

Usage (from repo root, pytorch_env activated):
    python scripts/generate_oxford_pets_det_splits.py

Produces:
    configs/splits/oxford_pets_det.json         -- full split (PLAN-P4 SS3.2)
    configs/splits/oxford_pets_subset_det.json   -- deterministic subset (PLAN-P4 SS3.3)
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
RATIO = {"train": 0.70, "valid": 0.15, "test": 0.15}
CREATED_AT = "2026-08-19"

SUBSET_PER_CLASS = {"train": 8, "valid": 2, "test": 2}

NOTE = (
    "Detection uses its own split, independent from configs/splits/oxford_pets.json (PLAN-P4 "
    "SS3.2): test.txt has 0 XML annotations among its 3,669 ids, so the canonical split cannot "
    "provide a boxed test set for detection. Population here is trainval.txt intersected with "
    "annotations/xmls (3,671 ids); 15 further xml ids absent from list.txt are excluded because "
    "they have no breed CLASS-ID to stratify on. As a result some Detection test images overlap "
    "Classification train images; this is allowed because AC-10/NFR-03 only require disjointness "
    "within one task's own split, not across tasks (OUT-01)."
)


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


def list_xml_stems(xmls_dir):
    return set(os.path.splitext(f)[0] for f in os.listdir(xmls_dir) if f.endswith(".xml"))


def stratified_ratio_split(population_ids, stem_to_class, seed, ratio):
    """Stratify by breed CLASS-ID with a per-class RNG stream seeded from a single generator
    (mirrors scripts/generate_oxford_pets_splits.py's approach, PLAN-P2 SS3.2)."""
    groups = {}
    for stem in population_ids:
        groups.setdefault(stem_to_class[stem], []).append(stem)

    rng = np.random.default_rng(seed)
    train_ids, valid_ids, test_ids = [], [], []
    for class_id in sorted(groups):
        ids = sorted(groups[class_id])
        shuffled = list(ids)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(round(n * ratio["train"]))
        n_valid = int(round(n * ratio["valid"]))
        train_ids.extend(shuffled[:n_train])
        valid_ids.extend(shuffled[n_train:n_train + n_valid])
        test_ids.extend(shuffled[n_train + n_valid:])
    return sorted(train_ids), sorted(valid_ids), sorted(test_ids)


def take_deterministic_subset(ids_pool, stem_to_class, per_class_n):
    """Sort IDs lexicographically then take the first per_class_n per class (PLAN-P4 SS3.3,
    same approach as PLAN-P2 SS3.3). Raises if any class has fewer than per_class_n samples."""
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
    xmls_dir = os.path.join(DATASET_ROOT, "annotations", "xmls")

    stem_to_class = read_list_txt(list_path)
    trainval_ids = set(read_id_list(trainval_path))
    xml_stems = list_xml_stems(xmls_dir)

    population = sorted((trainval_ids & xml_stems) & set(stem_to_class.keys()))
    excluded_no_class = sorted((trainval_ids & xml_stems) - set(stem_to_class.keys()))
    print(f"population (trainval ^ xml ^ list.txt) = {len(population)}; "
          f"excluded (xml id absent from list.txt) = {len(excluded_no_class)}")

    train_ids, valid_ids, test_ids = stratified_ratio_split(population, stem_to_class, SEED, RATIO)
    if len(train_ids) + len(valid_ids) + len(test_ids) != len(population):
        raise RuntimeError(
            f"train+valid+test={len(train_ids) + len(valid_ids) + len(test_ids)} does not match "
            f"population={len(population)}"
        )

    full_split = {
        "dataset": "oxford_pets_det",
        "created_at": CREATED_AT,
        "source": (
            "trainval.txt intersected with annotations/xmls, stratified by list.txt CLASS-ID, "
            "70/15/15 (PLAN-P4 SS3.2)"
        ),
        "seed": SEED,
        "note": NOTE,
        "train": train_ids,
        "valid": valid_ids,
        "test": test_ids,
    }
    assert_disjoint(full_split)
    assert_subset(full_split, all_ids=population)

    full_path = os.path.join(REPO_ROOT, "configs", "splits", "oxford_pets_det.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(full_split, f, indent=2)
    print(f"wrote {full_path}: train={len(train_ids)} valid={len(valid_ids)} test={len(test_ids)}")

    subset_split = {
        "dataset": "oxford_pets_det",
        "created_at": CREATED_AT,
        "source": "deterministic subset of configs/splits/oxford_pets_det.json (PLAN-P4 SS3.3)",
        "seed": SEED,
        "note": NOTE,
        "train": take_deterministic_subset(train_ids, stem_to_class, SUBSET_PER_CLASS["train"]),
        "valid": take_deterministic_subset(valid_ids, stem_to_class, SUBSET_PER_CLASS["valid"]),
        "test": take_deterministic_subset(test_ids, stem_to_class, SUBSET_PER_CLASS["test"]),
    }
    assert_disjoint(subset_split)
    assert_subset(subset_split, all_ids=population)

    subset_path = os.path.join(REPO_ROOT, "configs", "splits", "oxford_pets_subset_det.json")
    with open(subset_path, "w", encoding="utf-8") as f:
        json.dump(subset_split, f, indent=2)
    print(
        f"wrote {subset_path}: train={len(subset_split['train'])} "
        f"valid={len(subset_split['valid'])} test={len(subset_split['test'])}"
    )


if __name__ == "__main__":
    main()
