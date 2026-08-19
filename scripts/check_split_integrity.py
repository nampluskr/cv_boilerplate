import glob
import json
import sys


def check_disjoint(path):
    with open(path) as f:
        data = json.load(f)
    errors = []
    keys = [k for k in ("train", "valid", "test") if k in data and isinstance(data[k], list)]
    sets = {k: set(data[k]) for k in keys}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            overlap = sets[a] & sets[b]
            if overlap:
                errors.append(f"{path}: {a} and {b} overlap on {len(overlap)} ids "
                               f"(e.g. {sorted(overlap)[:3]})")
    if "oxford_pets" in path and "det" in path and "note" not in data:
        errors.append(f"{path}: detection split missing required 'note' field (PLAN-P4 SS3.2)")
    return errors


def main():
    all_errors = []
    for path in sorted(glob.glob("configs/splits/*.json")):
        all_errors.extend(check_disjoint(path))

    if all_errors:
        print("SPLIT INTEGRITY FAILED:")
        for e in all_errors:
            print(f"  {e}")
        sys.exit(1)
    print("SPLIT INTEGRITY OK: all configs/splits/*.json train/valid/test are pairwise disjoint.")


if __name__ == "__main__":
    main()
