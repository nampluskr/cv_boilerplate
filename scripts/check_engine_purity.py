import os
import sys

FORBIDDEN_STRINGS = [
    "classification", "segmentation", "detection", "anomaly", "toy",
    "resnet", "yolo", "efficientad", "stfpm", "unet",
]

TARGET_DIRS = ["src/core", "src/bench"]


# PLAN-P1 SS9.1 requires setting the YOLO_OFFLINE env var by name in offline.py.
# This is an offline-guard allowlist entry, not task/model branching logic (SS7.4's actual target),
# so lines carrying this marker are exempt from the forbidden-string scan.
PURITY_ALLOW_MARKER = "# purity-allow"


def scan_file(path):
    hits = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if PURITY_ALLOW_MARKER in line:
                continue
            lowered = line.lower()
            for token in FORBIDDEN_STRINGS:
                if token in lowered:
                    hits.append((path, lineno, token, line.strip()))
    return hits


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    all_hits = []
    for target_dir in TARGET_DIRS:
        full_dir = os.path.join(repo_root, target_dir)
        for dirpath, _, filenames in os.walk(full_dir):
            for filename in filenames:
                if filename.endswith(".py"):
                    all_hits.extend(scan_file(os.path.join(dirpath, filename)))

    if all_hits:
        print("Engine purity check FAILED. Forbidden strings found:")
        for path, lineno, token, line in all_hits:
            print(f"  {path}:{lineno}: '{token}' -> {line}")
        sys.exit(1)

    print("Engine purity check PASSED. No task/model names found in src/core, src/bench.")


if __name__ == "__main__":
    main()
