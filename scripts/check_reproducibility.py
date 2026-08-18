import argparse
import json
import sys

RATIO_METRIC_TOL_ABS = 1e-3
LOSS_TOL_REL = 1e-2
LOSS_TOL_ABS = 1e-4

# PLAN-P1 SS5.2: fps is a wall-clock measurement, excluded from every reproducibility judgement.
EXCLUDED_FIELDS = {"fps"}
# Exact-match fields per SS5.2 (params/FLOPs must match bit-for-bit, not within a ratio tolerance).
EXACT_FIELDS = {"params_total", "params_trainable", "flops_g"}


def compare_metric(name, value_a, value_b):
    leaf = name.rsplit(".", 1)[-1]
    diff = abs(value_a - value_b)
    if leaf in EXACT_FIELDS:
        return value_a == value_b
    if "loss" in leaf:
        rel = diff / max(abs(value_a), 1e-8)
        return rel <= LOSS_TOL_REL or diff <= LOSS_TOL_ABS
    return diff <= RATIO_METRIC_TOL_ABS


def compare(path_a, path_b):
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    mismatches = []

    def walk(prefix, node_a, node_b):
        leaf = prefix.rsplit(".", 1)[-1]
        if leaf in EXCLUDED_FIELDS:
            return
        if isinstance(node_a, dict) or isinstance(node_b, dict):
            if not (isinstance(node_a, dict) and isinstance(node_b, dict)):
                mismatches.append((prefix, node_a, node_b))
                return
            keys = set(node_a) | set(node_b)
            for key in sorted(keys):
                if key not in node_a or key not in node_b:
                    mismatches.append((f"{prefix}.{key}", node_a.get(key, "<missing>"),
                                        node_b.get(key, "<missing>")))
                    continue
                walk(f"{prefix}.{key}", node_a[key], node_b[key])
        elif isinstance(node_a, bool) or isinstance(node_b, bool):
            if node_a != node_b:
                mismatches.append((prefix, node_a, node_b))
        elif isinstance(node_a, (int, float)) and isinstance(node_b, (int, float)):
            if node_a is None or node_b is None:
                if node_a != node_b:
                    mismatches.append((prefix, node_a, node_b))
            elif not compare_metric(prefix, float(node_a), float(node_b)):
                mismatches.append((prefix, node_a, node_b))
        elif node_a != node_b:
            mismatches.append((prefix, node_a, node_b))

    walk("", a, b)
    return mismatches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    args = parser.parse_args()

    mismatches = compare(args.run_a, args.run_b)
    if mismatches:
        print(f"REPRODUCIBILITY FAILED between {args.run_a} and {args.run_b}:")
        for field, value_a, value_b in mismatches:
            print(f"  {field}: {value_a} vs {value_b}")
        sys.exit(1)
    print(f"REPRODUCIBILITY OK between {args.run_a} and {args.run_b}")


if __name__ == "__main__":
    main()
