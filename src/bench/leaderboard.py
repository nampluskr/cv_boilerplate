import os

from src.utils.io import save_csv_rows


def sort_rows(rows, monitor_metric, monitor_mode):
    def key(row):
        value = row.get(monitor_metric)
        if row.get("status") != "ok" or value in (None, ""):
            return (1, 0.0)
        sign = -1 if monitor_mode == "max" else 1
        return (0, sign * float(value))

    return sorted(rows, key=key)


def build_leaderboard(bench_dir, bench_name, rows, metric_names, monitor_metric, monitor_mode,
                       control_report):
    fieldnames = (
        ["split", "model", "status"] + metric_names +
        ["params_total", "flops_g", "fps", "best_epoch", "train_time_sec", "batch_size",
         "image_size", "seed", "control_status", "exceptions", "run_dir"]
    )
    ordered_rows = sort_rows(rows, monitor_metric, monitor_mode)

    csv_path = os.path.join(bench_dir, "leaderboard.csv")
    save_csv_rows(csv_path, ordered_rows, fieldnames)

    md_path = os.path.join(bench_dir, "leaderboard.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Leaderboard: {bench_name}\n\n")
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in ordered_rows:
            f.write("| " + " | ".join(str(row.get(name, "")) for name in fieldnames) + " |\n")
        f.write("\n## Control conditions\n\n")
        for field, values in control_report["fields"].items():
            f.write(f"- `{field}`: {values}\n")
        if control_report["exceptions_applied"]:
            f.write("\n## Approved exceptions\n\n")
            for exception in control_report["exceptions_applied"]:
                f.write(f"- {exception}\n")
        f.write(
            "\n`fps` measures model forward pass only (batch=1, warmup 10, 50 iterations), "
            "excludes data loading and postprocessing.\n"
        )
        f.write("\nv0.1's purpose is pipeline validation, not absolute performance.\n")

    return csv_path, md_path
