import os
import time


def default_run_name(config_path):
    stem = os.path.splitext(os.path.basename(config_path))[0]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{stem}__{timestamp}"


def build_run_dir(output_root, task_name, run_name, config_path):
    run_name = run_name or default_run_name(config_path)
    return os.path.join(output_root, "runs", task_name, run_name)


def build_benchmark_split_dir(output_root, bench_name, split_name):
    return os.path.join(output_root, "benchmarks", bench_name, "splits", split_name)


def build_benchmark_dir(output_root, bench_name):
    return os.path.join(output_root, "benchmarks", bench_name)
