import argparse


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m src")
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_assets = subparsers.add_parser("check-assets")
    check_assets.add_argument("--assets", default="configs/assets.yaml")

    config_cmd = subparsers.add_parser("config")
    config_cmd.add_argument("config_path")
    config_cmd.add_argument("--set", dest="set", action="append", default=[])

    train_cmd = subparsers.add_parser("train")
    train_cmd.add_argument("config_path")
    train_cmd.add_argument("--set", dest="set", action="append", default=[])
    train_cmd.add_argument("--resume", default=None)

    evaluate_cmd = subparsers.add_parser("evaluate")
    evaluate_cmd.add_argument("config_path")
    evaluate_cmd.add_argument("--set", dest="set", action="append", default=[])
    evaluate_cmd.add_argument("--checkpoint", required=True)
    evaluate_cmd.add_argument("--split", default="test")

    predict_cmd = subparsers.add_parser("predict")
    predict_cmd.add_argument("config_path")
    predict_cmd.add_argument("--set", dest="set", action="append", default=[])
    predict_cmd.add_argument("--checkpoint", required=True)
    predict_cmd.add_argument("--input", required=True)
    predict_cmd.add_argument("--output", default=None)

    benchmark_cmd = subparsers.add_parser("benchmark")
    benchmark_cmd.add_argument("bench_path")
    benchmark_cmd.add_argument("--set", dest="set", action="append", default=[])
    benchmark_cmd.add_argument("--only", default=None)
    benchmark_cmd.add_argument("--overwrite", action="store_true")

    leaderboard_cmd = subparsers.add_parser("leaderboard")
    leaderboard_cmd.add_argument("bench_output_dir")

    return parser
