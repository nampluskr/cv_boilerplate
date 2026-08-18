import sys

# Offline guard must be armed before any module that might touch the network at import time
# (torch, torchvision, task packages, ...) is imported. src.core.offline has no heavy dependencies,
# so this import+call pair is safe to run first (Codex A1 finding #8).
from src.core.offline import enable_offline_guard

enable_offline_guard()

import src.tasks  # noqa: F401,E402  populate registries after the guard is armed
from src.cli.parser import build_parser  # noqa: E402
from src.cli.commands import (  # noqa: E402
    build_leaderboard, check_assets, evaluate, predict, print_resolved_config, run_benchmark, train,
)
from src.core.errors import (  # noqa: E402
    ConfigError, ControlViolationError, LocalAssetError, OfflineViolationError, RegistryError,
)

USER_FACING_ERRORS = (
    ConfigError, RegistryError, LocalAssetError, OfflineViolationError, ControlViolationError,
)


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        dispatch(args)
    except USER_FACING_ERRORS as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)


def dispatch(args):
    if args.command == "check-assets":
        check_assets(args.assets)
    elif args.command == "config":
        print_resolved_config(args.config_path, args.set)
    elif args.command == "train":
        train(args.config_path, args.set, args.log_level, resume=args.resume)
    elif args.command == "evaluate":
        evaluate(args.config_path, args.set, args.log_level, args.checkpoint, args.split)
    elif args.command == "predict":
        predict(args.config_path, args.set, args.log_level, args.checkpoint, args.input, args.output)
    elif args.command == "benchmark":
        run_benchmark(args.bench_path, args.set, args.log_level, args.only, args.overwrite)
    elif args.command == "leaderboard":
        build_leaderboard(args.bench_output_dir)
    else:
        parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
