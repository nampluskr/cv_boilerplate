import os


def check_artifact(path, repro_command, kind="Artifact"):
    """Verify a v0.1 output artifact (checkpoint, leaderboard, metrics log, resolved config,
    ...) exists before a notebook tries to read it.

    outputs/ is gitignored, so a fresh clone or a different environment has no benchmark
    outputs. Rather than silently falling back to a randomly initialized model or skipping a
    check, this raises with the exact CLI command that reproduces the missing benchmark run.
    """
    if not os.path.isfile(path):
        message = (
            f"{kind} not found: {path}\n"
            f"Reproduce it with:\n"
            f"    conda activate pytorch_env\n"
            f"    {repro_command}\n"
        )
        raise FileNotFoundError(message)
    return path


def check_checkpoint(checkpoint_path, repro_command):
    return check_artifact(checkpoint_path, repro_command, kind="Checkpoint")


def check_leaderboard(leaderboard_csv, repro_command):
    return check_artifact(leaderboard_csv, repro_command, kind="Leaderboard")
