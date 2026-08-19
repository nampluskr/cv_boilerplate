import os


def check_checkpoint(checkpoint_path, repro_command):
    """Verify a v0.1 checkpoint exists before a notebook tries to load it.

    outputs/ is gitignored, so a fresh clone or a different environment has no checkpoints.
    Rather than silently falling back to a randomly initialized model, this raises with the
    exact CLI command that reproduces the missing benchmark run.
    """
    if not os.path.isfile(checkpoint_path):
        message = (
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Reproduce it with:\n"
            f"    conda activate pytorch_env\n"
            f"    {repro_command}\n"
        )
        raise FileNotFoundError(message)
    return checkpoint_path


def check_leaderboard(leaderboard_csv, repro_command):
    """Same guidance as check_checkpoint, for a missing leaderboard.csv."""
    if not os.path.isfile(leaderboard_csv):
        message = (
            f"Leaderboard not found: {leaderboard_csv}\n"
            f"Reproduce it with:\n"
            f"    conda activate pytorch_env\n"
            f"    {repro_command}\n"
        )
        raise FileNotFoundError(message)
    return leaderboard_csv
