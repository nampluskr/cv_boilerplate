import pandas as pd
import matplotlib.pyplot as plt


def _last_run(df):
    """metrics_epoch.csv (src/core/logger.MetricsCsvWriter) opens in append mode, so a
    benchmark split re-run for verification (as v0.1's did repeatedly across P3-P6) leaves
    several past runs' rows stacked in one file, each restarting at epoch 1. Keep only the
    rows from the last run (the one whose weights are actually in checkpoints/best.pth) so a
    training curve does not fold back on itself epoch 5 -> 1 -> 5 -> 1 ...."""
    train_epoch_one = df.index[(df["split"] == "train") & (df["epoch"] == 1)]
    if len(train_epoch_one) == 0:
        return df
    return df.loc[train_epoch_one[-1]:]


def plot_training_curves(metrics_epoch_csv, metric_names):
    """Plot train/valid loss and the given metric columns over epoch, reading directly from
    metrics_epoch.csv as written by src/core/logger.MetricsCsvWriter. Returns the loaded
    DataFrame (last run only) so the notebook can inspect raw numbers too."""
    df = _last_run(pd.read_csv(metrics_epoch_csv))
    columns = ["loss"] + [name for name in metric_names if name in df.columns]
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4))
    if len(columns) == 1:
        axes = [axes]
    for ax, column in zip(axes, columns):
        for split in ("train", "valid"):
            subset = df[df["split"] == split]
            if column in subset.columns and subset[column].notna().any():
                ax.plot(subset["epoch"], subset[column], label=split, marker="o")
        ax.set_xlabel("epoch")
        ax.set_ylabel(column)
        ax.legend()
    fig.tight_layout()
    plt.show()
    return df
