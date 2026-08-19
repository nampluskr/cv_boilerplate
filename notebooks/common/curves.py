import pandas as pd
import matplotlib.pyplot as plt


def plot_training_curves(metrics_epoch_csv, metric_names):
    """Plot train/valid loss and the given metric columns over epoch, reading directly from
    metrics_epoch.csv as written by src/core/logger.MetricsCsvWriter. Returns the loaded
    DataFrame so the notebook can inspect raw numbers too."""
    df = pd.read_csv(metrics_epoch_csv)
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
