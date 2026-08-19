import pandas as pd
import matplotlib.pyplot as plt


def load_leaderboard(leaderboard_csv):
    return pd.read_csv(leaderboard_csv)


def plot_tradeoff(df, metric_column, cost_column, label_column="split"):
    """Scatter metric_column (higher is better) against cost_column (e.g. params_total,
    flops_g, fps) for each model row of a leaderboard.csv, annotated with label_column."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df[cost_column], df[metric_column])
    for _, row in df.iterrows():
        ax.annotate(str(row[label_column]), (row[cost_column], row[metric_column]))
    ax.set_xlabel(cost_column)
    ax.set_ylabel(metric_column)
    fig.tight_layout()
    plt.show()
