import matplotlib.pyplot as plt


def plot_distribution(counter, title=None, xlabel=None, ylabel="count", rotate=45):
    """Bar chart from a dict/Counter of {label: count}, sorted by key."""
    keys = sorted(counter.keys())
    values = [counter[k] for k in keys]
    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 0.3), 4))
    ax.bar([str(k) for k in keys], values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=rotate, ha="right")
    fig.tight_layout()
    plt.show()


def plot_histogram(values, bins=30, title=None, xlabel=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    plt.show()


def show_image_grid(images, titles=None, ncols=4, title=None):
    """images: list of PIL.Image (or anything matplotlib.imshow accepts)."""
    nrows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
            if titles:
                ax.set_title(str(titles[i]), fontsize=8)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()
