import glob
import os
import tempfile

import matplotlib.pyplot as plt
from PIL import Image


def show_predictions(adapter, batch, predictions, max_items=4, title=None):
    """Render task-specific prediction overlays by calling the task's own TaskAdapter.visualize
    (src/tasks/<task>/adapter.py), which in turn calls src/tasks/<task>/visualize.py -- the same
    rendering code v0.1 uses for benchmark output tiles. This module only loads the resulting
    PNGs from a temp directory and lays them out inline; it does not draw boxes, masks, or
    heatmaps itself, so the notebook never shows a different rendering than the CLI does.
    """
    with tempfile.TemporaryDirectory() as output_dir:
        adapter.visualize(batch, predictions, output_dir, max_items)
        paths = sorted(glob.glob(os.path.join(output_dir, "*.png")))
        if not paths:
            print("visualize() produced no images for this batch.")
            return
        fig, axes = plt.subplots(1, len(paths), figsize=(5 * len(paths), 5))
        if len(paths) == 1:
            axes = [axes]
        for ax, path in zip(axes, paths):
            ax.imshow(Image.open(path))
            ax.set_title(os.path.basename(path), fontsize=8)
            ax.axis("off")
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        plt.show()
