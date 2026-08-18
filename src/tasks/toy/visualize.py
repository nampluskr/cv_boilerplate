import os

import torchvision.utils as vutils


def save_image_grid(images, output_dir, prefix, max_items):
    os.makedirs(output_dir, exist_ok=True)
    subset = images[:max_items]
    if len(subset) == 0:
        return
    normalized = (subset - subset.min()) / (subset.max() - subset.min() + 1e-8)
    vutils.save_image(normalized, os.path.join(output_dir, f"{prefix}.png"))
