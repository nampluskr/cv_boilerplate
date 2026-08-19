import os

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Fixed color palette (PLAN-P3 SS7.2): background=black, pet=green, boundary=red.
CLASS_COLORS = {
    0: (0, 0, 0),
    1: (0, 200, 0),
    2: (200, 0, 0),
}


def denormalize(image):
    image = image.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return image.clamp(0, 1)


def tensor_to_pil(image):
    array = (denormalize(image) * 255.0).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def mask_to_rgb(mask):
    """mask: (H, W) LongTensor with values in [0, 2] -> (H, W, 3) uint8 numpy array."""
    mask_array = mask.detach().cpu().numpy()
    rgb = np.zeros((*mask_array.shape, 3), dtype=np.uint8)
    for class_index, color in CLASS_COLORS.items():
        rgb[mask_array == class_index] = color
    return rgb


def save_comparison_grid(images, targets, pred_masks, output_dir, max_items):
    """Save one image per sample: original | GT mask | predicted mask, concatenated
    horizontally (PLAN-P3 SS7.2)."""
    os.makedirs(output_dir, exist_ok=True)
    num_items = images.shape[0]
    if pred_masks is not None:
        num_items = min(num_items, pred_masks.shape[0])
    num_items = min(num_items, max_items)

    for i in range(num_items):
        image_tile = tensor_to_pil(images[i])
        width, height = image_tile.size

        gt_tile = (
            Image.fromarray(mask_to_rgb(targets[i])).resize((width, height), Image.NEAREST)
            if targets is not None else Image.new("RGB", (width, height), (128, 128, 128))
        )
        pred_tile = (
            Image.fromarray(mask_to_rgb(pred_masks[i])).resize((width, height), Image.NEAREST)
            if pred_masks is not None else Image.new("RGB", (width, height), (128, 128, 128))
        )

        combined = Image.new("RGB", (width * 3, height))
        combined.paste(image_tile, (0, 0))
        combined.paste(gt_tile, (width, 0))
        combined.paste(pred_tile, (width * 2, 0))
        combined.save(os.path.join(output_dir, f"{i:03d}.png"))
