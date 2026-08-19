import os

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(image):
    image = image.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return image.clamp(0, 1)


def tensor_to_pil(image):
    array = (denormalize(image) * 255.0).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def mask_to_rgb(mask, color=(200, 0, 0)):
    """mask: (H, W) tensor/array with values in {0, 1} -> (H, W, 3) uint8 numpy array."""
    mask_array = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    rgb = np.zeros((*mask_array.shape, 3), dtype=np.uint8)
    rgb[mask_array > 0] = color
    return rgb


def anomaly_map_to_heatmap(anomaly_map):
    """(H, W) float Tensor -> (H, W, 3) uint8 heatmap, normalized per-image (min-max) so a
    weak model's low-magnitude map is still visible instead of reading as uniformly cold."""
    array = anomaly_map.detach().cpu().numpy().astype(np.float32)
    low, high = float(array.min()), float(array.max())
    normalized = (array - low) / (high - low) if high > low else np.zeros_like(array)
    # Simple red channel ramp (no external colormap dependency): blue->red as anomaly rises.
    heatmap = np.zeros((*array.shape, 3), dtype=np.uint8)
    heatmap[..., 0] = (normalized * 255).astype(np.uint8)
    heatmap[..., 2] = ((1 - normalized) * 255).astype(np.uint8)
    return heatmap


def save_anomaly_grid(images, gt_masks, anomaly_maps, pixel_threshold, output_dir, max_items):
    """Save one 4-panel image per sample: original | GT mask | anomaly heatmap |
    threshold-binarized map, concatenated horizontally (PLAN-P5 SS7.3). Both normal and
    anomalous samples are eligible; the caller controls which batch is passed in (evaluate's
    single-batch visualization call in src/cli/commands.py picks whichever batch it iterates
    first, so callers wanting a guaranteed mix should widen batch_size or shuffle=False on a
    mixed valid/test loader -- this function itself does not filter or reorder samples)."""
    os.makedirs(output_dir, exist_ok=True)
    num_items = images.shape[0]
    if anomaly_maps is not None:
        num_items = min(num_items, anomaly_maps.shape[0])
    num_items = min(num_items, max_items)

    for i in range(num_items):
        image_tile = tensor_to_pil(images[i])
        width, height = image_tile.size

        gt_tile = (
            Image.fromarray(mask_to_rgb(gt_masks[i])).resize((width, height), Image.NEAREST)
            if gt_masks is not None else Image.new("RGB", (width, height), (128, 128, 128))
        )

        if anomaly_maps is not None:
            heatmap_tile = Image.fromarray(anomaly_map_to_heatmap(anomaly_maps[i])).resize(
                (width, height), Image.NEAREST)
            threshold = pixel_threshold if pixel_threshold is not None else 0.0
            binarized = (anomaly_maps[i].detach().cpu().numpy() >= threshold)
            binarized_tile = Image.fromarray(mask_to_rgb(binarized)).resize((width, height), Image.NEAREST)
        else:
            heatmap_tile = Image.new("RGB", (width, height), (128, 128, 128))
            binarized_tile = Image.new("RGB", (width, height), (128, 128, 128))

        combined = Image.new("RGB", (width * 4, height))
        combined.paste(image_tile, (0, 0))
        combined.paste(gt_tile, (width, 0))
        combined.paste(heatmap_tile, (width * 2, 0))
        combined.paste(binarized_tile, (width * 3, 0))
        combined.save(os.path.join(output_dir, f"{i:03d}.png"))
