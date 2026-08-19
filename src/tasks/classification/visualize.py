import os

import torch
from PIL import Image, ImageDraw

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(image):
    image = image.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return image.clamp(0, 1)


def tensor_to_pil(image):
    array = (denormalize(image) * 255.0).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def class_name_of(class_names, index):
    if class_names is not None and 0 <= index < len(class_names):
        return class_names[index]
    return str(index)


def save_prediction_grid(images, targets, predictions, class_names, output_dir, max_items):
    """Save one annotated tile per sample: GT breed, predicted breed, confidence.
    Misclassified tiles are prefixed with '[X]' in the filename (no emoji, CON-14)."""
    os.makedirs(output_dir, exist_ok=True)
    num_items = min(images.shape[0], len(predictions), max_items)

    for i in range(num_items):
        tile = tensor_to_pil(images[i]).resize((224, 224))
        pred_index = predictions[i]["pred_index"]
        pred_name = predictions[i]["pred_class"]
        confidence = predictions[i]["confidence"]

        gt_index = int(targets[i]) if targets is not None else None
        gt_name = class_name_of(class_names, gt_index) if gt_index is not None else "unknown"
        is_correct = gt_index is None or gt_index == pred_index

        draw = ImageDraw.Draw(tile)
        label_lines = [f"gt: {gt_name}", f"pred: {pred_name} ({confidence:.2f})"]
        for line_index, line in enumerate(label_lines):
            draw.text((4, 4 + line_index * 12), line, fill=(255, 0, 0))

        prefix = "" if is_correct else "[X]_"
        filename = f"{prefix}{i:03d}.png"
        tile.save(os.path.join(output_dir, filename))
