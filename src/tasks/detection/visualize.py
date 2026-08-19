import os

import torch
from PIL import Image, ImageDraw

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Fixed per-label colors (PLAN-P4 SS7.1): label 1=cat, label 2=dog. Any further foreground
# label (a future non-oxford_pets dataset with more classes) falls back to a deterministic
# palette entry instead of raising, so the visualizer stays usable with num_classes > 3.
FIXED_CLASS_COLORS = {1: (0, 160, 255), 2: (255, 160, 0)}
FALLBACK_PALETTE = [
    (160, 0, 255), (0, 200, 120), (200, 200, 0), (200, 0, 120), (0, 120, 200),
]
GT_COLOR = (255, 255, 255)


def color_for_label(label):
    if label in FIXED_CLASS_COLORS:
        return FIXED_CLASS_COLORS[label]
    return FALLBACK_PALETTE[label % len(FALLBACK_PALETTE)]


def denormalize(image):
    image = image.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return image.clamp(0, 1)


def tensor_to_pil(image):
    array = (denormalize(image) * 255.0).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def draw_dashed_rectangle(draw, box, color, dash_len=6, width=2):
    """PIL has no native dashed-rectangle primitive; draw each of the 4 sides as a run of short
    line segments (PLAN-P4 SS7.1: GT boxes are drawn as white dashed rectangles)."""
    x1, y1, x2, y2 = box
    edges = [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]
    for (sx, sy), (ex, ey) in edges:
        length = max(abs(ex - sx), abs(ey - sy))
        steps = max(1, int(length // dash_len))
        for step in range(0, steps, 2):
            t0 = step / steps
            t1 = min(1.0, (step + 1) / steps)
            px0, py0 = sx + (ex - sx) * t0, sy + (ey - sy) * t0
            px1, py1 = sx + (ex - sx) * t1, sy + (ey - sy) * t1
            draw.line([(px0, py0), (px1, py1)], fill=color, width=width)


def draw_predictions(image, boxes, scores, labels, class_names):
    draw = ImageDraw.Draw(image)
    for box, score, label in zip(boxes, scores, labels):
        color = color_for_label(int(label))
        draw.rectangle(box, outline=color, width=2)
        name = class_names[label] if 0 <= label < len(class_names) else str(label)
        draw.text((box[0] + 2, max(0, box[1] - 12)), f"{name} {score:.2f}", fill=color)
    return image


def draw_ground_truth(image, boxes, labels):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw_dashed_rectangle(draw, box, GT_COLOR)
    return image


def save_detection_grid(images, targets, predictions, class_names, output_dir, max_items):
    """Save one annotated image per sample: predicted boxes (fixed color, solid) plus GT boxes
    (white, dashed) drawn on top (PLAN-P4 SS7.1). Images with zero predictions are saved too, so
    an empty result is visible rather than silently skipped (PLAN-P4 SS7.1/SS9.2)."""
    os.makedirs(output_dir, exist_ok=True)
    num_items = min(len(images), len(predictions), max_items)

    for i in range(num_items):
        tile = tensor_to_pil(images[i])
        prediction = predictions[i]
        tile = draw_predictions(tile, prediction["boxes"], prediction["scores"], prediction["labels"],
                                 class_names)
        if targets is not None:
            gt_boxes = targets[i]["boxes"].tolist() if hasattr(targets[i]["boxes"], "tolist") \
                else targets[i]["boxes"]
            gt_labels = targets[i]["labels"].tolist() if hasattr(targets[i]["labels"], "tolist") \
                else targets[i]["labels"]
            tile = draw_ground_truth(tile, gt_boxes, gt_labels)

        num_detections = len(prediction["boxes"])
        prefix = "empty_" if num_detections == 0 else ""
        tile.save(os.path.join(output_dir, f"{prefix}{i:03d}.png"))
