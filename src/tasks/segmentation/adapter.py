import os

import numpy as np
import torch
from PIL import Image

from src.core.adapter import TaskAdapter
from src.core.registry import ADAPTERS


@ADAPTERS.register("segmentation")
class SegmentationAdapter(TaskAdapter):
    """PLAN-P3 SS6. collate_fn is None: after transform every mask is a fixed (256, 256)
    LongTensor, so torch default_collate suffices."""

    def __init__(self, loss_fn, metrics, class_names=None, **params):
        super().__init__(loss_fn, metrics, **params)
        self.class_names = class_names or ["background", "pet", "boundary"]
        # Ephemeral cache set by predict_step, consumed by save_predictions/visualize in the
        # same call sequence (src/cli/commands.py always calls predict_step once per batch and
        # immediately feeds the same batch/predictions to visualize/save_predictions). Needed
        # because predict_step's own return value must stay JSON-serializable
        # ({"stem", "mask_path", "class_pixel_ratio"} per PLAN-P3 SS6): commands.py calls
        # save_json() directly on it.
        self._last_pred_masks = None

    def train_step(self, model, batch, device):
        images, masks = batch[0].to(device), batch[1].to(device)
        logits = model(images)
        loss = self.loss_fn(logits, masks)
        return {"loss": loss, "loss_dict": {"loss": float(loss.detach())}}

    def eval_step(self, model, batch, device):
        images, masks = batch[0].to(device), batch[1].to(device)
        logits = model(images)
        loss = self.loss_fn(logits, masks)
        return {"loss": loss, "outputs": {"preds": logits.argmax(dim=1), "targets": masks}}

    def update_metrics(self, outputs):
        preds = outputs["outputs"]["preds"]
        targets = outputs["outputs"]["targets"]
        for metric in self.metrics.values():
            metric.to(preds.device)
            metric.update(preds, targets)

    def compute_metrics(self):
        return {name: float(metric.compute()) for name, metric in self.metrics.items()}

    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    def batch_size(self, batch):
        return batch[0].shape[0]

    def collate_fn(self):
        return None  # dense targets have fixed shape after transform (PLAN-P3 SS6)

    def predict_step(self, model, batch, device):
        """Postprocess is argmax(dim=1) only (PLAN-P3 SS7.1). batch[1] carries a sample
        identifier: the common `predict` CLI command (src/cli/commands.py) collates
        (image, stem) pairs from input filenames; the Dataset's __getitem__ collates
        (image, mask) pairs instead, so a non-string batch[1] means no real stem is available."""
        images = batch[0].to(device)
        raw_ids = batch[1] if len(batch) > 1 else None
        logits = model(images)
        pred_masks = logits.argmax(dim=1)  # (B, H, W) long, values in [0, num_classes)
        self._last_pred_masks = pred_masks.detach().cpu()

        num_classes = len(self.class_names)
        predictions = []
        for i in range(images.shape[0]):
            mask = pred_masks[i]
            total = mask.numel()
            ratios = {
                self.class_names[c]: float((mask == c).sum()) / total
                for c in range(num_classes)
            }
            stem = (
                raw_ids[i] if isinstance(raw_ids, (list, tuple)) and isinstance(raw_ids[i], str)
                else f"sample_{i:04d}"
            )
            predictions.append({
                "stem": stem,
                "mask_path": f"{stem}.png",
                "class_pixel_ratio": ratios,
            })
        return predictions

    def save_predictions(self, predictions, output_dir):
        """Writes each predicted mask as a single-channel PNG with pixel values 0..2
        (PLAN-P3 SS6) directly under output_dir (predictions/)."""
        if self._last_pred_masks is None:
            return
        os.makedirs(output_dir, exist_ok=True)
        for i, prediction in enumerate(predictions):
            mask_array = self._last_pred_masks[i].numpy().astype(np.uint8)
            Image.fromarray(mask_array, mode="L").save(os.path.join(output_dir, prediction["mask_path"]))

    def visualize(self, batch, predictions, output_dir, max_items):
        from src.tasks.segmentation.visualize import save_comparison_grid

        images = batch[0]
        targets = batch[1] if len(batch) > 1 and torch.is_tensor(batch[1]) else None
        save_comparison_grid(images, targets, self._last_pred_masks, output_dir, max_items)
