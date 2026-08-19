import torch
import torch.nn.functional as F

from src.core.adapter import TaskAdapter
from src.core.registry import ADAPTERS


@ADAPTERS.register("classification")
class ClassificationAdapter(TaskAdapter):
    """PLAN-P2 SS6. collate_fn is None: labels are scalars, so torch default_collate suffices."""

    def __init__(self, loss_fn, metrics, topk=5, class_names=None, **params):
        super().__init__(loss_fn, metrics, **params)
        self.topk = topk
        self.class_names = class_names

    def train_step(self, model, batch, device):
        images, targets = batch[0].to(device), batch[1].to(device)
        logits = model(images)
        loss = self.loss_fn(logits, targets)
        return {"loss": loss, "loss_dict": {"loss": float(loss.detach())}}

    def eval_step(self, model, batch, device):
        images, targets = batch[0].to(device), batch[1].to(device)
        logits = model(images)
        loss = self.loss_fn(logits, targets)
        return {"loss": loss, "outputs": {"logits": logits, "targets": targets}}

    def update_metrics(self, outputs):
        logits = outputs["outputs"]["logits"]
        targets = outputs["outputs"]["targets"]
        preds = logits.argmax(dim=1)
        # cli/bench now call TaskAdapter.to(device) before use (PLAN-P1 SS6.4); this per-call
        # `.to()` is a cheap no-op safety net in case an adapter is ever driven without that call.
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
        return None  # torch default_collate is sufficient (PLAN-P2 SS6)

    def predict_step(self, model, batch, device):
        """Postprocess: softmax -> argmax -> top-k, performed here per PLAN-P2 SS7.1.

        batch[1] carries a sample identifier: the common `predict` CLI command
        (src/cli/commands.py) collates (image, stem) pairs from input filenames; the Dataset's
        __getitem__ collates (image, label) pairs instead, so a non-string batch[1] (an int
        label tensor) means no real stem is available and 'stem' stays None.
        """
        images = batch[0].to(device)
        raw_ids = batch[1] if len(batch) > 1 else None
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        confidences, pred_indices = probs.max(dim=1)
        k = min(self.topk, probs.shape[1])
        topk_values, topk_indices = probs.topk(k, dim=1)

        predictions = []
        for i in range(images.shape[0]):
            pred_index = int(pred_indices[i])
            topk_list = [
                {
                    "index": int(topk_indices[i, j]),
                    "class": self.class_name(int(topk_indices[i, j])),
                    "confidence": float(topk_values[i, j]),
                }
                for j in range(k)
            ]
            stem = raw_ids[i] if isinstance(raw_ids, (list, tuple)) and isinstance(raw_ids[i], str) else None
            predictions.append({
                "stem": stem,
                "pred_index": pred_index,
                "pred_class": self.class_name(pred_index),
                "confidence": float(confidences[i]),
                "topk": topk_list,
            })
        return predictions

    def class_name(self, index):
        if self.class_names is not None and 0 <= index < len(self.class_names):
            return self.class_names[index]
        return str(index)

    def save_predictions(self, predictions, output_dir):
        # The common `predict` command (src/cli/commands.py) already writes
        # predictions/predict.json with {"paths": [...], "predictions": [...]}; nothing
        # further to add here for classification.
        pass

    def visualize(self, batch, predictions, output_dir, max_items):
        from src.tasks.classification.visualize import save_prediction_grid

        images = batch[0]
        targets = batch[1] if len(batch) > 1 else None
        save_prediction_grid(images, targets, predictions, self.class_names, output_dir, max_items)
