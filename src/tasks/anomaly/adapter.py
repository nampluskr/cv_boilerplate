import torch

from src.core.adapter import TaskAdapter
from src.core.registry import ADAPTERS
from src.tasks.anomaly.collate import anomaly_collate
from src.tasks.anomaly.postprocess import compute_thresholds, smooth_anomaly_map


@ADAPTERS.register("anomaly")
class AnomalyAdapter(TaskAdapter):
    """PLAN-P5 SS7.2. Loss lives inside each model's train_step (SS4.1); eval_step's loss is
    always None -- the three models' loss definitions are not comparable across architectures
    (reconstruction MSE vs. teacher-student feature MSE vs. EfficientAD's compound loss), so
    model selection uses image_auroc (train.monitor), never valid loss."""

    def __init__(self, loss_fn, metrics, smooth_sigma=4.0, **params):
        super().__init__(loss_fn, metrics, **params)
        self.smooth_sigma = smooth_sigma
        self.image_threshold = None
        self.pixel_threshold = None
        # Ephemeral cache set by predict_step, consumed by visualize in the same call sequence
        # (src/cli/commands.py's evaluate() always calls predict_step once per batch and
        # immediately feeds the same batch/predictions to visualize), mirroring
        # SegmentationAdapter._last_pred_masks.
        self._last_maps = None

    def collate_fn(self):
        return anomaly_collate

    def batch_size(self, batch):
        return batch[0].shape[0]

    def _smooth(self, anomaly_map):
        return smooth_anomaly_map(anomaly_map, self.smooth_sigma)

    def train_step(self, model, batch, device):
        images = batch[0].to(device)
        return model.train_step(images, batch[1])  # targets are empty dicts, unused

    def eval_step(self, model, batch, device):
        images = batch[0].to(device)
        targets = batch[1]
        outputs = model(images)
        labels = torch.stack([t["label"] for t in targets]).to(device)
        masks = torch.stack([t["mask"] for t in targets]).to(device)
        maps = self._smooth(outputs["anomaly_map"])
        return {
            "loss": None,
            "outputs": {
                "scores": outputs["pred_score"],
                "maps": maps,
                "labels": labels,
                "masks": masks,
            },
        }

    def update_metrics(self, outputs):
        o = outputs["outputs"]
        if "image_auroc" in self.metrics:
            self.metrics["image_auroc"].update(o["scores"], o["labels"])
        if "pixel_auroc" in self.metrics:
            self.metrics["pixel_auroc"].update(o["maps"].flatten(), o["masks"].flatten().long())

    def compute_metrics(self):
        return {name: float(metric.compute()) for name, metric in self.metrics.items()}

    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    def predict_step(self, model, batch, device):
        images = batch[0].to(device)
        raw_ids = batch[1] if len(batch) > 1 else None
        outputs = model(images)
        maps = self._smooth(outputs["anomaly_map"])
        self._last_maps = maps.detach().cpu()

        predictions = []
        for i in range(images.shape[0]):
            score = float(outputs["pred_score"][i])
            is_anomalous = (
                bool(score >= self.image_threshold) if self.image_threshold is not None else None
            )
            stem = (
                raw_ids[i] if isinstance(raw_ids, (list, tuple)) and isinstance(raw_ids[i], str)
                else f"sample_{i:04d}"
            )
            predictions.append({
                "stem": stem,
                "anomaly_score": score,
                "is_anomalous": is_anomalous,
                "image_threshold": self.image_threshold,
            })
        return predictions

    def save_predictions(self, predictions, output_dir):
        # The common `predict` command (src/cli/commands.py) already writes
        # predictions/predict.json with {"paths": [...], "predictions": [...]}; nothing further
        # to add here (anomaly_map is only used for visualization, not a persisted prediction
        # artifact, mirroring DetectionAdapter.save_predictions).
        pass

    def on_fit_start(self, model, loaders, device):
        # loaders only contains {"train", "valid"} (PLAN-P5 SS5) -- no test leakage possible here.
        if hasattr(model, "on_fit_start"):
            model.on_fit_start(loaders["train"], device)

    def on_fit_end(self, model, loaders, device):
        if hasattr(model, "on_fit_end"):
            model.on_fit_end(loaders["valid"], device)
        # valid-only threshold decision (PLAN-P5 SS6, NFR-03).
        self.image_threshold, self.pixel_threshold = compute_thresholds(
            model, loaders["valid"], device, self.smooth_sigma)

    def extra_final_metrics(self):
        # PLAN-P5 SS6: persist the valid-only threshold decision into metrics_final.json.
        # None until on_fit_end has run (e.g. train.epochs=0 or checkpoint_dir disabled).
        return {"image_threshold": self.image_threshold, "pixel_threshold": self.pixel_threshold}

    def visualize(self, batch, predictions, output_dir, max_items):
        from src.tasks.anomaly.visualize import save_anomaly_grid

        images = batch[0]
        targets = batch[1] if len(batch) > 1 else None
        has_masks = isinstance(targets, list) and targets and "mask" in targets[0]
        gt_masks = torch.stack([t["mask"] for t in targets]) if has_masks else None
        save_anomaly_grid(images, gt_masks, self._last_maps, self.pixel_threshold, output_dir, max_items)
