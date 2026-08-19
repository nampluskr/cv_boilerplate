from torchmetrics.detection import MeanAveragePrecision

from src.core.registry import METRICS

# Registered under a distinct key from src/tasks/toy/metric.py's "map" (same underlying
# torchmetrics class, different fixed construction args per PLAN-P4 SS6) so both fixtures can
# coexist in the global METRICS namespace without a RegistryError.


@METRICS.register("map_50_95")
def build_map_50_95(box_format="xyxy", **params):
    # backend="faster_coco_eval" per PLAN-P4 SS6 / CON-08 (2026-08-18 approval). iou_type="bbox"
    # is the torchmetrics default for this backend and is set explicitly for clarity.
    return MeanAveragePrecision(box_format=box_format, iou_type="bbox", backend="faster_coco_eval")
