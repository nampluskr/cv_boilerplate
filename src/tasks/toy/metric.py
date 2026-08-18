from torchmetrics.classification import BinaryAUROC, MulticlassAccuracy, MulticlassJaccardIndex
from torchmetrics.detection import MeanAveragePrecision

from src.core.registry import METRICS

METRICS.register("accuracy")(MulticlassAccuracy)
METRICS.register("miou")(MulticlassJaccardIndex)
METRICS.register("image_auroc")(BinaryAUROC)
METRICS.register("pixel_auroc")(BinaryAUROC)


@METRICS.register("map")
def build_map_metric(backend="faster_coco_eval", **params):
    return MeanAveragePrecision(backend=backend, **params)
