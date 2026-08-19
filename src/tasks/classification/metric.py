from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from src.core.registry import METRICS


@METRICS.register("top1_accuracy")
def build_top1_accuracy(num_classes=37, **params):
    return MulticlassAccuracy(num_classes=num_classes, average="micro")


@METRICS.register("macro_f1")
def build_macro_f1(num_classes=37, **params):
    return MulticlassF1Score(num_classes=num_classes, average="macro")
