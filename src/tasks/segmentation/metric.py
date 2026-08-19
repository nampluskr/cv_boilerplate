from torchmetrics.classification import MulticlassF1Score

from src.core.registry import METRICS

# "miou" is intentionally NOT registered here: src/tasks/toy/metric.py already registers
# METRICS.register("miou")(MulticlassJaccardIndex) in the shared global METRICS namespace
# (imported by every task package, PLAN-P1 SS4/toy fixtures), and torchmetrics'
# MulticlassJaccardIndex defaults average="macro" already, so METRICS.build("miou",
# num_classes=3) yields exactly PLAN-P3 SS5.2's MulticlassJaccardIndex(num_classes=3,
# average="macro"). Re-registering the same key here would raise RegistryError (the same
# reuse pattern P2's classification._base.yaml loss.name=cross_entropy relies on, since
# "cross_entropy" is likewise only registered in src/tasks/toy/loss.py).


@METRICS.register("dice")
def build_dice(num_classes=3, **params):
    # Dice coefficient == per-class F1 (PLAN-P3 SS5.2). multidim_average="global" accumulates
    # every pixel across the batch into one confusion matrix instead of averaging per-image,
    # which would over-weight small objects.
    return MulticlassF1Score(num_classes=num_classes, average="macro", multidim_average="global")
