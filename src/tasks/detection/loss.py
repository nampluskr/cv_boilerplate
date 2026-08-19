from src.core.registry import LOSSES


@LOSSES.register("none")
class NoLoss:
    """Placeholder loss for Detection (PLAN-P4 SS5). Detection loss lives inside each model's
    train_step (PLAN-P4 SS4.1), so DetectionAdapter never calls self.loss_fn. This placeholder
    exists so config.loss.name=none is explicit at the config level and a real bug that
    accidentally calls it fails loudly instead of silently computing nothing."""

    def __init__(self, **params):
        pass

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "loss.name=none was called directly. Detection loss is computed inside the model's "
            "train_step (PLAN-P4 SS4.1/SS5); DetectionAdapter must never call self.loss_fn."
        )
