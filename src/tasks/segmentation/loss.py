import torch.nn as nn

from src.core.registry import LOSSES


@LOSSES.register("seg_cross_entropy")
def build_seg_cross_entropy(class_weight=None, ignore_index=-100, **params):
    # PLAN-P3 SS5.1: pixel-wise CrossEntropyLoss over logits (B, C, H, W) and target (B, H, W).
    # ignore_index defaults to -100 (inactive): the boundary class is trained and evaluated as
    # a first-class label, never discarded (PLAN-P3 SS3.1).
    return nn.CrossEntropyLoss(weight=class_weight, ignore_index=ignore_index)
