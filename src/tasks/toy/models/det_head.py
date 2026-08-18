import torch.nn as nn

from src.core.registry import MODELS


@MODELS.register("toy_det_head")
class ToyDetHead(nn.Module):
    """Small anchor-free head over an 8x downsampled grid.

    Output channel layout per cell: [objectness(1), class_logits(num_fg_classes), box_ltrb(4)].
    """

    def __init__(self, num_fg_classes=2, stride=8, **params):
        super().__init__()
        self.num_fg_classes = num_fg_classes
        self.stride = stride
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64, 1 + num_fg_classes + 4, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
