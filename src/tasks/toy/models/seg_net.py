import torch.nn as nn

from src.core.registry import MODELS


@MODELS.register("toy_segnet")
class ToySegNet(nn.Module):
    def __init__(self, num_classes=3, **params):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 1),
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)
