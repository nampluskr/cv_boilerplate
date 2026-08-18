import torch.nn as nn

from src.core.registry import MODELS


@MODELS.register("toy_cnn")
class ToyCnnClassifier(nn.Module):
    def __init__(self, num_classes=4, **params):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.head(x)


@MODELS.register("toy_cnn_small")
class ToyCnnSmallClassifier(nn.Module):
    def __init__(self, num_classes=4, **params):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.head(x)


@MODELS.register("toy_mlp")
class ToyMlpClassifier(nn.Module):
    def __init__(self, num_classes=4, image_size=(32, 32), **params):
        super().__init__()
        in_features = 3 * image_size[0] * image_size[1]
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 64), nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)
