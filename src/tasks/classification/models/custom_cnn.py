import torch.nn as nn

from src.core.registry import MODELS


class ConvBlock(nn.Module):
    # Basic block used throughout the custom CNN backbone.
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CustomCnnBackbone(nn.Module):
    # Shared backbone across classification, segmentation, and detection tasks.
    # 5 stages, each one ConvBlock with stride=2, channel progression 3 -> 32 -> 64 -> 128 -> 256 -> 512.
    def __init__(self):
        super().__init__()
        self.stage1 = ConvBlock(3, 32)
        self.stage2 = ConvBlock(32, 64)
        self.stage3 = ConvBlock(64, 128)
        self.stage4 = ConvBlock(128, 256)
        self.stage5 = ConvBlock(256, 512)

    def forward(self, images):
        c1 = self.stage1(images)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return {"final": c5, "stages": [c1, c2, c3, c4, c5]}


@MODELS.register("custom_cnn_cls")
class CustomCnnClassifier(nn.Module):
    def __init__(self, num_classes=37, backbone_name=None, weights_path=None, **params):
        super().__init__()
        # backbone_name and weights_path are unused for this from-scratch model;
        # kept only for interface uniformity with the other two models.
        self.backbone = CustomCnnBackbone()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, images):
        backbone_output = self.backbone(images)
        features = backbone_output["final"]
        pooled = self.pool(features).flatten(1)
        logits = self.classifier(pooled)
        return logits
