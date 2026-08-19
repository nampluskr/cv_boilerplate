import torch
import torch.nn as nn

from src.core.registry import MODELS


class ConvBlock(nn.Module):
    # Basic block used throughout the custom U-Net encoder and decoder.
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DeconvBlock(nn.Module):
    # Nearest-neighbor upsample followed by a stride=1 ConvBlock.
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_block = ConvBlock(in_channels, out_channels, stride=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv_block(x)
        return x


class CustomUnetEncoder(nn.Module):
    # 5 stages, each one ConvBlock with stride=2, channel progression 3 -> 32 -> 64 -> 128 -> 256 -> 512.
    def __init__(self):
        super().__init__()
        self.stage1 = ConvBlock(3, 32, stride=2)
        self.stage2 = ConvBlock(32, 64, stride=2)
        self.stage3 = ConvBlock(64, 128, stride=2)
        self.stage4 = ConvBlock(128, 256, stride=2)
        self.stage5 = ConvBlock(256, 512, stride=2)

    def forward(self, images):
        c1 = self.stage1(images)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return [c1, c2, c3, c4, c5]


class CustomUnetDecoder(nn.Module):
    # Symmetric decoder: upsample, concat with encoder skip, then ConvBlock.
    # Channel progression along the upsample path: 512 -> 256 -> 128 -> 64 -> 32 -> 32.
    def __init__(self):
        super().__init__()
        self.up4 = DeconvBlock(512 + 256, 256)
        self.up3 = DeconvBlock(256 + 128, 128)
        self.up2 = DeconvBlock(128 + 64, 64)
        self.up1 = DeconvBlock(64 + 32, 32)
        self.up0 = DeconvBlock(32, 32)

    def forward(self, encoder_features):
        c1, c2, c3, c4, c5 = encoder_features

        up = nn.functional.interpolate(c5, scale_factor=2, mode="nearest")
        d4 = self.up4.conv_block(torch.cat([up, c4], dim=1))

        up = nn.functional.interpolate(d4, scale_factor=2, mode="nearest")
        d3 = self.up3.conv_block(torch.cat([up, c3], dim=1))

        up = nn.functional.interpolate(d3, scale_factor=2, mode="nearest")
        d2 = self.up2.conv_block(torch.cat([up, c2], dim=1))

        up = nn.functional.interpolate(d2, scale_factor=2, mode="nearest")
        d1 = self.up1.conv_block(torch.cat([up, c1], dim=1))

        d0 = self.up0(d1)
        return d0


@MODELS.register("custom_unet_seg")
class CustomUnet(nn.Module):
    def __init__(self, num_classes=3, backbone_name=None, weights_path=None, **params):
        super().__init__()
        # backbone_name and weights_path are unused for this from-scratch model;
        # kept only for interface uniformity with the other segmentation models.
        self.encoder = CustomUnetEncoder()
        self.decoder = CustomUnetDecoder()
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, images):
        encoder_features = self.encoder(images)
        decoder_output = self.decoder(encoder_features)
        logits = self.head(decoder_output)
        # Stride-2 encoder stages round non-multiple-of-32 spatial sizes upward; realign
        # logits to the exact input resolution (PLAN-P3 SS4.1).
        if logits.shape[-2:] != images.shape[-2:]:
            logits = nn.functional.interpolate(
                logits, size=images.shape[-2:], mode="bilinear", align_corners=False
            )
        return logits
