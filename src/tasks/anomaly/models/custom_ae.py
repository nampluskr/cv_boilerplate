import torch.nn as nn
import torch.nn.functional as F

from src.core.registry import MODELS


class CustomAeEncoder(nn.Module):
    # 4 stages, each stride=2 Conv2d(kernel=4, padding=1), which exactly halves an
    # even spatial size. Channel progression: 3 -> 32 -> 64 -> 128 -> latent_channels.
    def __init__(self, latent_channels=256):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, latent_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, images):
        x = self.stage1(images)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


class CustomAeDecoder(nn.Module):
    # Mirrors the encoder with 4 stride=2 ConvTranspose2d(kernel=4, padding=1) stages,
    # each exactly doubling spatial size, so the input resolution is inverted exactly
    # for any input side length divisible by 16 (256 in this project's config).
    # Channel progression: latent_channels -> 128 -> 64 -> 32 -> 3.
    def __init__(self, latent_channels=256):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Final stage reconstructs raw pixel-space values; no activation so the
        # output range matches whatever normalization the input transform applies.
        self.stage4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


@MODELS.register("custom_ae_anomaly")
class CustomAe(nn.Module):
    """Plain convolutional autoencoder for anomaly detection (PLAN-P5 SS4.4).

    Trained on normal-only images with pixel-wise reconstruction MSE. At inference
    time, the per-pixel squared reconstruction error (channel-averaged) is used as
    the anomaly map, and the image-level anomaly score is the max of that map.

    weights_path is accepted only for interface uniformity with the other anomaly
    models (SS4.3 local-weight loading); this model is trained from scratch and
    never loads external weights, so weights_path is unused here.
    """

    def __init__(self, weights_path=None, latent_channels=256, **params):
        super().__init__()
        # weights_path is intentionally unused: no pretrained weights apply to this
        # from-scratch autoencoder.
        self.encoder = CustomAeEncoder(latent_channels=latent_channels)
        self.decoder = CustomAeDecoder(latent_channels=latent_channels)

    def reconstruct(self, images):
        latent = self.encoder(images)
        recon = self.decoder(latent)
        # The 4-stage stride=2 encoder/decoder pair inverts spatial size exactly when
        # H and W are divisible by 16; interpolate as a safety net for other sizes.
        if recon.shape[-2:] != images.shape[-2:]:
            recon = F.interpolate(recon, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return recon

    def train_step(self, images, targets):
        # targets is unused: this model trains on normal images only, with no labels.
        recon = self.reconstruct(images)
        loss = F.mse_loss(recon, images)
        return {"loss": loss, "loss_dict": {"loss": loss.item()}}

    def forward(self, images):
        recon = self.reconstruct(images)
        anomaly_map = (recon - images).pow(2).mean(dim=1)
        pred_score = anomaly_map.flatten(1).amax(dim=1)
        return {"pred_score": pred_score, "anomaly_map": anomaly_map}
