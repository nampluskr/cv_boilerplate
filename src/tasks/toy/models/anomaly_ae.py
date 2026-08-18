import torch.nn as nn
import torch.nn.functional as F

from src.core.registry import MODELS


class ConvAutoencoder(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


@MODELS.register("toy_ae")
class ToyAutoencoder(ConvAutoencoder):
    """Standard path: adapter builds loss via loss_fn(reconstruction, image)."""


@MODELS.register("toy_ae_trainstep")
class ToyAutoencoderTrainStep(ConvAutoencoder):
    """Model-specific training path (FR-10): defines its own train_step."""

    def train_step(self, images, targets):
        reconstruction = self(images)
        recon_loss = F.mse_loss(reconstruction, images)
        smoothness = (reconstruction[:, :, 1:, :] - reconstruction[:, :, :-1, :]).abs().mean()
        loss = recon_loss + 0.01 * smoothness
        return {"loss": loss, "loss_dict": {"recon_loss": float(recon_loss.detach()),
                                             "smoothness": float(smoothness.detach())}}
