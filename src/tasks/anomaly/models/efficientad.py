# EfficientAD (PDN-Small teacher + student + autoencoder) for anomaly detection (PLAN-P5 SS4.4).
# Original implementation written directly from the method description below (Batzner et al.,
# "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies"), not a verbatim
# port of any specific external repository -- github.com/nampluskr/defectvad was suggested as a
# reference pattern but is not reachable from this offline sandbox, so the PDN-Small teacher/student
# architecture and training scheme below follow the method description and the interface contract
# in PLAN-P5 SS4.2/SS4.4. The autoencoder branch uses a simplified encoder-decoder (plain strided
# convs + bilinear upsampling to the teacher's exact feature resolution) rather than the paper's
# specific magic-number upsampling schedule, since v0.1's acceptance criteria do not require
# literal architectural fidelity (PLAN-P5 SS8.2: 5 epochs is far below EfficientAD's standard
# training budget, low absolute accuracy is expected and not judged).
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.offline import load_local_weights
from src.core.registry import MODELS

TEACHER_OUT_CHANNELS = 384


class PdnSmall(nn.Module):
    """PDN-Small patch descriptor network. With padding=True (the default used here) and a
    256x256 input this produces an exact 64x64 spatial output -- verified against the checkpoint
    shapes in /mnt/d/backbones/efficientad_pretrained_weights/pretrained_teacher_small.pth
    (conv1..conv4, 8 keys, PLAN-P5 SS4.3), which this module's layer names match exactly so
    load_local_weights needs no key_map."""

    def __init__(self, out_channels=TEACHER_OUT_CHANNELS):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x


class AutoEncoder(nn.Module):
    """Bottleneck encoder-decoder that learns to reproduce the (normalized) teacher output from
    the raw image alone. Decoding upsamples to `target_size` explicitly, so the output always
    matches the teacher/student feature resolution regardless of input size."""

    def __init__(self, out_channels=TEACHER_OUT_CHANNELS):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, target_size):
        z = self.encoder(x)
        z = F.interpolate(z, size=target_size, mode="bilinear", align_corners=False)
        return self.decoder(z)


@MODELS.register("efficientad_anomaly")
class EfficientAd(nn.Module):
    """PDN-Small teacher (frozen) + PDN-Small student + autoencoder (PLAN-P5 SS4.4).

    teacher: PdnSmall loaded with local ImageNet-distilled weights, frozen for the model's
    lifetime (parameters excluded from gradients, forced eval() regardless of outer train()
    calls -- PLAN-P5 SS4.2, identical pattern to Stfpm).
    student: PdnSmall with 2x teacher_out_channels: the first half is trained to match the
    (channel-normalized) teacher output on normal images; the second half is trained to match
    the autoencoder's output.
    autoencoder: trained to reconstruct the channel-normalized teacher output from the image
    alone. Its divergence from the student's second half is a second anomaly signal that
    catches structural anomalies the direct teacher-student channel misses (per the method).

    Hooks (PLAN-P5 SS5):
      on_fit_start(train_loader, device): one pass over the *train* loader (no test leakage) to
        compute the teacher's per-channel output mean/std, used to normalize teacher features
        before every distance computation.
      on_fit_end(valid_loader, device): one pass over the *valid* loader (no test leakage) to
        compute quantile normalization constants (qa/qb, 90th/99.5th percentile) for the
        student-teacher map and the student-autoencoder map, so the two maps can be combined on
        a comparable scale. AnomalyAdapter.on_fit_end calls this, then separately determines the
        F1-optimal image/pixel threshold on the same valid split (PLAN-P5 SS6) -- this hook only
        produces the quantile constants, not the threshold itself.
    """

    def __init__(self, weights_path="/mnt/d/backbones/efficientad_pretrained_weights/"
                                     "pretrained_teacher_small.pth",
                 teacher_out_channels=TEACHER_OUT_CHANNELS, hard_mining_quantile=0.999, **params):
        super().__init__()
        self.teacher_out_channels = teacher_out_channels
        self.hard_mining_quantile = hard_mining_quantile

        self.teacher = PdnSmall(out_channels=teacher_out_channels)
        # Missing/None weights_path fails loudly via LocalAssetError -- never a silent
        # random-initialized fallback (PLAN-P5 SS4.3).
        load_local_weights(self.teacher, weights_path, strict=True)
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.teacher.eval()

        self.student = PdnSmall(out_channels=2 * teacher_out_channels)
        self.autoencoder = AutoEncoder(out_channels=teacher_out_channels)

        # Channel normalization statistics for the teacher output (SS5 on_fit_start), and
        # quantile normalization constants for combining the two anomaly maps (SS5 on_fit_end).
        # Registered as buffers so they travel with the checkpoint; initialized to the identity
        # transform until the corresponding hook actually runs.
        self.register_buffer("teacher_mean", torch.zeros(1, teacher_out_channels, 1, 1))
        self.register_buffer("teacher_std", torch.ones(1, teacher_out_channels, 1, 1))
        self.register_buffer("qa_st", torch.tensor(0.0))
        self.register_buffer("qb_st", torch.tensor(1.0))
        self.register_buffer("qa_ae", torch.tensor(0.0))
        self.register_buffer("qb_ae", torch.tensor(1.0))

    def train(self, mode=True):
        super().train(mode)
        # Frozen teacher stays in eval() regardless of the outer call, so the engine's
        # per-epoch model.train() never drifts its BatchNorm-free but still mode-sensitive
        # layers (dropout is absent from PdnSmall, but this mirrors Stfpm's guard exactly and
        # protects against any future layer additions) -- PLAN-P5 SS4.2.
        self.teacher.eval()
        return self

    def _teacher_normalized(self, images):
        with torch.no_grad():
            teacher_out = self.teacher(images)
        return (teacher_out - self.teacher_mean) / self.teacher_std.clamp_min(1e-8)

    def train_step(self, images, targets):
        # targets is unused: normal images only, no labels (PLAN-P5 SS4.1).
        teacher_norm = self._teacher_normalized(images)
        student_out = self.student(images)
        student_st = student_out[:, :self.teacher_out_channels]
        student_ae = student_out[:, self.teacher_out_channels:]

        feature_size = teacher_norm.shape[-2:]
        ae_out = self.autoencoder(images, feature_size)

        distance_st = (teacher_norm - student_st).pow(2).mean(dim=1)  # (B, h, w)
        hard_threshold = torch.quantile(distance_st.detach().flatten(), self.hard_mining_quantile)
        hard_mask = distance_st >= hard_threshold
        loss_st = distance_st[hard_mask].mean() if hard_mask.any() else distance_st.mean()

        loss_ae = F.mse_loss(ae_out, teacher_norm.detach())
        loss_stae = F.mse_loss(student_ae, ae_out.detach())

        loss = loss_st + loss_ae + loss_stae
        loss_dict = {
            "loss": loss.item(),
            "loss_st": loss_st.item(),
            "loss_ae": loss_ae.item(),
            "loss_stae": loss_stae.item(),
        }
        return {"loss": loss, "loss_dict": loss_dict}

    def _raw_maps(self, images):
        teacher_norm = self._teacher_normalized(images)
        student_out = self.student(images)
        student_st = student_out[:, :self.teacher_out_channels]
        student_ae = student_out[:, self.teacher_out_channels:]
        feature_size = teacher_norm.shape[-2:]
        ae_out = self.autoencoder(images, feature_size)

        map_st = (teacher_norm - student_st).pow(2).mean(dim=1)   # (B, h, w)
        map_ae = (student_ae - ae_out).pow(2).mean(dim=1)         # (B, h, w)
        return map_st, map_ae

    def forward(self, images):
        map_st, map_ae = self._raw_maps(images)
        map_st_norm = (map_st - self.qa_st) / (self.qb_st - self.qa_st).clamp_min(1e-8)
        map_ae_norm = (map_ae - self.qa_ae) / (self.qb_ae - self.qa_ae).clamp_min(1e-8)
        combined = 0.5 * (map_st_norm + map_ae_norm)

        target_size = images.shape[-2:]
        combined = F.interpolate(
            combined.unsqueeze(1), size=target_size, mode="bilinear", align_corners=False
        ).squeeze(1)
        pred_score = combined.flatten(1).amax(dim=1)
        return {"pred_score": pred_score, "anomaly_map": combined}

    @torch.no_grad()
    def on_fit_start(self, train_loader, device):
        # One pass over *train only* (PLAN-P5 SS5 constraint: no valid/test leakage here) to
        # compute the teacher's per-channel output mean/std on normal images.
        self.eval()
        channel_sum = torch.zeros(self.teacher_out_channels, device=device)
        channel_sq_sum = torch.zeros(self.teacher_out_channels, device=device)
        pixel_count = 0
        for images, _ in train_loader:
            images = images.to(device)
            teacher_out = self.teacher(images)
            channel_sum += teacher_out.sum(dim=(0, 2, 3))
            channel_sq_sum += teacher_out.pow(2).sum(dim=(0, 2, 3))
            pixel_count += teacher_out.shape[0] * teacher_out.shape[2] * teacher_out.shape[3]

        mean = channel_sum / pixel_count
        variance = (channel_sq_sum / pixel_count) - mean.pow(2)
        std = variance.clamp_min(1e-8).sqrt()
        self.teacher_mean.copy_(mean.view(1, -1, 1, 1))
        self.teacher_std.copy_(std.view(1, -1, 1, 1))

    @torch.no_grad()
    def on_fit_end(self, valid_loader, device):
        # One pass over *valid only* (PLAN-P5 SS5/SS6: no test leakage) to compute quantile
        # normalization constants for map_st and map_ae so forward() can combine them on a
        # comparable scale. Threshold decision itself is AnomalyAdapter's job, run separately
        # right after this hook returns.
        self.eval()
        st_values, ae_values = [], []
        for images, _ in valid_loader:
            images = images.to(device)
            map_st, map_ae = self._raw_maps(images)
            st_values.append(map_st.flatten().cpu())
            ae_values.append(map_ae.flatten().cpu())

        st_values = torch.cat(st_values)
        ae_values = torch.cat(ae_values)
        qa_st, qb_st = torch.quantile(st_values, torch.tensor([0.9, 0.995]))
        qa_ae, qb_ae = torch.quantile(ae_values, torch.tensor([0.9, 0.995]))
        self.qa_st.copy_(qa_st.to(device))
        self.qb_st.copy_(qb_st.to(device))
        self.qa_ae.copy_(qa_ae.to(device))
        self.qb_ae.copy_(qb_ae.to(device))
