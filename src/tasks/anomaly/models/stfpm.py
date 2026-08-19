# STFPM (Student-Teacher Feature Pyramid Matching) for anomaly detection (PLAN-P5 SS4.4).
# This is an original implementation written directly from the method description below
# (Wang et al., "Student-Teacher Feature Pyramid Matching for Anomaly Detection"), not a
# verbatim port of any specific external repository -- github.com/nampluskr/defectvad was
# suggested as a reference pattern but is not reachable from this offline sandbox, so the
# architecture and loss below were implemented from the STFPM method description and the
# interface contract in PLAN-P5 SS4.2/SS4.4.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from src.core.offline import load_local_weights
from src.core.registry import MODELS

# Standard STFPM layer selection: outputs of layer1/layer2/layer3 (4x, 8x, 16x
# downsampling stages), skipping the stem and layer4 as in the original paper.
FEATURE_LAYERS = ("layer1", "layer2", "layer3")


class ResNet18FeaturePyramid(nn.Module):
    """Thin wrapper around torchvision resnet18 that exposes layer1/layer2/layer3
    activations via a manual forward (no torch.hub, no network access -- this only
    builds the architecture; weights are loaded separately with local checkpoints)."""

    def __init__(self):
        super().__init__()
        backbone = tvm.resnet18(weights=None)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.fc = backbone.fc

    def forward(self, images):
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        features = {}
        x = self.layer1(x)
        features["layer1"] = x
        x = self.layer2(x)
        features["layer2"] = x
        x = self.layer3(x)
        features["layer3"] = x
        return features


@MODELS.register("stfpm_anomaly")
class Stfpm(nn.Module):
    """Student-teacher feature pyramid matching (PLAN-P5 SS4.4).

    teacher: torchvision resnet18 loaded with local ImageNet weights, frozen for the
    lifetime of the model (parameters excluded from gradient updates, and forced into
    eval() mode regardless of the outer train()/eval() calls the engine makes, so its
    BatchNorm running stats never drift -- PLAN-P5 SS4.2).
    student: an architecturally identical resnet18, randomly initialized, trained to
    reproduce the teacher's layer1/layer2/layer3 activations on normal images only.

    Loss (train_step): for each of layer1/layer2/layer3, L2-normalize both teacher and
    student feature maps along the channel dimension, then take the per-layer MSE
    (mean of squared differences over all elements) between them; the total loss is the
    sum of the three per-layer losses.

    anomaly_map (forward): for each layer, the per-pixel squared L2 distance between
    the channel-L2-normalized teacher/student feature maps (summed over channels,
    equivalent to sum of squared differences since both are unit-norm along channels),
    upsampled to input resolution with bilinear interpolation, then averaged across the
    3 layers to produce the final (B, H, W) map. pred_score is the per-image max of
    that averaged map.
    """

    def __init__(self, weights_path="/mnt/d/backbones/resnet18-f37072fd.pth",
                 feature_layers=FEATURE_LAYERS, **params):
        super().__init__()
        self.feature_layers = tuple(feature_layers)

        self.teacher = ResNet18FeaturePyramid()
        # weights_path is never guarded by a None-check here: load_local_weights raises
        # LocalAssetError itself for a missing/None path, so a missing checkpoint fails
        # loudly instead of silently random-initializing the teacher.
        load_local_weights(self.teacher, weights_path, strict=True)
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.teacher.eval()

        # Student shares the exact same architecture but starts from random init --
        # never loads the pretrained checkpoint, since matching the teacher from a
        # random start is the whole point of the method.
        self.student = ResNet18FeaturePyramid()

    def train(self, mode=True):
        super().train(mode)
        # Keep the frozen teacher in eval() regardless of the outer call, so the
        # engine's per-epoch model.train() never lets student-batch statistics drift
        # the teacher's BatchNorm running stats (PLAN-P5 SS4.2).
        self.teacher.eval()
        return self

    @staticmethod
    def _normalize(feature):
        return F.normalize(feature, p=2, dim=1, eps=1e-8)

    def _extract(self, images):
        with torch.no_grad():
            teacher_features = self.teacher(images)
        student_features = self.student(images)
        return teacher_features, student_features

    def train_step(self, images, targets):
        # targets is unused: this model trains on normal images only, with no labels.
        teacher_features, student_features = self._extract(images)
        loss = None
        loss_dict = {}
        for name in self.feature_layers:
            teacher_norm = self._normalize(teacher_features[name])
            student_norm = self._normalize(student_features[name])
            layer_loss = F.mse_loss(student_norm, teacher_norm)
            loss_dict[f"loss_{name}"] = layer_loss.item()
            loss = layer_loss if loss is None else loss + layer_loss
        loss_dict["loss"] = loss.item()
        return {"loss": loss, "loss_dict": loss_dict}

    def forward(self, images):
        teacher_features, student_features = self._extract(images)
        target_size = images.shape[-2:]
        layer_maps = []
        for name in self.feature_layers:
            teacher_norm = self._normalize(teacher_features[name])
            student_norm = self._normalize(student_features[name])
            # Sum of squared differences over channels == squared L2 distance per
            # spatial location, since both feature maps are unit-norm along channels.
            error_map = (teacher_norm - student_norm).pow(2).sum(dim=1, keepdim=True)
            error_map = F.interpolate(
                error_map, size=target_size, mode="bilinear", align_corners=False
            )
            layer_maps.append(error_map)
        anomaly_map = torch.stack(layer_maps, dim=0).mean(dim=0).squeeze(1)
        pred_score = anomaly_map.flatten(1).amax(dim=1)
        return {"pred_score": pred_score, "anomaly_map": anomaly_map}
