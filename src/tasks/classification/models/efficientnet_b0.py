# torchvision EfficientNet-B0 classifier for oxford_pets 37-breed classification.
# Pretrained weights are injected from a local ImageNet checkpoint only (offline, CON-04/CON-05).
#
# Note on the local checkpoint: /mnt/d/backbones/efficientnet_b0_rwightman-7f5810bc.pth is
# named after timm's Ross Wightman despite that, its state_dict keys were verified to match
# torchvision's native efficientnet_b0() module naming 1:1 (360 keys, no missing/unexpected,
# no shape mismatches), so it loads with strict=True and no key_map is required.
import torch.nn as nn
import torchvision.models as tvm

from src.core.offline import load_local_weights
from src.core.registry import MODELS


@MODELS.register("efficientnet_b0_cls")
class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes=37, backbone_name="efficientnet_b0", weights_path=None, **params):
        super().__init__()
        self.backbone = tvm.efficientnet_b0(weights=None)
        # weights_path must always be resolved through load_local_weights, never skipped:
        # a None or missing path must raise LocalAssetError, not silently random-init.
        load_local_weights(self.backbone, weights_path, strict=True)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, images):
        return self.backbone(images)
