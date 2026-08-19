# torchvision ResNet50 backbone with a local ImageNet-pretrained checkpoint,
# repurposed for oxford_pets 37-breed classification.
import torch.nn as nn
import torchvision.models as tvm

from src.core.offline import load_local_weights
from src.core.registry import MODELS


@MODELS.register("resnet50_cls")
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=37, backbone_name="resnet50", weights_path=None, **params):
        super().__init__()
        self.backbone = tvm.resnet50(weights=None)
        # Load the local ImageNet checkpoint while fc is still the original
        # 1000-class head, so strict=True key matching succeeds. weights_path
        # is never None-checked here: load_local_weights raises LocalAssetError
        # for a missing/None path instead of silently random-initializing.
        load_local_weights(self.backbone, weights_path, strict=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, images):
        return self.backbone(images)
