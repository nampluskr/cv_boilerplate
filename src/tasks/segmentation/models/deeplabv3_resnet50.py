# torchvision DeepLabV3-ResNet50 backbone with a local COCO-pretrained checkpoint,
# repurposed for oxford_pets 3-class trimap segmentation.
import torch.nn as nn
import torchvision.models.segmentation as tvm_seg

from src.core.offline import load_local_weights
from src.core.registry import MODELS


@MODELS.register("deeplabv3_resnet50_seg")
class DeepLabV3ResNet50Segmenter(nn.Module):
    def __init__(self, num_classes=3, backbone_name="resnet50", weights_path=None, **params):
        super().__init__()
        # aux_loss=True and num_classes=21 are required at construction time so the
        # module's state_dict keys (including aux_classifier.*) match the local COCO
        # checkpoint exactly, enabling strict=True loading below.
        self.model = tvm_seg.deeplabv3_resnet50(
            weights=None, weights_backbone=None, num_classes=21, aux_loss=True
        )
        # Load the local COCO checkpoint while the head is still the original
        # 21-class head, so strict=True key matching succeeds. weights_path is
        # never None-checked here: load_local_weights raises LocalAssetError
        # for a missing/None path instead of silently random-initializing.
        load_local_weights(self.model, weights_path, strict=True)

        # Replace the final classifier conv with a num_classes head, after the
        # checkpoint load so the replacement does not affect key matching above.
        final_conv = self.model.classifier[-1]
        self.model.classifier[-1] = nn.Conv2d(
            final_conv.in_channels, num_classes, kernel_size=final_conv.kernel_size
        )

        # Drop the aux head so this model's loss/output structure matches the
        # other models compared in this benchmark (no aux branch).
        self.model.aux_classifier = None

    def forward(self, images):
        # torchvision's deeplabv3 forward returns an OrderedDict with an "out" key
        # (and "aux" when aux_classifier is set). Unwrap it here so a plain Tensor
        # is the only thing this file ever exposes to the adapter/engine.
        output = self.model(images)
        return output["out"]
