import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from src.core.registry import TRANSFORMS

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AnomalyTransform:
    """Wraps a deterministic v2.Compose (Resize + ToDtype + Normalize) applied identically to
    train and eval (PLAN-P5 SS3.5: no augmentation for anomaly detection -- flips/crops would
    fabricate false anomalies and blur the "normal distribution only" training premise).

    Three call shapes, mirroring SegmentationTransform's dual shape plus MVTecAnomaly's own
    empty-dict train target:
      - target is None: predict CLI path (src/cli/commands.py calls transform(image) on a raw
        input image with no target at all). Returns the transformed image only.
      - target == {}: MVTecAnomaly train split (empty dict, PLAN-P5 SS3.4). Returns
        (image, {}) -- the mask never enters a train sample.
      - target has "mask": MVTecAnomaly valid/test split. The mask is wrapped in
        tv_tensors.Mask so v2 selects NEAREST interpolation and Normalize skips it, then resized
        jointly with the image so anomaly_map and mask stay pixel-aligned (PLAN-P5 SS3.5/SS7.1).
    """

    def __init__(self, compose):
        self.compose = compose

    def _resize_image_only(self, image):
        height, width = image.shape[-2], image.shape[-1]
        dummy_mask = tv_tensors.Mask(torch.zeros(1, height, width, dtype=torch.int64))
        transformed_image, _ = self.compose(image, dummy_mask)
        return transformed_image

    def __call__(self, image, target=None):
        image = tv_tensors.Image(image) if not isinstance(image, tv_tensors.Image) else image

        if target is None:
            return self._resize_image_only(image)

        if not target:
            return self._resize_image_only(image), {}

        mask = target["mask"]
        mask = tv_tensors.Mask(mask) if not isinstance(mask, tv_tensors.Mask) else mask
        transformed_image, transformed_mask = self.compose(image, mask)
        new_target = dict(target)
        new_target["mask"] = transformed_mask.to(torch.int64)
        return transformed_image, new_target


@TRANSFORMS.register("anomaly_default")
def build_anomaly_transform(image_size, train=True, **params):
    # image_size is fixed [256, 256] (PLAN-P5 SS3.5) so anomaly_map and mask share resolution.
    compose = v2.Compose([
        v2.Resize(image_size, antialias=True),
        v2.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64}, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return AnomalyTransform(compose)
