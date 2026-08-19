import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from src.core.registry import TRANSFORMS

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SegmentationTransform:
    """Wraps a v2.Compose that jointly transforms an image and a dense mask target
    (PLAN-P3 SS3.5, the only template deviation for P3). The mask is wrapped in
    tv_tensors.Mask so v2 selects NEAREST interpolation for it and Normalize skips it
    entirely -- see build_seg_train_transform/build_seg_eval_transform below.

    target is optional (default None) because the common `predict` CLI command
    (src/cli/commands.py) calls the eval transform as transform(image) with no target on raw
    input images that have no mask at all; OxfordPetsSegmentation.__getitem__ always calls it
    as transform(image, mask). Mirrors ClassificationTransform's dual call shape."""

    def __init__(self, compose):
        self.compose = compose

    def __call__(self, image, target=None):
        image = tv_tensors.Image(image) if not isinstance(image, tv_tensors.Image) else image
        if target is None:
            # Predict path (src/cli/commands.py) has no mask. Feed a same-size dummy mask
            # through the joint compose so RandomResizedCrop/Resize compute identical crop/
            # resize parameters as they would with a real mask, then discard it.
            height, width = image.shape[-2], image.shape[-1]
            dummy_mask = tv_tensors.Mask(torch.zeros(1, height, width, dtype=torch.int64))
            transformed_image, _ = self.compose(image, dummy_mask)
            return transformed_image

        mask = tv_tensors.Mask(target) if not isinstance(target, tv_tensors.Mask) else target
        transformed_image, transformed_mask = self.compose(image, mask)
        return transformed_image, transformed_mask.to(torch.int64)


@TRANSFORMS.register("seg_train")
def build_seg_train_transform(image_size, train=True, **params):
    compose = v2.Compose([
        v2.RandomResizedCrop(image_size, scale=(0.7, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64}, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return SegmentationTransform(compose)


@TRANSFORMS.register("seg_eval")
def build_seg_eval_transform(image_size, train=False, **params):
    # No CenterCrop: cropping would discard part of the GT mask from evaluation and distort
    # mIoU (PLAN-P3 SS3.5).
    compose = v2.Compose([
        v2.Resize(image_size, antialias=True),
        v2.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64}, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return SegmentationTransform(compose)
