import torch
from torchvision.transforms import v2

from src.core.registry import TRANSFORMS

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ClassificationTransform:
    """Wraps a v2.Compose that only ever sees the image. The scalar label target is passed
    through unchanged (PLAN-P2 SS3.4: 'Classification은 target을 변형하지 않지만 시그니처는
    4태스크 공통으로 유지한다'). Feeding a plain label Tensor into v2.Compose alongside the
    image would risk it being treated as an image-like leaf by transforms such as Normalize.

    target is optional (default None) because the common `predict` CLI command
    (src/cli/commands.py) calls the eval transform as transform(image) with no target, on raw
    input images that have no label at all; OxfordPetsClassification.__getitem__ always calls
    it as transform(image, target). See the P2 change-request note about this dual call shape."""

    def __init__(self, compose):
        self.compose = compose

    def __call__(self, image, target=None):
        transformed = self.compose(image)
        if target is None:
            return transformed
        return transformed, target


@TRANSFORMS.register("cls_train")
def build_cls_train_transform(image_size, train=True, **params):
    compose = v2.Compose([
        v2.RandomResizedCrop(image_size, scale=(0.7, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return ClassificationTransform(compose)


@TRANSFORMS.register("cls_eval")
def build_cls_eval_transform(image_size, train=False, **params):
    resize_size = int(image_size[0] * 256 / 224)
    compose = v2.Compose([
        v2.Resize(resize_size, antialias=True),
        v2.CenterCrop(image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return ClassificationTransform(compose)
