import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from src.core.registry import TRANSFORMS

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DetectionTransform:
    """Wraps a v2.Compose that jointly transforms an image and a detection target dict
    (PLAN-P4 SS3.5). Boxes are wrapped in tv_tensors.BoundingBoxes so v2 keeps them in sync with
    geometric ops (flip/resize) automatically; labels ride along as target["labels"], which
    v2.SanitizeBoundingBoxes finds via its default "labels" key heuristic and filters alongside
    dropped boxes.

    target is optional (default None) because the common `predict` CLI command
    (src/cli/commands.py) calls the eval transform as transform(image) on raw input images that
    have no annotation at all; OxfordPetsDetection.__getitem__ always calls it as
    transform(image, target). Mirrors ClassificationTransform/SegmentationTransform's dual call
    shape.
    """

    def __init__(self, compose):
        self.compose = compose

    def __call__(self, image, target=None):
        image = tv_tensors.Image(image) if not isinstance(image, tv_tensors.Image) else image
        height, width = image.shape[-2], image.shape[-1]

        if target is None:
            # Predict path has no GT. Feed empty boxes/labels through the joint compose so
            # Resize computes the same output size it would with a real target, then discard.
            dummy_target = {
                "boxes": tv_tensors.BoundingBoxes(
                    torch.zeros((0, 4), dtype=torch.float32), format="XYXY", canvas_size=(height, width)
                ),
                "labels": torch.zeros((0,), dtype=torch.long),
            }
            transformed_image, _ = self.compose(image, dummy_target)
            return transformed_image

        boxes = target["boxes"]
        boxes = boxes if isinstance(boxes, tv_tensors.BoundingBoxes) else tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=(height, width)
        )
        wrapped_target = {"boxes": boxes, "labels": target["labels"]}
        transformed_image, transformed_target = self.compose(image, wrapped_target)

        # Strip back down to plain tensors at the dataset/adapter boundary: absolute xyxy float
        # boxes, long labels (PLAN-P4 SS4.1) -- never leak tv_tensors past this transform.
        new_target = {
            "boxes": transformed_target["boxes"].as_subclass(torch.Tensor).to(torch.float32),
            "labels": transformed_target["labels"].to(torch.long),
        }
        return transformed_image, new_target


@TRANSFORMS.register("det_train")
def build_det_train_transform(image_size, train=True, **params):
    compose = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.Resize(image_size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        v2.SanitizeBoundingBoxes(min_size=1.0),
    ])
    return DetectionTransform(compose)


@TRANSFORMS.register("det_eval")
def build_det_eval_transform(image_size, train=False, **params):
    compose = v2.Compose([
        v2.Resize(image_size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        v2.SanitizeBoundingBoxes(min_size=1.0),
    ])
    return DetectionTransform(compose)
