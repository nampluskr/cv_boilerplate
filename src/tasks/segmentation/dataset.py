import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.core.errors import LocalAssetError
from src.core.registry import DATASETS
from src.data.split import assert_disjoint, load_split_file

# Oxford-IIIT Pet trimap value -> project class index mapping (PLAN-P3 SS3.1).
# 1=Foreground(pet) -> 1, 2=Background -> 0, 3=Not-classified(boundary) -> 2.
TRIMAP_VALUE_TO_CLASS_INDEX = {2: 0, 1: 1, 3: 2}
CLASS_NAMES = ["background", "pet", "boundary"]


def breed_name_from_stem(stem):
    """'Abyssinian_100' -> 'Abyssinian', 'Egyptian_Mau_100' -> 'Egyptian_Mau'."""
    return stem.rsplit("_", 1)[0]


def build_trimap_lookup_table():
    """256-entry lookup table used with a fancy-index remap. Every entry not covered by
    TRIMAP_VALUE_TO_CLASS_INDEX is set to -1 so unexpected pixel values are caught explicitly
    instead of being silently mapped to some default class (PLAN-P3 SS3.1, NFR-11)."""
    table = np.full(256, -1, dtype=np.int64)
    for raw_value, class_index in TRIMAP_VALUE_TO_CLASS_INDEX.items():
        table[raw_value] = class_index
    return table


TRIMAP_LOOKUP_TABLE = build_trimap_lookup_table()


def remap_trimap(mask_array):
    """Map raw trimap pixel values {1,2,3} to project class indices {1,0,2}. Raises if any
    pixel holds a value outside {1,2,3} instead of silently ignoring it (PLAN-P3 SS3.1)."""
    unexpected = sorted(set(np.unique(mask_array).tolist()) - set(TRIMAP_VALUE_TO_CLASS_INDEX.keys()))
    if unexpected:
        raise RuntimeError(
            f"trimap contains unexpected pixel values {unexpected}; only "
            f"{sorted(TRIMAP_VALUE_TO_CLASS_INDEX.keys())} are valid (PLAN-P3 SS3.1)."
        )
    return TRIMAP_LOOKUP_TABLE[mask_array]


@DATASETS.register("oxford_pets_seg")
class OxfordPetsSegmentation(Dataset):
    """Oxford-IIIT Pet trimap 3-class segmentation. Returns (image, mask) where mask is a
    LongTensor of shape (H, W) with values in [0, 2] (PLAN-P3 SS3.4)."""

    def __init__(self, root, split, transform=None, split_path=None, **params):
        if split not in ("train", "valid", "test"):
            raise ValueError(f"split must be one of train/valid/test, got '{split}'")
        if split_path is None:
            raise LocalAssetError(
                "OxfordPetsSegmentation requires a split file path (data.split.path in the "
                "config, forwarded automatically by build_dataset() when data.split.mode=='file')."
            )

        images_dir = os.path.join(root, "images")
        trimaps_dir = os.path.join(root, "annotations", "trimaps")
        if not os.path.isdir(root) or not os.path.isdir(images_dir) or not os.path.isdir(trimaps_dir):
            raise LocalAssetError(
                f"oxford_pets dataset not found under root='{root}'. Expected "
                f"'{images_dir}' and '{trimaps_dir}' to exist. This dataset is never "
                f"downloaded automatically (CON-03); place it under /mnt/d/datasets."
            )
        if not os.path.isfile(split_path):
            raise LocalAssetError(f"split file not found: {split_path}")

        self.root = root
        self.split = split
        self.transform = transform
        self.split_path = split_path
        self.classes = list(CLASS_NAMES)

        split_dict = load_split_file(split_path)  # already asserts disjoint internally
        assert_disjoint(split_dict)
        # PLAN-P3 SS3.2: this dataset shares the canonical split with Classification. All
        # 7,349 canonical IDs have a trimap, so no further membership check against list.txt
        # is needed beyond what the split file's own assert_disjoint already guarantees.

        self.ids = sorted(split_dict[split])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        stem = self.ids[index]
        image_path = os.path.join(self.root, "images", stem + ".jpg")
        mask_path = os.path.join(self.root, "annotations", "trimaps", stem + ".png")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"failed to load image '{image_path}' (stem='{stem}'): {exc}") from exc
        try:
            mask_array = np.array(Image.open(mask_path))
        except Exception as exc:
            raise RuntimeError(f"failed to load trimap '{mask_path}' (stem='{stem}'): {exc}") from exc

        remapped = remap_trimap(mask_array)
        mask = torch.from_numpy(remapped).to(torch.int64)

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            import torchvision.transforms.v2.functional as F

            image = F.to_dtype(F.to_image(image), torch.float32, scale=True)

        return image, mask
