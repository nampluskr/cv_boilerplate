import os

from PIL import Image
from torch.utils.data import Dataset

from src.core.errors import LocalAssetError
from src.core.registry import DATASETS
from src.data.split import assert_disjoint, assert_subset, load_split_file

import torch


def read_list_txt(list_path):
    """Parse annotations/list.txt into {stem: class_id} with class_id 1-based.

    Comment lines starting with '#' are skipped. list.txt is the sample index basis
    (PLAN.md SS6.4), not the images/ directory listing, because the two disagree by 41 files.
    """
    mapping = {}
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            mapping[parts[0]] = int(parts[1])
    return mapping


def breed_name_from_stem(stem):
    """'Abyssinian_100' -> 'Abyssinian', 'Egyptian_Mau_100' -> 'Egyptian_Mau'."""
    return stem.rsplit("_", 1)[0]


def build_class_names(stem_to_class):
    """Return a list of 37 breed names indexed by 0-based label."""
    id_to_name = {}
    for stem, class_id in stem_to_class.items():
        id_to_name.setdefault(class_id, breed_name_from_stem(stem))
    return [id_to_name[class_id] for class_id in sorted(id_to_name)]


@DATASETS.register("oxford_pets_cls")
class OxfordPetsClassification(Dataset):
    """Oxford-IIIT Pet 37-breed classification. Returns (image, label) where label is a
    0-based breed index in [0, 36] (PLAN-P2 SS3.4, CON-10)."""

    def __init__(self, root, split, transform=None, split_path=None, **params):
        if split not in ("train", "valid", "test"):
            raise ValueError(f"split must be one of train/valid/test, got '{split}'")
        if split_path is None:
            raise LocalAssetError(
                "OxfordPetsClassification requires a split file path (data.split.path in the "
                "config, forwarded automatically by build_dataset() when data.split.mode=='file')."
            )

        list_path = os.path.join(root, "annotations", "list.txt")
        images_dir = os.path.join(root, "images")
        if not os.path.isdir(root) or not os.path.isfile(list_path) or not os.path.isdir(images_dir):
            raise LocalAssetError(
                f"oxford_pets dataset not found under root='{root}'. Expected "
                f"'{list_path}' and '{images_dir}' to exist. This dataset is never "
                f"downloaded automatically (CON-03); place it under /mnt/d/datasets."
            )
        if not os.path.isfile(split_path):
            raise LocalAssetError(f"split file not found: {split_path}")

        self.root = root
        self.split = split
        self.transform = transform
        self.split_path = split_path

        self.stem_to_class = read_list_txt(list_path)
        invalid = sorted({v for v in self.stem_to_class.values() if not (1 <= v <= 37)})
        if invalid:
            raise LocalAssetError(
                f"list.txt at '{list_path}' contains CLASS-ID values outside 1..37: {invalid}. "
                f"Cannot guarantee the 0-based [0, 36] label contract (CON-10)."
            )
        self.classes = build_class_names(self.stem_to_class)

        split_dict = load_split_file(split_path)  # already asserts disjoint internally
        assert_disjoint(split_dict)
        assert_subset(split_dict, all_ids=list(self.stem_to_class.keys()))

        self.ids = sorted(split_dict[split])
        self.samples = [(stem, self.stem_to_class[stem] - 1) for stem in self.ids]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        stem, label = self.samples[index]
        image_path = os.path.join(self.root, "images", stem + ".jpg")
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"failed to load image '{image_path}' (stem='{stem}'): {exc}") from exc

        target = torch.tensor(label, dtype=torch.long)
        if self.transform is not None:
            image, target = self.transform(image, target)
        else:
            import torchvision.transforms.v2.functional as F

            image = F.to_dtype(F.to_image(image), torch.float32, scale=True)

        return image, target
