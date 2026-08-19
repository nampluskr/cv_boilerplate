import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.core.errors import LocalAssetError
from src.core.registry import DATASETS
from src.data.split import assert_disjoint, load_split_file

# Sample id scheme (PLAN-P5 SS3.2): train ids are "train_good/<stem>" (from category/train/good),
# valid/test ids are "<defect_type>/<stem>" (from category/test/<defect_type>), where defect_type
# is one of "good", "broken_large", "broken_small", "contamination". The "train_good" prefix
# (rather than reusing "good") avoids an id collision with test/good, whose files share the same
# zero-padded stem numbering as train/good.
TRAIN_PREFIX = "train_good"


@DATASETS.register("mvtec_anomaly")
class MVTecAnomaly(Dataset):
    """MVTec AD single-category anomaly dataset (PLAN-P5 SS3.4).

    train split returns (image, {}) -- normal-only, unlabeled.
    valid/test splits return (image, {"label": LongTensor scalar, "mask": LongTensor(H, W)}).
    label is 0=normal, 1=anomalous. mask is always present: a normal test image gets an
    all-zero (H, W) mask (never omitted -- excluding normal images from pixel metrics would
    overstate pixel-AUROC, PLAN-P5 SS3.1). Mask pixel values {0, 255} are converted to {0, 1}.
    """

    def __init__(self, root, split, transform=None, split_path=None, category="bottle", **params):
        if split not in ("train", "valid", "test"):
            raise ValueError(f"split must be one of train/valid/test, got '{split}'")
        if split_path is None:
            raise LocalAssetError(
                "MVTecAnomaly requires a split file path (data.split.path in the config, "
                "forwarded automatically by build_dataset() when data.split.mode=='file')."
            )

        category_dir = os.path.join(root, category)
        train_dir = os.path.join(category_dir, "train", "good")
        test_dir = os.path.join(category_dir, "test")
        ground_truth_dir = os.path.join(category_dir, "ground_truth")
        if not os.path.isdir(category_dir) or not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
            raise LocalAssetError(
                f"mvtec category '{category}' not found under root='{root}'. Expected "
                f"'{train_dir}' and '{test_dir}' to exist. This dataset is never downloaded "
                f"automatically (CON-03); place it under /mnt/d/datasets/mvtec."
            )
        if not os.path.isfile(split_path):
            raise LocalAssetError(f"split file not found: {split_path}")

        self.root = root
        self.category = category
        self.category_dir = category_dir
        self.ground_truth_dir = ground_truth_dir
        self.split = split
        self.transform = transform
        self.split_path = split_path
        # image-level classes for adapter/bind_class_names, mirroring the other tasks' .classes.
        self.classes = ["normal", "anomalous"]

        split_dict = load_split_file(split_path)  # already asserts disjoint internally
        assert_disjoint(split_dict)

        self.ids = sorted(split_dict[split])

    def __len__(self):
        return len(self.ids)

    def _image_path(self, defect_type, stem):
        if defect_type == TRAIN_PREFIX:
            return os.path.join(self.category_dir, "train", "good", stem + ".png")
        return os.path.join(self.category_dir, "test", defect_type, stem + ".png")

    def _mask_path(self, defect_type, stem):
        return os.path.join(self.ground_truth_dir, defect_type, stem + "_mask.png")

    def __getitem__(self, index):
        sample_id = self.ids[index]
        defect_type, stem = sample_id.split("/", 1)
        image_path = self._image_path(defect_type, stem)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"failed to load image '{image_path}' (sample_id='{sample_id}'): {exc}"
            ) from exc

        if self.split == "train":
            target = {}
            if self.transform is not None:
                return self.transform(image, target)
            import torchvision.transforms.v2.functional as F

            return F.to_dtype(F.to_image(image), torch.float32, scale=True), target

        is_anomalous = defect_type != "good"
        width, height = image.size
        if is_anomalous:
            mask_path = self._mask_path(defect_type, stem)
            try:
                mask_array = np.array(Image.open(mask_path).convert("L"))
            except Exception as exc:
                raise RuntimeError(
                    f"failed to load ground_truth mask '{mask_path}' (sample_id='{sample_id}'): {exc}"
                ) from exc
            # {0, 255} -> {0, 1} (PLAN-P5 SS3.1): leaving raw 255 values would break AUROC labels.
            mask_array = (mask_array > 0).astype(np.int64)
        else:
            # test/good has no ground_truth file; a normal image is an all-zero mask
            # (PLAN-P5 SS3.1), never excluded from the pixel-level target.
            mask_array = np.zeros((height, width), dtype=np.int64)

        mask = torch.from_numpy(mask_array).to(torch.int64)
        label = torch.tensor(int(is_anomalous), dtype=torch.long)
        target = {"label": label, "mask": mask}

        if self.transform is not None:
            image, target = self.transform(image, target)
        else:
            import torchvision.transforms.v2.functional as F

            image = F.to_dtype(F.to_image(image), torch.float32, scale=True)

        return image, target
