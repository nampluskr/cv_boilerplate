import numpy as np
import torch
from torch.utils.data import Dataset

from src.core.registry import DATASETS

SPLIT_SIZES = {"train": 64, "valid": 32, "test": 32}
SPLIT_OFFSETS = {"train": 0, "valid": 100000, "test": 200000}


def item_rng(seed, split, index):
    return np.random.default_rng(seed + SPLIT_OFFSETS[split] + index)


class ToyDatasetBase(Dataset):
    def __init__(self, root, split, transform=None, num_samples=None, seed=42, **params):
        self.root = root
        self.split = split
        self.transform = transform
        self.seed = seed
        self.num_samples = num_samples or SPLIT_SIZES[split]

    def __len__(self):
        return self.num_samples

    def apply_transform(self, image):
        return self.transform(image) if self.transform is not None else image


@DATASETS.register("toy_cls")
class ToyClsDataset(ToyDatasetBase):
    def __init__(self, root, split, transform=None, num_samples=None, seed=42,
                 image_size=(32, 32), num_classes=4, **params):
        super().__init__(root, split, transform, num_samples, seed, **params)
        self.image_size = image_size
        self.num_classes = num_classes

    def __getitem__(self, idx):
        rng = item_rng(self.seed, self.split, idx)
        h, w = self.image_size
        image = torch.from_numpy(rng.standard_normal((3, h, w)).astype(np.float32))
        label = int(rng.integers(0, self.num_classes))
        target = torch.tensor(label, dtype=torch.long)
        return self.apply_transform(image), target


@DATASETS.register("toy_seg")
class ToySegDataset(ToyDatasetBase):
    def __init__(self, root, split, transform=None, num_samples=None, seed=42,
                 image_size=(32, 32), num_classes=3, **params):
        super().__init__(root, split, transform, num_samples, seed, **params)
        self.image_size = image_size
        self.num_classes = num_classes

    def __getitem__(self, idx):
        rng = item_rng(self.seed, self.split, idx)
        h, w = self.image_size
        image = torch.from_numpy(rng.standard_normal((3, h, w)).astype(np.float32))
        mask = torch.from_numpy(rng.integers(0, self.num_classes, size=(h, w)).astype(np.int64))
        return self.apply_transform(image), mask


@DATASETS.register("toy_det")
class ToyDetDataset(ToyDatasetBase):
    def __init__(self, root, split, transform=None, num_samples=None, seed=42,
                 image_size=(64, 64), num_fg_classes=2, **params):
        super().__init__(root, split, transform, num_samples, seed, **params)
        self.image_size = image_size
        self.num_fg_classes = num_fg_classes

    def __getitem__(self, idx):
        rng = item_rng(self.seed, self.split, idx)
        h, w = self.image_size
        image = torch.from_numpy(rng.standard_normal((3, h, w)).astype(np.float32))
        num_boxes = idx % 4
        boxes = []
        labels = []
        for _ in range(num_boxes):
            box_w = int(rng.integers(6, max(7, w // 3)))
            box_h = int(rng.integers(6, max(7, h // 3)))
            x1 = int(rng.integers(0, max(1, w - box_w)))
            y1 = int(rng.integers(0, max(1, h - box_h)))
            boxes.append([x1, y1, x1 + box_w, y1 + box_h])
            labels.append(int(rng.integers(1, self.num_fg_classes + 1)))
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)
        target = {"boxes": boxes_tensor, "labels": labels_tensor}
        return self.apply_transform(image), target


@DATASETS.register("toy_anomaly")
class ToyAnomalyDataset(ToyDatasetBase):
    def __init__(self, root, split, transform=None, num_samples=None, seed=42,
                 image_size=(32, 32), **params):
        super().__init__(root, split, transform, num_samples, seed, **params)
        self.image_size = image_size

    def __getitem__(self, idx):
        rng = item_rng(self.seed, self.split, idx)
        h, w = self.image_size
        image = torch.from_numpy(rng.standard_normal((3, h, w)).astype(np.float32))

        if self.split == "train":
            return self.apply_transform(image), {}

        is_anomalous = idx % 2 == 1
        mask = np.zeros((h, w), dtype=np.float32)
        if is_anomalous:
            region_h = max(2, h // 4)
            region_w = max(2, w // 4)
            top = int(rng.integers(0, h - region_h))
            left = int(rng.integers(0, w - region_w))
            mask[top:top + region_h, left:left + region_w] = 1.0
            image[:, top:top + region_h, left:left + region_w] += 3.0
        target = {
            "label": torch.tensor(int(is_anomalous), dtype=torch.long),
            "mask": torch.from_numpy(mask),
        }
        return self.apply_transform(image), target
