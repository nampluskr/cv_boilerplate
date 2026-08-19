import os
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.core.errors import LocalAssetError
from src.core.registry import DATASETS
from src.data.split import assert_disjoint, load_split_file


def parse_voc_xml(xml_path, class_names, drop_difficult):
    """Parse one Pascal VOC annotation file with xml.etree.ElementTree (PLAN-P4 SS3.1: single-
    line XML, never parse with regex/grep). Returns (boxes, labels) as plain python lists;
    labels are 1-based indices into class_names (0 is reserved for background, PLAN-P4 SS3.4)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    labels = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in class_names:
            raise RuntimeError(
                f"'{xml_path}' has object name '{name}' which is not in class_names="
                f"{list(class_names)} (PLAN-P4 SS3.4, NFR-11: unknown labels are reported, "
                f"never silently dropped)."
            )
        difficult_text = obj.findtext("difficult")
        is_difficult = difficult_text is not None and int(difficult_text) == 1
        if drop_difficult and is_difficult:
            continue
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.findtext("xmin"))
        ymin = float(bndbox.findtext("ymin"))
        xmax = float(bndbox.findtext("xmax"))
        ymax = float(bndbox.findtext("ymax"))
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(class_names.index(name) + 1)
    return boxes, labels


@DATASETS.register("oxford_pets_det")
class OxfordPetsDetection(Dataset):
    """Pascal VOC XML detection dataset. Returns (image, target) where target is
    {"boxes": FloatTensor(N, 4) absolute xyxy, "labels": LongTensor(N) 1-based} (PLAN-P4 SS3.4).

    N=0 is a normal sample (empty (0, 4)/(0,) tensors, never None) -- both when an XML has no
    <object> and, more commonly downstream, after SanitizeBoundingBoxes drops every box in the
    train/eval transform (PLAN-P4 SS3.5). This dataset itself never drops a whole sample.

    New detection datasets (VOC/COCO variants) only need to satisfy this same (image, target)
    contract; the engine, adapter, model and metric never depend on oxford_pets being N<=1
    per image (PLAN-P4 SS2.1, NFR-06).
    """

    def __init__(self, root, split, transform=None, split_path=None,
                 class_names=("cat", "dog"), drop_difficult=False, **params):
        if split not in ("train", "valid", "test"):
            raise ValueError(f"split must be one of train/valid/test, got '{split}'")
        if split_path is None:
            raise LocalAssetError(
                "OxfordPetsDetection requires a split file path (data.split.path in the "
                "config, forwarded automatically by build_dataset() when data.split.mode=='file')."
            )

        images_dir = os.path.join(root, "images")
        xmls_dir = os.path.join(root, "annotations", "xmls")
        if not os.path.isdir(root) or not os.path.isdir(images_dir) or not os.path.isdir(xmls_dir):
            raise LocalAssetError(
                f"oxford_pets dataset not found under root='{root}'. Expected "
                f"'{images_dir}' and '{xmls_dir}' to exist. This dataset is never "
                f"downloaded automatically (CON-03); place it under /mnt/d/datasets."
            )
        if not os.path.isfile(split_path):
            raise LocalAssetError(f"split file not found: {split_path}")

        self.root = root
        self.split = split
        self.transform = transform
        self.split_path = split_path
        self.class_names = list(class_names)
        # index 0 is the reserved background class (PLAN-P4 SS2, CON-10); exposed as `classes`
        # so src/cli/commands.py's bind_class_names() picks it up like the other tasks do.
        self.classes = ["background"] + self.class_names
        self.drop_difficult = drop_difficult

        split_dict = load_split_file(split_path)  # already asserts disjoint internally
        assert_disjoint(split_dict)

        self.ids = sorted(split_dict[split])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        stem = self.ids[index]
        image_path = os.path.join(self.root, "images", stem + ".jpg")
        xml_path = os.path.join(self.root, "annotations", "xmls", stem + ".xml")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"failed to load image '{image_path}' (stem='{stem}'): {exc}") from exc
        try:
            boxes, labels = parse_voc_xml(xml_path, self.class_names, self.drop_difficult)
        except Exception as exc:
            raise RuntimeError(f"failed to parse VOC xml '{xml_path}' (stem='{stem}'): {exc}") from exc

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)
        target = {"boxes": boxes_tensor, "labels": labels_tensor}

        if self.transform is not None:
            image, target = self.transform(image, target)
        else:
            import torchvision.transforms.v2.functional as F

            image = F.to_dtype(F.to_image(image), torch.float32, scale=True)

        return image, target
