# YOLOv8n detection wrapper (PLAN-P4 SS4.3.3). This is the only file in the repository that is
# allowed to know about ultralytics' native tensor formats: batch_idx/cls/bboxes construction,
# cxcywh<->xyxy conversion, the nc<->num_classes off-by-one, and 0-based<->1-based label
# conversion all live here and must never leak into Dataset/adapter/engine (PLAN-P4 SS4.1).
#
# Only the low-level building blocks are used, never the high-level ultralytics.YOLO() API or
# its trainer (that trainer expects a dataset YAML, downloads assets, and does its own logging,
# which would violate this project's pure-PyTorch (CON-01) and offline (NFR-07) constraints):
#   - ultralytics.nn.tasks.DetectionModel  (raw nn.Module architecture)
#   - DetectionModel.loss() / ultralytics.utils.loss.v8DetectionLoss (loss, used via the public
#     DetectionModel.loss() entry point, which lazily builds v8DetectionLoss(self) internally)
#   - ultralytics.utils.nms.non_max_suppression (postprocess; import path corrected vs
#     PLAN-P4 SS4.3.3's ultralytics.utils.ops.non_max_suppression, which does not exist in the
#     installed ultralytics==8.4.101)
import torch
import torch.nn as nn

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.nms import non_max_suppression

from src.core.registry import MODELS

# model.0 .. model.21 are yolov8n's backbone+neck (class-count independent); model.22 is the
# Detect() head, whose channel counts depend on nc and therefore may legitimately mismatch a
# checkpoint trained with a different nc (PLAN-P4 SS4.3.3).
MAX_BACKBONE_NECK_LAYER_INDEX = 21


def is_backbone_or_neck_key(key):
    """Return True if a yolov8n state_dict key (e.g. "model.5.conv.weight") belongs to the
    backbone/neck (model.0 .. model.21) rather than the detection head (model.22.*)."""
    parts = key.split(".")
    if len(parts) < 2 or parts[0] != "model":
        return False
    try:
        layer_index = int(parts[1])
    except ValueError:
        return False
    return layer_index <= MAX_BACKBONE_NECK_LAYER_INDEX


@MODELS.register("yolov8n_det")
class YOLOv8nDetector(nn.Module):
    def __init__(self, num_classes=3, weights_path=None, score_thresh=0.05, nms_iou=0.5,
                 max_det=100, **params):
        super().__init__()
        # This project's num_classes includes background at index 0 (PLAN-P1 SS6.1 / CON-10:
        # 0=background, 1=cat, 2=dog -> num_classes=3). ultralytics' own `nc` convention is
        # foreground-only, so the conversion here is explicit and one-directional: every place
        # in this file that talks to ultralytics uses self.nc; every place that talks to the
        # rest of the codebase uses self.num_classes / 1-based labels.
        self.num_classes = num_classes
        self.nc = num_classes - 1
        self.score_thresh = score_thresh
        self.nms_iou = nms_iou
        self.max_det = max_det

        self.model = DetectionModel(cfg="yolov8n.yaml", ch=3, nc=self.nc, verbose=False)
        # v8DetectionLoss (invoked lazily by DetectionModel.loss()) reads box/cls/dfl loss gains
        # off model.args. DetectionModel does not set this itself outside of the ultralytics
        # Trainer, so the library's own default hyperparameters are supplied directly here
        # (get_cfg() only parses ultralytics' bundled default.yaml, no network access) without
        # going through the high-level YOLO()/Trainer API, which PLAN-P4 SS4.3.3 forbids.
        self.model.args = get_cfg()

        if weights_path is not None:
            self._load_pretrained_backbone(weights_path)

    def _load_pretrained_backbone(self, weights_path):
        # This checkpoint is not weights_only=True safe and does not hold a plain state_dict
        # (checkpoint["model"] is a full ultralytics DetectionModel instance with nc=80/COCO),
        # so src.core.offline.load_local_weights (weights_only=True, plain state_dict) cannot
        # be used here; load it directly instead.
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        pretrained_model = checkpoint["model"].float()
        pretrained_state_dict = pretrained_model.state_dict()

        # torch's load_state_dict(strict=False) tolerates missing/unexpected keys but still
        # raises on shape mismatches, so shape-mismatched keys (expected for the nc-dependent
        # detection head, e.g. model.22.cv3.*) are pre-filtered out here and folded into
        # `missing` below, exactly as if the key were absent from the checkpoint.
        # Excluding a shape-mismatched key from filtered_state_dict makes load_state_dict
        # report it via its own `missing` list below (since own_state_dict still has the key
        # but filtered_state_dict does not) -- no separate bookkeeping needed.
        own_state_dict = self.model.state_dict()
        filtered_state_dict = {
            key: value for key, value in pretrained_state_dict.items()
            if key in own_state_dict and own_state_dict[key].shape == value.shape
        }

        missing, unexpected = self.model.load_state_dict(filtered_state_dict, strict=False)
        missing = list(missing)
        unexpected = list(unexpected)

        bad_missing = [key for key in missing if is_backbone_or_neck_key(key)]
        bad_unexpected = [key for key in unexpected if is_backbone_or_neck_key(key)]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                f"yolov8n_det: backbone/neck weight keys failed to load from {weights_path}: "
                f"missing={bad_missing}, unexpected={bad_unexpected}"
            )
        print(
            f"[yolov8n_det] loaded local pretrained weights from {weights_path} "
            f"(nc={self.nc}); benign head-only mismatches: "
            f"missing={missing}, unexpected={list(unexpected)}"
        )

    def _build_ultralytics_batch(self, images, targets):
        """Convert this project's {boxes(xyxy abs), labels(1-based)} targets into ultralytics'
        {batch_idx, cls(0-based), bboxes(cxcywh, normalized), img} batch dict. N=0 images simply
        contribute zero rows; the batch/image is never skipped."""
        device = images[0].device
        img = torch.stack(images, dim=0)
        _, _, height, width = img.shape

        batch_idx_parts = []
        cls_parts = []
        bboxes_parts = []
        for image_index, target in enumerate(targets):
            boxes = target["boxes"]
            labels = target["labels"]
            n = boxes.shape[0]
            if n == 0:
                continue
            x1, y1, x2, y2 = boxes.unbind(dim=1)
            cx = (x1 + x2) / 2.0 / width
            cy = (y1 + y2) / 2.0 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            bboxes_parts.append(torch.stack([cx, cy, w, h], dim=1))
            cls_parts.append((labels.to(torch.float32) - 1.0).view(-1, 1))  # 1-based -> 0-based
            batch_idx_parts.append(torch.full((n,), image_index, dtype=torch.float32))

        if len(bboxes_parts) == 0:
            bboxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
            cls = torch.zeros((0, 1), dtype=torch.float32, device=device)
            batch_idx = torch.zeros((0,), dtype=torch.float32, device=device)
        else:
            bboxes = torch.cat(bboxes_parts, dim=0).to(device)
            cls = torch.cat(cls_parts, dim=0).to(device)
            batch_idx = torch.cat(batch_idx_parts, dim=0).to(device)

        return {"img": img, "batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}

    def train_step(self, images, targets):
        self.model.train()
        batch = self._build_ultralytics_batch(images, targets)
        loss, loss_components = self.model.loss(batch)
        total_loss = loss.sum()
        loss_dict = {
            "box_loss": float(loss_components[0].detach()),
            "cls_loss": float(loss_components[1].detach()),
            "dfl_loss": float(loss_components[2].detach()),
        }
        return {"loss": total_loss, "loss_dict": loss_dict}

    def forward(self, images):
        """Inference only. Raw ultralytics predictions are decoded and NMS-filtered here, then
        converted back to this project's contract: absolute xyxy boxes and 1-based labels."""
        self.model.eval()
        img = torch.stack(images, dim=0)
        height, width = img.shape[-2], img.shape[-1]

        with torch.no_grad():
            raw_output = self.model(img)
            # Detect.forward in eval mode returns (y, preds): y is the decoded, absolute-pixel
            # xyxy + per-class-score tensor of shape (batch, 4 + nc, num_anchors) that
            # non_max_suppression expects.
            predictions, _ = raw_output
            detections = non_max_suppression(
                predictions, conf_thres=self.score_thresh, iou_thres=self.nms_iou,
                max_det=self.max_det, nc=self.nc,
            )

        # Stay on the input device here, like custom_fcos/fasterrcnn_r50_fpn -- the adapter
        # (not the model) moves predictions to CPU when it needs to (update_metrics/
        # predict_step), so profile_model()'s FPS measurement doesn't pay a device-to-host
        # copy for this model only (PLAN-P4 SS4.1/SS4.2/SS8.3 fair-comparison control).
        results = []
        for det in detections:
            if det.numel() == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32, device=img.device)
                scores = torch.zeros((0,), dtype=torch.float32, device=img.device)
                labels = torch.zeros((0,), dtype=torch.long, device=img.device)
            else:
                boxes = det[:, :4].detach().clone()
                boxes[:, 0].clamp_(0, width)
                boxes[:, 2].clamp_(0, width)
                boxes[:, 1].clamp_(0, height)
                boxes[:, 3].clamp_(0, height)
                scores = det[:, 4].detach()
                labels = det[:, 5].detach().long() + 1  # 0-based -> this project's 1-based
            results.append({"boxes": boxes, "scores": scores, "labels": labels})
        return results
