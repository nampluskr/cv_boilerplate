# Custom 1-stage anchor-free detector (FCOS-style), implemented from scratch.
# PLAN-P4 SS4.3.1: backbone shared with custom_cnn_cls (P2) / custom_unet_seg (P3),
# FPN neck (P3/P4/P5, stride 8/16/32), weight-shared head (cls + reg + centerness),
# center-sampling assigner with per-level scale ranges, focal + GIoU + BCE loss.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms, generalized_box_iou_loss

from src.core.registry import MODELS


class ConvBlock(nn.Module):
    # Basic block used throughout the custom backbone (shared pattern with
    # custom_cnn_cls / custom_unet_seg). Kept local per the "only touch your file" rule.
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CustomFcosBackbone(nn.Module):
    # 5 stages, each one ConvBlock with stride=2, channel progression
    # 3 -> 32 -> 64 -> 128 -> 256 -> 512, cumulative stride 2/4/8/16/32.
    def __init__(self):
        super().__init__()
        self.stage1 = ConvBlock(3, 32)
        self.stage2 = ConvBlock(32, 64)
        self.stage3 = ConvBlock(64, 128)
        self.stage4 = ConvBlock(128, 256)
        self.stage5 = ConvBlock(256, 512)

    def forward(self, images):
        c1 = self.stage1(images)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c3, c4, c5


class FpnNeck(nn.Module):
    # Standard top-down FPN: lateral 1x1 convs + 3x3 smoothing convs + nearest upsample-and-add.
    # Takes C3/C4/C5 (stride 8/16/32) and produces P3/P4/P5 (same strides), channel width 128.
    def __init__(self, in_channels=(128, 256, 512), out_channels=128):
        super().__init__()
        self.lateral3 = nn.Conv2d(in_channels[0], out_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(in_channels[1], out_channels, kernel_size=1)
        self.lateral5 = nn.Conv2d(in_channels[2], out_channels, kernel_size=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, c3, c4, c5):
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        return p3, p4, p5


class FcosHead(nn.Module):
    # Weight-shared head across all FPN levels. Classification branch outputs
    # num_classes-1 channels (per-class sigmoid logits), regression branch outputs
    # 4 channels (l,t,r,b, pre-exp), centerness branch outputs 1 channel (sigmoid logit).
    def __init__(self, in_channels, num_foreground_classes, num_convs=4):
        super().__init__()
        cls_tower = []
        reg_tower = []
        for _ in range(num_convs):
            cls_tower += [nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                          nn.GroupNorm(32, in_channels), nn.ReLU(inplace=True)]
            reg_tower += [nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                          nn.GroupNorm(32, in_channels), nn.ReLU(inplace=True)]
        self.cls_tower = nn.Sequential(*cls_tower)
        self.reg_tower = nn.Sequential(*reg_tower)
        self.cls_logits = nn.Conv2d(in_channels, num_foreground_classes, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.centerness_logits = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        # Standard focal-loss prior init: bias so initial foreground probability is low.
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, feature, stride):
        cls_feature = self.cls_tower(feature)
        reg_feature = self.reg_tower(feature)
        cls_logits = self.cls_logits(cls_feature)
        centerness_logits = self.centerness_logits(reg_feature)
        # Regression is exp() scaled by stride to keep distances positive and level-appropriate.
        reg_dist = torch.exp(self.reg_pred(reg_feature)) * stride
        return cls_logits, reg_dist, centerness_logits


def _compute_locations(height, width, stride, device):
    # Feature-map locations mapped back to image coordinates (center of each cell).
    shifts_x = torch.arange(0, width, dtype=torch.float32, device=device) * stride + stride / 2
    shifts_y = torch.arange(0, height, dtype=torch.float32, device=device) * stride + stride / 2
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    locations = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1)], dim=1)
    return locations


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    # Standard binary focal loss, summed over all elements (per-class independent sigmoids).
    prob = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.sum()


@MODELS.register("custom_fcos_det")
class CustomFCOSDetector(nn.Module):
    def __init__(self, num_classes=3, weights_path=None, score_thresh=0.05, nms_iou=0.5,
                 max_det=100, fpn_channels=128, scale_ranges=None, pre_nms_topk=1000, **params):
        """num_classes includes background (index 0). oxford_pets uses 3.

        weights_path is accepted for interface consistency but unused: this model is
        trained from scratch (PLAN-P4 SS4.3.1), that is the point of the custom-vs-pretrained
        comparison axis.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_foreground_classes = num_classes - 1
        self.score_thresh = score_thresh
        self.nms_iou = nms_iou
        self.max_det = max_det
        self.pre_nms_topk = pre_nms_topk
        self.strides = (8, 16, 32)

        # Per-level scale-range assignment (simplified FCOS): P3 handles small boxes,
        # P4 medium, P5 the rest. Thresholds chosen for a 512x512 input: T1=64, T2=256.
        # A box is assigned to level i if T[i] <= max(l,t,r,b)-of-GT-side < T[i+1].
        if scale_ranges is None:
            scale_ranges = [(0, 64), (64, 256), (256, float("inf"))]
        self.scale_ranges = scale_ranges

        self.backbone = CustomFcosBackbone()
        self.neck = FpnNeck(in_channels=(128, 256, 512), out_channels=fpn_channels)
        self.head = FcosHead(fpn_channels, self.num_foreground_classes)

    def _forward_features(self, images):
        c3, c4, c5 = self.backbone(images)
        p3, p4, p5 = self.neck(c3, c4, c5)
        features = (p3, p4, p5)

        cls_logits_list = []
        reg_dist_list = []
        centerness_list = []
        locations_list = []
        for feature, stride in zip(features, self.strides):
            cls_logits, reg_dist, centerness_logits = self.head(feature, stride)
            height, width = feature.shape[-2:]
            locations = _compute_locations(height, width, stride, feature.device)

            cls_logits_list.append(cls_logits.permute(0, 2, 3, 1).reshape(cls_logits.shape[0], -1,
                                                                            self.num_foreground_classes))
            reg_dist_list.append(reg_dist.permute(0, 2, 3, 1).reshape(reg_dist.shape[0], -1, 4))
            centerness_list.append(centerness_logits.permute(0, 2, 3, 1).reshape(centerness_logits.shape[0], -1))
            locations_list.append(locations)

        return cls_logits_list, reg_dist_list, centerness_list, locations_list

    def _assign_targets(self, locations, stride, scale_range, boxes, labels):
        # locations: (L, 2) image-coordinate centers for this level.
        # boxes: (N, 4) xyxy, labels: (N,) in 1..num_foreground_classes.
        # Returns per-location: cls_target (L, num_foreground_classes), reg_target (L, 4) ltrb,
        # centerness_target (L,), pos_mask (L,) bool.
        num_locations = locations.shape[0]
        device = locations.device
        cls_target = torch.zeros(num_locations, self.num_foreground_classes, device=device)
        reg_target = torch.zeros(num_locations, 4, device=device)
        centerness_target = torch.zeros(num_locations, device=device)
        pos_mask = torch.zeros(num_locations, dtype=torch.bool, device=device)

        if boxes.numel() == 0:
            return cls_target, reg_target, centerness_target, pos_mask

        num_boxes = boxes.shape[0]
        xs = locations[:, 0].unsqueeze(1)  # (L, 1)
        ys = locations[:, 1].unsqueeze(1)  # (L, 1)

        left = xs - boxes[:, 0].unsqueeze(0)
        top = ys - boxes[:, 1].unsqueeze(0)
        right = boxes[:, 2].unsqueeze(0) - xs
        bottom = boxes[:, 3].unsqueeze(0) - ys
        ltrb = torch.stack([left, top, right, bottom], dim=2)  # (L, N, 4)

        is_inside_box = ltrb.min(dim=2).values > 0  # (L, N)

        # Center sampling: location must additionally lie within a small radius of the
        # GT box center, scaled by stride (standard FCOS center-sampling radius = 1.5 cells).
        radius = 1.5 * stride
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        center_dist_x = (xs - cx.unsqueeze(0)).abs()
        center_dist_y = (ys - cy.unsqueeze(0)).abs()
        is_in_center = (center_dist_x < radius) & (center_dist_y < radius)

        max_reg = ltrb.max(dim=2).values  # (L, N), the scale-range matching value
        low, high = scale_range
        is_in_scale_range = (max_reg >= low) & (max_reg < high)

        is_candidate = is_inside_box & is_in_center & is_in_scale_range  # (L, N)

        box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # (N,)
        area_for_loc = box_area.unsqueeze(0).repeat(num_locations, 1)
        area_for_loc[~is_candidate] = float("inf")
        min_area, min_area_idx = area_for_loc.min(dim=1)

        pos_mask = torch.isfinite(min_area)
        if pos_mask.sum() == 0:
            return cls_target, reg_target, centerness_target, pos_mask

        pos_indices = pos_mask.nonzero(as_tuple=True)[0]
        matched_box_idx = min_area_idx[pos_indices]
        matched_ltrb = ltrb[pos_indices, matched_box_idx]  # (P, 4)
        matched_labels = labels[matched_box_idx]  # (P,), 1-based

        reg_target[pos_indices] = matched_ltrb
        cls_target[pos_indices, matched_labels - 1] = 1.0

        l, t, r, b = matched_ltrb.unbind(dim=1)
        centerness = torch.sqrt(
            (torch.min(l, r) / torch.clamp(torch.max(l, r), min=1e-6)) *
            (torch.min(t, b) / torch.clamp(torch.max(t, b), min=1e-6))
        )
        centerness_target[pos_indices] = centerness

        return cls_target, reg_target, centerness_target, pos_mask

    def train_step(self, images, targets):
        """images: list[Tensor(3, H, W)], targets: list[dict{boxes, labels}].
        returns: {"loss": scalar Tensor with grad, "loss_dict": {str: float}}"""
        device = next(self.parameters()).device
        image_batch = torch.stack(images, dim=0).to(device)

        cls_logits_list, reg_dist_list, centerness_list, locations_list = self._forward_features(image_batch)

        batch_size = image_batch.shape[0]
        num_levels = len(self.strides)

        total_cls_loss = image_batch.new_zeros(())
        total_reg_loss = image_batch.new_zeros(())
        total_centerness_loss = image_batch.new_zeros(())
        total_num_pos = 0

        for image_idx in range(batch_size):
            boxes = targets[image_idx]["boxes"].to(device)
            labels = targets[image_idx]["labels"].to(device)

            for level_idx in range(num_levels):
                locations = locations_list[level_idx]
                stride = self.strides[level_idx]
                scale_range = self.scale_ranges[level_idx]

                cls_target, reg_target, centerness_target, pos_mask = self._assign_targets(
                    locations, stride, scale_range, boxes, labels
                )

                cls_logits = cls_logits_list[level_idx][image_idx]
                total_cls_loss = total_cls_loss + sigmoid_focal_loss(cls_logits, cls_target)

                num_pos = int(pos_mask.sum().item())
                total_num_pos += num_pos
                if num_pos == 0:
                    continue

                reg_dist = reg_dist_list[level_idx][image_idx][pos_mask]
                reg_tgt = reg_target[pos_mask]
                loc = locations[pos_mask]

                # Decode predicted and target ltrb into xyxy absolute boxes for GIoU loss.
                pred_boxes = torch.stack([
                    loc[:, 0] - reg_dist[:, 0], loc[:, 1] - reg_dist[:, 1],
                    loc[:, 0] + reg_dist[:, 2], loc[:, 1] + reg_dist[:, 3],
                ], dim=1)
                gt_boxes = torch.stack([
                    loc[:, 0] - reg_tgt[:, 0], loc[:, 1] - reg_tgt[:, 1],
                    loc[:, 0] + reg_tgt[:, 2], loc[:, 1] + reg_tgt[:, 3],
                ], dim=1)

                centerness_weight = centerness_target[pos_mask]
                giou_loss = generalized_box_iou_loss(pred_boxes, gt_boxes, reduction="none")
                total_reg_loss = total_reg_loss + (giou_loss * centerness_weight).sum()

                centerness_logits = centerness_list[level_idx][image_idx][pos_mask]
                total_centerness_loss = total_centerness_loss + F.binary_cross_entropy_with_logits(
                    centerness_logits, centerness_weight, reduction="sum"
                )

        # Normalize by number of positive locations (clamped to avoid div-by-zero when a
        # batch happens to contain zero GT boxes across all images).
        normalizer = max(total_num_pos, 1)
        cls_loss = total_cls_loss / normalizer
        reg_loss = total_reg_loss / normalizer
        centerness_loss = total_centerness_loss / normalizer

        loss = cls_loss + reg_loss + centerness_loss

        loss_dict = {
            "loss": float(loss.detach().item()),
            "cls_loss": float(cls_loss.detach().item()),
            "reg_loss": float(reg_loss.detach().item()),
            "centerness_loss": float(centerness_loss.detach().item()),
        }
        return {"loss": loss, "loss_dict": loss_dict}

    @torch.no_grad()
    def forward(self, images):
        """Inference only. images: list[Tensor(3, H, W)].
        returns: list[dict{"boxes": (M,4) xyxy, "scores": (M,), "labels": (M,) long}]
        already filtered by score_thresh / nms_iou / max_det."""
        device = next(self.parameters()).device
        image_batch = torch.stack(images, dim=0).to(device)
        batch_size = image_batch.shape[0]

        cls_logits_list, reg_dist_list, centerness_list, locations_list = self._forward_features(image_batch)

        results = []
        for image_idx in range(batch_size):
            all_boxes = []
            all_scores = []
            all_labels = []

            for level_idx in range(len(self.strides)):
                cls_logits = cls_logits_list[level_idx][image_idx]  # (L, C)
                reg_dist = reg_dist_list[level_idx][image_idx]  # (L, 4)
                centerness_logits = centerness_list[level_idx][image_idx]  # (L,)
                locations = locations_list[level_idx]  # (L, 2)

                cls_scores = torch.sigmoid(cls_logits)
                centerness_scores = torch.sigmoid(centerness_logits).unsqueeze(1)
                scores = cls_scores * centerness_scores  # (L, C)

                scores_flat = scores.reshape(-1)
                num_candidates = min(self.pre_nms_topk, scores_flat.numel())
                if num_candidates == 0:
                    continue
                top_scores, top_idx = scores_flat.topk(num_candidates)
                keep = top_scores > self.score_thresh
                if keep.sum() == 0:
                    continue
                top_scores = top_scores[keep]
                top_idx = top_idx[keep]

                loc_idx = top_idx // self.num_foreground_classes
                class_idx = top_idx % self.num_foreground_classes

                loc = locations[loc_idx]
                dist = reg_dist[loc_idx]
                decoded_boxes = torch.stack([
                    loc[:, 0] - dist[:, 0], loc[:, 1] - dist[:, 1],
                    loc[:, 0] + dist[:, 2], loc[:, 1] + dist[:, 3],
                ], dim=1)

                all_boxes.append(decoded_boxes)
                all_scores.append(top_scores)
                all_labels.append(class_idx + 1)  # back to 1-based labels

            if len(all_boxes) == 0:
                results.append({
                    "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                    "scores": torch.zeros((0,), dtype=torch.float32, device=device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=device),
                })
                continue

            boxes = torch.cat(all_boxes, dim=0)
            scores = torch.cat(all_scores, dim=0)
            labels = torch.cat(all_labels, dim=0)

            # Clamp to image bounds.
            _, _, height, width = image_batch.shape
            boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=width)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=height)

            keep_idx = batched_nms(boxes, scores, labels, self.nms_iou)
            keep_idx = keep_idx[:self.max_det]

            results.append({
                "boxes": boxes[keep_idx].float(),
                "scores": scores[keep_idx].float(),
                "labels": labels[keep_idx].long(),
            })

        return results
