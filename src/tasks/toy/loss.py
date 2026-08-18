import torch
import torch.nn as nn

from src.core.registry import LOSSES

LOSSES.register("cross_entropy")(nn.CrossEntropyLoss)
LOSSES.register("mse")(nn.MSELoss)


@LOSSES.register("toy_det_loss")
class ToyDetLoss(nn.Module):
    """Anchor-free grid loss: assigns each GT box to the grid cell containing its center."""

    def __init__(self, num_fg_classes=2, stride=8, **params):
        super().__init__()
        self.num_fg_classes = num_fg_classes
        self.stride = stride
        self.obj_loss_fn = nn.BCEWithLogitsLoss()
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.box_loss_fn = nn.L1Loss()

    def forward(self, outputs, targets):
        device = outputs.device
        batch_size, _, hc, wc = outputs.shape
        obj_logits = outputs[:, 0, :, :]
        cls_logits = outputs[:, 1:1 + self.num_fg_classes, :, :]
        box_pred = outputs[:, 1 + self.num_fg_classes:, :, :]

        obj_target = torch.zeros((batch_size, hc, wc), device=device)
        cls_target = torch.zeros((batch_size, hc, wc), dtype=torch.long, device=device)
        box_target = torch.zeros((batch_size, 4, hc, wc), device=device)
        pos_mask = torch.zeros((batch_size, hc, wc), dtype=torch.bool, device=device)

        for b, target in enumerate(targets):
            boxes = target["boxes"]
            labels = target["labels"]
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box.tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                col = min(max(int(cx / self.stride), 0), wc - 1)
                row = min(max(int(cy / self.stride), 0), hc - 1)
                obj_target[b, row, col] = 1.0
                cls_target[b, row, col] = int(label.item()) - 1
                box_target[b, 0, row, col] = (cx - x1) / self.stride
                box_target[b, 1, row, col] = (cy - y1) / self.stride
                box_target[b, 2, row, col] = (x2 - cx) / self.stride
                box_target[b, 3, row, col] = (y2 - cy) / self.stride
                pos_mask[b, row, col] = True

        obj_loss = self.obj_loss_fn(obj_logits, obj_target)

        if pos_mask.any():
            cls_logits_pos = cls_logits.permute(0, 2, 3, 1)[pos_mask]
            cls_target_pos = cls_target[pos_mask]
            cls_loss = self.cls_loss_fn(cls_logits_pos, cls_target_pos)

            box_pred_pos = box_pred.permute(0, 2, 3, 1)[pos_mask]
            box_target_pos = box_target.permute(0, 2, 3, 1)[pos_mask]
            box_loss = self.box_loss_fn(box_pred_pos, box_target_pos)
        else:
            cls_loss = torch.zeros((), device=device)
            box_loss = torch.zeros((), device=device)

        return obj_loss + cls_loss + box_loss
