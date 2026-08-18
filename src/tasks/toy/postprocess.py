import torch


def decode_det_outputs(outputs, num_fg_classes=2, stride=8, score_threshold=0.5, image_size=(64, 64)):
    """Decode grid outputs (B, 1+num_fg_classes+4, Hc, Wc) into per-image boxes/scores/labels."""
    batch_size, _, hc, wc = outputs.shape
    obj_logits = outputs[:, 0, :, :]
    cls_logits = outputs[:, 1:1 + num_fg_classes, :, :]
    box_pred = outputs[:, 1 + num_fg_classes:, :, :]

    obj_scores = torch.sigmoid(obj_logits)
    cls_probs = torch.softmax(cls_logits, dim=1)

    results = []
    for b in range(batch_size):
        boxes, scores, labels = [], [], []
        for row in range(hc):
            for col in range(wc):
                score = float(obj_scores[b, row, col])
                if score < score_threshold:
                    continue
                cx = (col + 0.5) * stride
                cy = (row + 0.5) * stride
                l, t, r, btm = box_pred[b, :, row, col].tolist()
                x1 = max(0.0, cx - l * stride)
                y1 = max(0.0, cy - t * stride)
                x2 = min(float(image_size[1]), cx + r * stride)
                y2 = min(float(image_size[0]), cy + btm * stride)
                if x2 <= x1 or y2 <= y1:
                    continue
                class_id = int(torch.argmax(cls_probs[b, :, row, col]).item()) + 1
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                labels.append(class_id)
        if boxes:
            results.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "scores": torch.tensor(scores, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
            })
        else:
            results.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.long),
            })
    return results
