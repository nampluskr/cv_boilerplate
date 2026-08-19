import torch
import torchvision.transforms.v2.functional as F


def smooth_anomaly_map(anomaly_map, sigma):
    """Gaussian smoothing applied identically to every model's anomaly_map (PLAN-P5 SS7.3):
    control requires the same postprocessing across the model comparison, so smoothing lives
    here in the shared adapter, never inside a model file.

    anomaly_map: Tensor(B, H, W) float, higher = more anomalous. Returns the same shape.
    sigma <= 0 is a no-op (kept for cheap ablation via config, not used by default).
    """
    if sigma is None or sigma <= 0:
        return anomaly_map
    # gaussian_blur wants a channel dim; kernel_size must be odd and large enough to approximate
    # the given sigma (torchvision has no direct sigma-only API).
    kernel_size = max(3, int(2 * round(3 * sigma) + 1))
    smoothed = F.gaussian_blur(anomaly_map.unsqueeze(1), kernel_size=kernel_size, sigma=sigma)
    return smoothed.squeeze(1)


def best_f1_threshold(scores, labels):
    """Sweep every distinct score as a candidate threshold ("predict positive if score >=
    threshold") and return the one maximizing F1, via one O(n log n) sort instead of an O(n^2)
    or torchmetrics.BinaryF1Score-per-threshold loop -- this must scale to the pixel-level sweep
    (50 valid images x 256 x 256 ~= 2M scores, PLAN-P5 SS7.1).

    scores, labels: 1-D Tensors of equal length. labels in {0, 1}. Returns a python float.
    Degenerate case (no positives or empty input): returns 0.0, an inert threshold, since F1 is
    undefined there and no image/pixel in this split calls for review-time triage.
    """
    if scores.numel() == 0 or labels.sum() == 0:
        return 0.0

    sorted_scores, order = torch.sort(scores, descending=True)
    sorted_labels = labels[order].float()

    tp_cum = torch.cumsum(sorted_labels, dim=0)
    predicted_count = torch.arange(1, sorted_labels.numel() + 1, dtype=torch.float32, device=scores.device)
    precision = tp_cum / predicted_count
    total_positive = sorted_labels.sum()
    recall = tp_cum / total_positive
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)

    # Tied scores must be predicted positive/negative together ("score >= threshold" cannot
    # split a tie), so only the LAST cumulative position within each run of equal sorted_scores
    # is a valid candidate threshold -- any earlier position within a tie represents a predicted
    # set that "score >= threshold" can never actually produce. Mask every other position to -inf
    # before argmax (Codex A5 Major #1: unmasked ties could return a wrong-F1 threshold).
    is_last_in_run = torch.ones_like(sorted_scores, dtype=torch.bool)
    is_last_in_run[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    f1 = torch.where(is_last_in_run, f1, torch.full_like(f1, float("-inf")))

    best_index = int(torch.argmax(f1))
    return float(sorted_scores[best_index])


def compute_thresholds(model, valid_loader, device, smooth_sigma):
    """valid-only threshold decision (PLAN-P5 SS6, NFR-03): image_threshold maximizes F1 on
    valid image scores/labels, pixel_threshold maximizes F1 on valid (smoothed) pixel
    scores/masks. Never sees test. Called from AnomalyAdapter.on_fit_end, which itself only
    receives {"train", "valid"} loaders (PLAN-P5 SS5) -- there is no test loader in scope here
    to leak from.
    """
    model.eval()
    image_scores, image_labels = [], []
    pixel_scores, pixel_labels = [], []

    with torch.no_grad():
        for images, targets in valid_loader:
            images = images.to(device)
            outputs = model(images)
            maps = smooth_anomaly_map(outputs["anomaly_map"], smooth_sigma)
            labels = torch.stack([t["label"] for t in targets]).to(device)
            masks = torch.stack([t["mask"] for t in targets]).to(device)

            image_scores.append(outputs["pred_score"].detach().cpu())
            image_labels.append(labels.detach().cpu())
            pixel_scores.append(maps.flatten().detach().cpu())
            pixel_labels.append(masks.flatten().detach().cpu())

    image_threshold = best_f1_threshold(torch.cat(image_scores), torch.cat(image_labels))
    pixel_threshold = best_f1_threshold(torch.cat(pixel_scores), torch.cat(pixel_labels))
    return image_threshold, pixel_threshold
