import torch


def anomaly_collate(batch):
    """Stacks images into a single Tensor(B, 3, H, W); keeps targets as list-of-dict so a
    train split's empty {} targets and a valid/test split's {"label", "mask"} targets both pass
    through unchanged (PLAN-P5 SS7.2)."""
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets
