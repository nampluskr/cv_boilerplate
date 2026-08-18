import torch


def det_collate(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def anomaly_collate(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets
