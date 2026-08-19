def detection_collate(samples):
    """samples: list of (image, target). Returns (list[Tensor], list[dict]) (PLAN-P4 SS3.6).

    The default_collate would try to stack variable-length targets and fail; images are kept as
    a list (not stacked) even though every image is a fixed 512x512 after transform, because
    torchvision detection models expect a list input and this keeps the door open for a future
    variable-resolution dataset without touching the collate contract.
    """
    images = [sample[0] for sample in samples]
    targets = [sample[1] for sample in samples]
    return images, targets
