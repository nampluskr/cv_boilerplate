from src.core.registry import TRANSFORMS


class IdentityTransform:
    """Toy images are already generated at the target size and normalized."""

    def __init__(self, image_size, train, **params):
        self.image_size = image_size
        self.train = train

    def __call__(self, image):
        return image


@TRANSFORMS.register("toy_train")
def build_toy_train_transform(image_size, train=True, **params):
    return IdentityTransform(image_size, train, **params)


@TRANSFORMS.register("toy_eval")
def build_toy_eval_transform(image_size, train=False, **params):
    return IdentityTransform(image_size, train, **params)
