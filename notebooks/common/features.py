import matplotlib.pyplot as plt


class FeatureMapRecorder:
    """Registers a forward hook on one named submodule (as returned by model.named_modules())
    and records its output tensor on the next forward pass. This only observes activations
    already computed by the model's own forward(); it does not change model behavior."""

    def __init__(self, model, layer_name):
        modules = dict(model.named_modules())
        if layer_name not in modules:
            raise KeyError(f"'{layer_name}' is not a submodule name of this model. "
                            f"Available: {sorted(modules.keys())[:20]} ...")
        self.layer_name = layer_name
        self.output = None
        self.handle = modules[layer_name].register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self.output = output.detach().cpu()

    def remove(self):
        self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove()


def show_feature_maps(feature_map, num_channels=8, title=None):
    """feature_map: (C, H, W) or (B, C, H, W) tensor captured by FeatureMapRecorder. Displays
    the first num_channels channels of the first batch item as a grid of grayscale images."""
    if feature_map.dim() == 4:
        feature_map = feature_map[0]
    num_channels = min(num_channels, feature_map.shape[0])
    fig, axes = plt.subplots(1, num_channels, figsize=(2 * num_channels, 2.2))
    if num_channels == 1:
        axes = [axes]
    for i in range(num_channels):
        axes[i].imshow(feature_map[i], cmap="viridis")
        axes[i].set_title(f"ch {i}", fontsize=8)
        axes[i].axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()
