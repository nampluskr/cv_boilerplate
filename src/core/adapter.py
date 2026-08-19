from abc import ABC, abstractmethod


class TaskAdapter(ABC):
    """Encapsulates how one batch is forwarded, how loss and predictions are produced,
    and how metrics are updated. The engine knows nothing else about a task."""

    def __init__(self, loss_fn, metrics, **params):
        self.loss_fn = loss_fn
        self.metrics = metrics

    # --- required ---
    @abstractmethod
    def train_step(self, model, batch, device) -> dict:
        """Return {"loss": scalar Tensor with grad, "loss_dict": {str: float}}."""

    @abstractmethod
    def eval_step(self, model, batch, device) -> dict:
        """Return {"loss": scalar Tensor or None, "outputs": Any}."""

    @abstractmethod
    def update_metrics(self, outputs) -> None:
        ...

    @abstractmethod
    def compute_metrics(self) -> dict:
        ...

    @abstractmethod
    def reset_metrics(self) -> None:
        ...

    @abstractmethod
    def predict_step(self, model, batch, device) -> list:
        """Return one serializable prediction per sample in the batch."""

    @abstractmethod
    def batch_size(self, batch) -> int:
        ...

    # --- optional, default no-op ---
    def collate_fn(self):
        return None

    def to(self, device):
        return self

    def on_fit_start(self, model, loaders, device):
        pass

    def on_fit_end(self, model, loaders, device):
        pass

    def on_epoch_start(self, model, epoch):
        pass

    def on_epoch_end(self, model, epoch, results):
        pass

    def save_predictions(self, predictions, output_dir):
        pass

    def visualize(self, batch, predictions, output_dir, max_items):
        pass

    def bind_class_names_from_config(self, data_config):
        """Bind class names from config['data'] alone, for CLI paths (predict) that have no
        Dataset instance to read .classes from (bind_class_names in src/cli/commands.py handles
        the train/evaluate paths, which do have one). Default no-op; override in tasks whose
        adapter carries a class_names attribute used to label predictions."""
        pass

    def extra_final_metrics(self) -> dict:
        """Extra fields to merge into metrics_final.json after training completes (PLAN-P1 SS16
        Grade B). Default is empty -- most tasks have nothing beyond the monitored valid metric.
        A task adapter that decides some threshold or calibration constant in on_fit_end (using
        only the valid loader) overrides this to report it."""
        return {}

    def dummy_forward_input(self, image_size, device):
        """Return a dummy input matching this task's model.forward() calling convention, used
        only for profiling (src/bench/profile.py: params/FLOPs/FPS measurement). Default is the
        single-batched-Tensor convention (forward(images: Tensor(B, C, H, W))) most task models
        use. A task whose model.forward() takes a different calling convention overrides this in
        its own adapter -- routing the convention through the adapter (like collate_fn/batch_size
        already do) keeps profile.py task-agnostic instead of guessing/duck-typing per model
        (PLAN-P1 SS16 Grade B)."""
        import torch

        return torch.zeros(1, 3, image_size[0], image_size[1], device=device)
