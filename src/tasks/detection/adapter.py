from src.core.adapter import TaskAdapter
from src.core.registry import ADAPTERS
from src.tasks.detection.collate import detection_collate


def move_targets_to_device(targets, device):
    return [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]


def detach_to_cpu(dict_list):
    return [{k: v.detach().cpu() for k, v in d.items()} for d in dict_list]


@ADAPTERS.register("detection")
class DetectionAdapter(TaskAdapter):
    """PLAN-P4 SS7. Loss lives inside the model (SS4.1/SS5); this adapter never calls
    self.loss_fn. eval_step's loss is always None (SS7): torchvision detection models don't
    return a loss in eval() mode, and the three models' losses aren't comparable across
    architectures anyway, so model selection uses map_50_95 (train.monitor), not valid loss."""

    def __init__(self, loss_fn, metrics, class_names=None, **params):
        super().__init__(loss_fn, metrics, **params)
        # class_names[0] is the reserved background label; bound by
        # src/cli/commands.py::bind_class_names() from OxfordPetsDetection.classes.
        self.class_names = class_names or ["background", "cat", "dog"]

    def collate_fn(self):
        return detection_collate

    def batch_size(self, batch):
        return len(batch[0])

    def bind_class_names_from_config(self, data_config):
        # PLAN-P4 SS2.1 generality: predict has no Dataset, so pull the same class_names the
        # Dataset would have derived (["background"] + data.params.class_names), instead of
        # silently keeping the __init__ default (["background", "cat", "dog"]), which would
        # mislabel any dataset whose foreground classes are not exactly [cat, dog].
        class_names = data_config.get("params", {}).get("class_names")
        if class_names:
            self.class_names = ["background"] + list(class_names)

    def dummy_forward_input(self, image_size, device):
        # Detection models take forward(images: list[Tensor(3, H, W)]) (PLAN-P4 SS4.1), unlike
        # the batched-Tensor convention TaskAdapter's default assumes -- override so
        # src/bench/profile.py's params/FLOPs/FPS measurement calls model(images) correctly.
        import torch

        return [torch.zeros(3, image_size[0], image_size[1], device=device)]

    def train_step(self, model, batch, device):
        images_raw, targets_raw = batch
        images = [image.to(device) for image in images_raw]
        targets = move_targets_to_device(targets_raw, device)
        return model.train_step(images, targets)  # loss lives in the model (SS4.1)

    def eval_step(self, model, batch, device):
        images_raw, targets_raw = batch
        images = [image.to(device) for image in images_raw]
        predictions = model(images)  # inference only; postprocess already applied (SS4.1)
        targets = move_targets_to_device(targets_raw, device)
        return {"loss": None, "outputs": {"preds": predictions, "targets": targets}}

    def update_metrics(self, outputs):
        preds = detach_to_cpu(outputs["outputs"]["preds"])
        targets = detach_to_cpu(outputs["outputs"]["targets"])
        for metric in self.metrics.values():
            metric.update(preds, targets)

    def compute_metrics(self):
        results = {}
        for name, metric in self.metrics.items():
            computed = metric.compute()
            if name == "map_50_95":
                results[name] = float(computed["map"])
                results["map_50"] = float(computed["map_50"])
                results["map_75"] = float(computed["map_75"])
            else:
                results[name] = float(computed)
        return results

    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    def predict_step(self, model, batch, device):
        """Postprocess (score threshold / NMS / max_det) is already applied inside model.forward
        (PLAN-P4 SS4.1/SS7); this method does not run NMS a second time.

        batch[1] carries a sample identifier: the common `predict` CLI command
        (src/cli/commands.py) collates (image, stem) pairs from input filenames through this
        adapter's own collate_fn (detection_collate), which just splits index0/index1, so this
        works unchanged for both the dataset path (image, target) and the predict path
        (image, stem).
        """
        images_raw, raw_ids = batch
        images = [image.to(device) for image in images_raw]
        predictions = model(images)

        results = []
        for i, pred in enumerate(predictions):
            stem = raw_ids[i] if isinstance(raw_ids, (list, tuple)) and isinstance(raw_ids[i], str) else None
            results.append({
                "stem": stem,
                "boxes": pred["boxes"].detach().cpu().tolist(),
                "scores": pred["scores"].detach().cpu().tolist(),
                "labels": pred["labels"].detach().cpu().tolist(),
                "class_names": [self.class_name(int(label)) for label in pred["labels"].detach().cpu().tolist()],
            })
        return results

    def class_name(self, label):
        if 0 <= label < len(self.class_names):
            return self.class_names[label]
        return str(label)

    def save_predictions(self, predictions, output_dir):
        # The common `predict` command (src/cli/commands.py) already writes
        # predictions/predict.json with {"paths": [...], "predictions": [...]}; nothing
        # further to add here for detection.
        pass

    def visualize(self, batch, predictions, output_dir, max_items):
        from src.tasks.detection.visualize import save_detection_grid

        images = batch[0]
        targets = batch[1] if len(batch) > 1 and isinstance(batch[1], list) and batch[1] and isinstance(
            batch[1][0], dict) and "boxes" in batch[1][0] else None
        save_detection_grid(images, targets, predictions, self.class_names, output_dir, max_items)
