import torch

from src.core.adapter import TaskAdapter
from src.core.registry import ADAPTERS
from src.tasks.toy.collate import anomaly_collate, det_collate
from src.tasks.toy.postprocess import decode_det_outputs


def run_model_step(model, images, targets, loss_fn, reconstruction_target=None):
    if hasattr(model, "train_step"):
        return model.train_step(images, targets)
    outputs = model(images)
    loss_target = images if reconstruction_target == "self" else targets
    loss = loss_fn(outputs, loss_target)
    return {"loss": loss, "loss_dict": {"loss": float(loss.detach())}}


@ADAPTERS.register("toy_cls")
class ToyClsAdapter(TaskAdapter):
    def train_step(self, model, batch, device):
        images, targets = batch
        images, targets = images.to(device), targets.to(device)
        return run_model_step(model, images, targets, self.loss_fn)

    def eval_step(self, model, batch, device):
        images, targets = batch
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = self.loss_fn(outputs, targets)
        preds = torch.argmax(outputs, dim=1)
        return {"loss": loss, "outputs": {"preds": preds.cpu(), "targets": targets.cpu()}}

    def update_metrics(self, outputs):
        preds, targets = outputs["outputs"]["preds"], outputs["outputs"]["targets"]
        for metric in self.metrics.values():
            metric.update(preds, targets)

    def compute_metrics(self):
        return {name: float(metric.compute()) for name, metric in self.metrics.items()}

    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    def predict_step(self, model, batch, device):
        images, _ = batch
        images = images.to(device)
        preds = torch.argmax(model(images), dim=1)
        return [{"label": int(p)} for p in preds.tolist()]

    def batch_size(self, batch):
        return batch[0].shape[0]


@ADAPTERS.register("toy_seg")
class ToySegAdapter(TaskAdapter):
    def train_step(self, model, batch, device):
        images, targets = batch
        images, targets = images.to(device), targets.to(device)
        return run_model_step(model, images, targets, self.loss_fn)

    def eval_step(self, model, batch, device):
        images, targets = batch
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = self.loss_fn(outputs, targets)
        preds = torch.argmax(outputs, dim=1)
        return {"loss": loss, "outputs": {"preds": preds.cpu(), "targets": targets.cpu()}}

    def update_metrics(self, outputs):
        preds, targets = outputs["outputs"]["preds"], outputs["outputs"]["targets"]
        for metric in self.metrics.values():
            metric.update(preds, targets)

    def compute_metrics(self):
        return {name: float(metric.compute()) for name, metric in self.metrics.items()}

    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    def predict_step(self, model, batch, device):
        images, _ = batch
        images = images.to(device)
        preds = torch.argmax(model(images), dim=1)
        return [p.tolist() for p in preds]

    def batch_size(self, batch):
        return batch[0].shape[0]


@ADAPTERS.register("toy_det")
class ToyDetAdapter(TaskAdapter):
    def __init__(self, loss_fn, metrics, num_fg_classes=2, stride=8, image_size=(64, 64), **params):
        super().__init__(loss_fn, metrics, **params)
        self.num_fg_classes = num_fg_classes
        self.stride = stride
        self.image_size = image_size

    def collate_fn(self):
        return det_collate

    def train_step(self, model, batch, device):
        images_list, targets = batch
        images = torch.stack(images_list, dim=0).to(device)
        targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]
        outputs = model(images)
        loss = self.loss_fn(outputs, targets)
        return {"loss": loss, "loss_dict": {"loss": float(loss.detach())}}

    def eval_step(self, model, batch, device):
        images_list, targets_raw = batch
        images = torch.stack(images_list, dim=0).to(device)
        outputs = model(images)
        targets_device = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)}
                           for t in targets_raw]
        loss = self.loss_fn(outputs, targets_device)
        preds = decode_det_outputs(outputs.detach().cpu(), self.num_fg_classes, self.stride, 0.5,
                                    self.image_size)
        return {"loss": loss, "outputs": {"preds": preds, "targets": targets_raw}}

    def update_metrics(self, outputs):
        preds, targets = outputs["outputs"]["preds"], outputs["outputs"]["targets"]
        if "map" in self.metrics:
            self.metrics["map"].update(preds, targets)

    def compute_metrics(self):
        results = {}
        if "map" in self.metrics:
            computed = self.metrics["map"].compute()
            results["map"] = float(computed["map"])
        return results

    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    def predict_step(self, model, batch, device):
        images_list, _ = batch
        images = torch.stack(images_list, dim=0).to(device)
        outputs = model(images)
        preds = decode_det_outputs(outputs.detach().cpu(), self.num_fg_classes, self.stride, 0.5,
                                    self.image_size)
        return [{"boxes": p["boxes"].tolist(), "scores": p["scores"].tolist(), "labels": p["labels"].tolist()}
                for p in preds]

    def batch_size(self, batch):
        return len(batch[0])


@ADAPTERS.register("toy_anomaly")
class ToyAnomalyAdapter(TaskAdapter):
    def __init__(self, loss_fn, metrics, **params):
        super().__init__(loss_fn, metrics, **params)
        self.threshold = None

    def collate_fn(self):
        return anomaly_collate

    def train_step(self, model, batch, device):
        images, targets = batch
        images = images.to(device)
        return run_model_step(model, images, targets, self.loss_fn, reconstruction_target="self")

    def eval_step(self, model, batch, device):
        images, targets = batch
        images = images.to(device)
        reconstruction = model(images)
        error_map = (reconstruction - images).pow(2).mean(dim=1)
        image_score = error_map.flatten(1).amax(dim=1)
        labels = torch.stack([t["label"] for t in targets]).to(device)
        masks = torch.stack([t["mask"] for t in targets]).to(device)
        return {
            "loss": None,
            "outputs": {
                "image_score": image_score.detach().cpu(),
                "pixel_score": error_map.detach().cpu(),
                "label": labels.cpu(),
                "mask": masks.cpu(),
            },
        }

    def update_metrics(self, outputs):
        o = outputs["outputs"]
        if "image_auroc" in self.metrics:
            self.metrics["image_auroc"].update(o["image_score"], o["label"])
        if "pixel_auroc" in self.metrics:
            self.metrics["pixel_auroc"].update(o["pixel_score"].flatten(), o["mask"].flatten().long())

    def compute_metrics(self):
        return {name: float(metric.compute()) for name, metric in self.metrics.items()}

    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    def predict_step(self, model, batch, device):
        images, _ = batch
        images = images.to(device)
        reconstruction = model(images)
        error_map = (reconstruction - images).pow(2).mean(dim=1)
        image_score = error_map.flatten(1).amax(dim=1)
        return [{"anomaly_score": float(s)} for s in image_score.tolist()]

    def batch_size(self, batch):
        return batch[0].shape[0]

    def on_fit_end(self, model, loaders, device):
        model.eval()
        scores = []
        with torch.no_grad():
            for images, targets in loaders["valid"]:
                images = images.to(device)
                reconstruction = model(images)
                error_map = (reconstruction - images).pow(2).mean(dim=1)
                scores.append(error_map.flatten(1).amax(dim=1).cpu())
        if scores:
            all_scores = torch.cat(scores)
            self.threshold = float(all_scores.mean() + 2 * all_scores.std())
