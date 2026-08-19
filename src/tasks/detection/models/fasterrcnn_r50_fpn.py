# torchvision Faster R-CNN R50-FPN with a local COCO-pretrained checkpoint,
# repurposed for oxford_pets 3-class (background/cat/dog) detection.
import torch.nn as nn
import torchvision.models.detection as tvm_det
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.core.offline import load_local_weights
from src.core.registry import MODELS


@MODELS.register("fasterrcnn_r50_fpn_det")
class FasterRCNNR50FPNDetector(nn.Module):
    def __init__(self, num_classes=3, weights_path=None, score_thresh=0.05, nms_iou=0.5,
                 max_det=100, image_size=(512, 512), **params):
        super().__init__()
        # num_classes=91 (COCO, including background) is required at construction time so
        # the module's state_dict keys (including the box_predictor head) match the local
        # COCO checkpoint exactly, enabling strict=True loading below.
        #
        # min_size/max_size are pinned to image_size (fixed 512x512, PLAN-P4 SS3.5) so
        # GeneralizedRCNNTransform's internal resize is a no-op on the already-resized input;
        # image_mean/image_std are set to identity so it does not re-normalize input that the
        # common det_train/det_eval transform (src/tasks/detection/transform.py) already
        # normalized with ImageNet stats. Without this, Faster R-CNN alone would see a
        # different effective resolution and double-normalized pixel values than custom_fcos
        # and yolov8n, breaking the fair-comparison control (PLAN-P4 SS4.2/SS8.3).
        self.model = tvm_det.fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=91,
            box_score_thresh=score_thresh, box_nms_thresh=nms_iou,
            box_detections_per_img=max_det,
            min_size=image_size[0], max_size=max(image_size),
            image_mean=[0.0, 0.0, 0.0], image_std=[1.0, 1.0, 1.0],
        )
        # Load the local COCO checkpoint while the head is still the original
        # 91-class head, so strict=True key matching succeeds. weights_path is
        # never None-checked here: load_local_weights raises LocalAssetError
        # for a missing/None path instead of silently random-initializing.
        load_local_weights(self.model, weights_path, strict=True)

        # Replace the box predictor with a num_classes head, after the checkpoint
        # load so the replacement does not affect key matching above.
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def train_step(self, images, targets):
        self.model.train()
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())
        return {"loss": loss, "loss_dict": {k: float(v.detach()) for k, v in loss_dict.items()}}

    def forward(self, images):
        self.model.eval()
        return self.model(images)
