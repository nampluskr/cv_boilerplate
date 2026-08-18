# PLAN-P4 — Object Detection

## 1. 문서 목적

이 문서는 `PLAN.md` 3.5의 P4(Object Detection)를 구현 가능한 수준으로 상세화한다. 절 번호
(`PLAN-P4 §4.1` 형식)가 적대적 검증과 `backlog.json`이 참조하는 조항 ID다.

- 범위: Pascal VOC XML 파싱, 가변 N 타깃과 `collate_fn`, mAP metric, 공통 후처리, 모델 3종, 벤치마크 config
- 비범위: 공통 계약 → `PLAN-P1`
- 상위 문서: `PLAN.md` 3.5, `PLAN-P1`, `PRD.md`

**위험도가 가장 높은 Phase다.** 적대적 검증 필수 Phase이며(`PLAN.md` 3.1), 모델 3종의 loss·postprocess
편차가 커서 모델 에이전트 간 부하가 균등하지 않다. `PLAN.md` 2.3 템플릿과의 편차는 §9.1에 정리한다.

관련 요구사항: `FR-14`, `FR-15`, `FR-18`~`FR-20`, `NFR-06`, `AC-09`, `CON-11`.

## 2. 태스크 정의

| 항목 | 값 |
|---|---|
| 입력 | RGB 이미지 |
| 출력 | 이미지당 가변 개수 N의 박스 + 클래스 + 점수 |
| target 규약 | `{"boxes": Tensor(N,4) float, "labels": Tensor(N) long}` (`PLAN-P1 §6.1`) |
| 박스 형식 | 절대 좌표 `xyxy` |
| 클래스 | `0=background`(예약), `1=cat`, `2=dog` (`CON-10`) |
| Metric | mAP@0.5:0.95 (`FR-19`) |
| 비교 모델 | Custom 1-stage / YOLOv8n / Faster R-CNN R50-FPN (`PRD.md` 2.1) |

비교 축은 **1-stage 대 2-stage**다.

### 2.1. 일반성 요구 (`CON-11`, `NFR-06`)

파이프라인(Dataset·collate·모델·Loss·Metric·Postprocess)은 **multi-class / multi-object를 일반적으로
지원한다.** `oxford_pets`가 사실상 단일 객체라는 이유로 축약하지 않는다. 구체적으로 다음을 만족해야 한다.

- 이미지당 객체 수 `N`이 `0`, `1`, `1 초과`인 샘플이 한 배치에 섞여도 Loss·Metric·Postprocess까지 완주한다.
- 클래스 수는 config `num_classes`로만 결정되며 코드에 `2`가 하드코딩되지 않는다.
- 데이터만 교체(`oxford_pets` → VOC/COCO)하면 엔진·Loss·Metric 수정 없이 동작한다.

## 3. 데이터

### 3.1. 원본 구조 실측 (2026-08-18)

| 항목 | 값 |
|---|---|
| `annotations/xmls/` | 3,686 파일 |
| XML 형식 | Pascal VOC. `size/{width,height}`, `object/{name, bndbox/{xmin,ymin,xmax,ymax}, difficult, truncated, occluded}` |
| `object/name` 값 | `cat` 1,189개 / `dog` 2,498개 (총 3,687 객체) |
| 이미지당 객체 수 | `N=1`이 3,685장, `N=2`가 1장 |
| `difficult=1` | 0건 |
| XML 파싱 오류 | 0건 |

XML은 개행 없는 단일 행이다. 정규식이나 `grep`으로 파싱하지 말고 `xml.etree.ElementTree`를 사용한다.

### 3.2. 분할이 정본 split을 쓸 수 없는 이유 (`FR-17`)

실측에서 확인된 제약이다. **`test.txt`의 3,669개 ID 중 XML을 보유한 것은 0개다.**

| 집합 | 개수 |
|---|---|
| XML 보유 ID | 3,686 |
| `list.txt`에 없는 XML ID | 15 (`Bombay_11`, `newfoundland_152` 등) |
| `trainval.txt` ∩ XML | **3,671** |
| `test.txt` ∩ XML | **0** |

따라서 `PLAN-P2 §3.2`의 정본 분할(`configs/splits/oxford_pets.json`)을 그대로 쓰면 Detection의 test
split에 박스가 하나도 없어 평가가 불가능하다. Detection은 **자체 분할**을 사용한다.

| split | 비율 | 샘플 수 |
|---|---|---|
| `train` | 0.70 | 2,569 |
| `valid` | 0.15 | 551 |
| `test` | 0.15 | 551 |

- 모집단은 `trainval.txt` ∩ XML = 3,671개다. `list.txt`에 없는 15개는 품종 라벨이 없어 층화 추출의
  기준이 없으므로 제외한다.
- 층화 기준은 `list.txt`의 `CLASS-ID`(품종 37종, ID당 90~100장)이며 seed는 `42`다.
- 결과는 `configs/splits/oxford_pets_det.json`으로 materialize하여 커밋한다(`PLAN-P1 §8.2` file 모드).
- **결과적으로 Detection의 test 이미지 일부가 Classification의 train 이미지와 겹친다.** 이는 허용된다.
  `NFR-03`과 `AC-10`이 요구하는 것은 **한 태스크 안에서** train/valid/test가 상호 배타적인 것이며,
  태스크 간 비교는 하지 않기 때문이다(`OUT-01`). 각 모델은 자기 태스크의 split만 본다.
- 이 사실을 `configs/splits/oxford_pets_det.json`의 `note` 필드와 leaderboard 각주에 명시한다.

### 3.3. subset 규격 (`PLAN-P1 §13`)

| split | 규격 | 샘플 수 |
|---|---|---|
| `train` | 품종당 8장 | 296 |
| `valid` | 품종당 2장 | 74 |
| `test` | 품종당 2장 | 74 |

추출 방식은 `PLAN-P2 §3.3`과 동일한 결정적 사전순 추출이다. 결과는
`configs/splits/oxford_pets_subset_det.json`으로 커밋한다.

### 3.4. Dataset 계약 (`FR-11`, `FR-14`, `NFR-06`)

```python
@DATASETS.register("oxford_pets_det")
class OxfordPetsDetection(Dataset):
    """Returns (image, target) where target is
    {"boxes": FloatTensor(N, 4) in absolute xyxy, "labels": LongTensor(N)}."""

    def __init__(self, root, split, transform=None, split_path=None, class_names=("cat", "dog")):
        ...
```

- `class_names`의 순서가 라벨 인덱스를 정한다. `labels = class_names.index(name) + 1`이며 `0`은 배경 예약이다.
  XML의 `name` 값이 `class_names`에 없으면 조용히 건너뛰지 않고 오류로 보고한다(`NFR-11`).
- `N = 0`인 샘플도 정상 샘플로 취급한다. 이때 `boxes`는 `shape (0, 4)` float, `labels`는 `shape (0,)` long
  텐서다. `None`이나 빈 리스트를 반환하지 않는다(`PLAN-P1 §6.1`).
- `difficult` 객체 제외 여부는 config `data.params.drop_difficult`로 제어하며 기본값은 `false`다.
  `oxford_pets`에는 `difficult=1`이 없으므로 실질 효과는 없으나, VOC 교체 시 필요하다.
- Dataset은 원본 해상도의 절대 좌표를 반환하고, 크기 변환은 transform이 담당한다.
- 새 검출 데이터셋(VOC/COCO)은 이 계약을 지키는 파서만 추가하면 되고 모델·Loss·Metric·엔진은 수정하지
  않는다(`NFR-06`).

### 3.5. Transform과 박스 정합성 (`FR-21`)

박스를 `torchvision.tv_tensors.BoundingBoxes`로 감싸 이미지와 함께 변환한다.

```python
from torchvision import tv_tensors

boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(h, w))
image, target = transform(image, target)
```

```python
@TRANSFORMS.register("det_train")
# v2.Compose([
#     v2.RandomHorizontalFlip(p=0.5),
#     v2.Resize(image_size, antialias=True),
#     v2.ToDtype(torch.float32, scale=True),
#     v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#     v2.SanitizeBoundingBoxes(min_size=1.0),
# ])

@TRANSFORMS.register("det_eval")
# v2.Compose([
#     v2.Resize(image_size, antialias=True),
#     v2.ToDtype(torch.float32, scale=True),
#     v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#     v2.SanitizeBoundingBoxes(min_size=1.0),
# ])
```

- `v2.SanitizeBoundingBoxes`가 캔버스 밖으로 나간 박스를 clamp하고 폭·높이가 1px 미만인 박스를 제거하며,
  `labels`도 함께 걸러낸다. **이 단계가 실데이터에서 `N=0`을 만들어내는 정상 경로다.**
- 따라서 `N=0` 처리는 이론적 방어가 아니라 실제로 발생하는 경로다. 파이프라인이 이를 오류 없이 통과해야 한다.
- `RandomResizedCrop`을 쓰지 않는다. 객체를 통째로 잘라내 라벨 없는 샘플을 대량 생성하며, 5 epoch
  규격에서는 학습 신호만 줄인다. augmentation은 좌우 반전으로 제한한다.
- `image_size`는 `[512, 512]`로 고정한다. 원본 종횡비를 유지하지 않는 단순 resize이며, GT와 예측에
  동일하게 적용되므로 mAP 계산에 편향을 만들지 않는다.

### 3.6. collate_fn (`FR-15`)

```python
def detection_collate(samples):
    """samples: list of (image, target). Returns (list[Tensor], list[dict])."""
    images = [s[0] for s in samples]
    targets = [s[1] for s in samples]
    return images, targets
```

- 기본 `default_collate`는 가변 길이 타깃을 스택하려다 실패하므로 반드시 이 `collate_fn`을 등록한다.
- 이미지는 transform 이후 모두 `512x512`로 같지만 **스택하지 않고 list로 유지한다.** torchvision detection
  모델의 입력 규약이 list이고, 향후 가변 해상도로 확장할 여지를 남기기 위해서다.
- `adapter.batch_size(batch)`는 `len(batch[0])`을 반환한다(`PLAN-P1 §6.1`).

## 4. 모델 인터페이스 명세

P4.1의 필수 산출물이다. 모델 3종의 내부 구조가 서로 매우 다르므로, 이 명세가 확정되지 않으면
모델 에이전트 3개가 서로 다른 시그니처로 구현한다.

### 4.1. 시그니처

Detection 모델은 **학습과 추론 경로를 명시적으로 분리한다.** `PLAN-P1 §6.3`의 `model.train_step` 경로를
사용하며, 어댑터는 분기 없이 이 규약만 호출한다.

```python
@MODELS.register("<model_name>")
class <ModelName>(nn.Module):
    def __init__(self, num_classes=3, weights_path=None,
                 score_thresh=0.05, nms_iou=0.5, max_det=100, **params):
        """num_classes includes background (index 0). oxford_pets uses 3."""

    def train_step(self, images, targets):
        """images: list[Tensor(3, H, W)], targets: list[dict{boxes, labels}].
        returns: {"loss": scalar Tensor with grad, "loss_dict": {str: float}}"""

    def forward(self, images):
        """Inference only. images: list[Tensor(3, H, W)].
        returns: list[dict{"boxes": (M,4) xyxy, "scores": (M,), "labels": (M,) long}]
        already filtered by score_thresh / nms_iou / max_det."""
```

- `train_step`은 모델 내부의 assigner·loss를 모두 포함한다. Loss는 어댑터가 계산하지 않는다. Detection의
  loss는 모델 구조와 분리할 수 없기 때문이며, 이것이 `FR-10`이 허용하는 모델 수준 분기다.
- `forward`는 **추론 전용**이다. `targets`를 받지 않는다.
- 반환 박스는 입력 이미지 좌표계(`512x512`)의 절대 `xyxy`다. 정규화 좌표나 `cxcywh`를 반환하지 않는다.
- `labels`는 `1..num_classes-1` 범위다. `0`(배경)을 반환하지 않는다.
- 검출이 없으면 `M=0`인 빈 텐서를 반환한다. `None`을 반환하지 않는다.
- ultralytics·anomalib 등 외부 형식 변환은 **모델 래퍼 안에 가둔다.** Dataset·어댑터·엔진은 외부 형식을
  알지 못한다(`PLAN.md` 3.5).

### 4.2. 후처리 파라미터 통제 (`NFR-02`)

`score_thresh`, `nms_iou`, `max_det`는 모델별 하이퍼파라미터가 아니라 **비교 통제 조건**이다.
세 모델이 서로 다른 값을 쓰면 mAP 비교가 성립하지 않는다.

- 세 모델 모두 `score_thresh=0.05`, `nms_iou=0.5`, `max_det=100`을 사용한다.
- Faster R-CNN은 torchvision 생성자의 `box_score_thresh`, `box_nms_thresh`, `box_detections_per_img`에
  이 값을 그대로 전달한다. 내부 NMS를 그대로 쓰고 이중으로 NMS를 걸지 않는다.
- YOLOv8n 래퍼는 `ultralytics.utils.ops.non_max_suppression`에 동일 값을 전달한다.
- Custom 모델은 `torchvision.ops.batched_nms`에 동일 값을 사용한다.
- 이 3개 필드는 `model` 블록 안에 있지만 **통제 필드로 승격한다**(§8.3). `model`이 자유 축이라는 이유로
  통제를 벗어나지 않도록 벤치마크 config에서 명시적으로 검사한다.

### 4.3. 모델 3종 배정 (`FR-18`)

| 에이전트 | registry key | 파일 | config | 분량 |
|---|---|---|---|---|
| 모델 에이전트 1 | `custom_fcos_det` | `src/tasks/detection/models/custom_fcos.py` | `configs/detection/custom_fcos.yaml` | 큼 |
| 모델 에이전트 2 | `fasterrcnn_r50_fpn_det` | `.../models/fasterrcnn_r50_fpn.py` | `configs/detection/fasterrcnn_r50_fpn.yaml` | 작음 |
| 모델 에이전트 3 | `yolov8n_det` | `.../models/yolov8n.py` | `configs/detection/yolov8n.yaml` | 중간 |

부하가 균등하지 않으므로 모델 에이전트 1을 가장 먼저 착수시킨다(`PLAN.md` 3.5).

#### 4.3.1. 모델 에이전트 1 — Custom 1-stage (anchor-free)

기준안은 FCOS 계열의 축소 구현이다.

| 구성 | 기준안 |
|---|---|
| backbone | from scratch. conv-BN-ReLU 스테이지 5단, 채널 `[32, 64, 128, 256, 512]` |
| neck | FPN. `C3, C4, C5` → `P3, P4, P5` (stride 8/16/32), 채널 128 통일 |
| head | 레벨 공유. classification 분기(`num_classes-1` 채널, sigmoid), regression 분기(`ltrb` 4채널), centerness 1채널 |
| assigner | center sampling. 각 레벨의 크기 범위로 객체를 배정 |
| loss | classification: focal loss / regression: GIoU loss / centerness: BCE. 가중합 |
| postprocess | 레벨별 top-k → score threshold → `batched_nms` → `max_det` |

pretrained를 사용하지 않는다. `weights_path`는 `null`이며 비교 축(custom 대 pretrained)의 전제다.

**범위 축소안** (`PLAN.md` 3.5의 사전 준비). P4.2에서 기준안이 기한 안에 완주하지 못하면 다음 순서로
축소하고 축소 사실을 `reviews/A4.md`와 leaderboard 각주에 기록한다.

1. centerness 분기를 제거한다 (loss 2종으로 축소).
2. FPN을 제거하고 단일 레벨(stride 16)만 사용한다.
3. focal loss를 BCE + 하드 네거티브 샘플링으로 대체한다.

축소해도 `N=0` 처리, 다중 클래스, `forward` 반환 규약(§4.1)은 유지한다. 이것들은 축소 대상이 아니다.

#### 4.3.2. 모델 에이전트 2 — Faster R-CNN R50-FPN

```python
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=None, weights_backbone=None, num_classes=91,
    box_score_thresh=score_thresh, box_nms_thresh=nms_iou, box_detections_per_img=max_det)
load_local_weights(model, weights_path, strict=True)          # COCO 91-class checkpoint
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)   # 3
```

- `num_classes=91`로 만든 뒤 로드하고, 그다음에 `box_predictor`를 교체한다. 순서를 바꾸면 `strict=True`가
  실패한다(`PLAN-P3 §4.2`와 동일 원칙).
- `train_step`은 `self.model.train()` 상태에서 `self.model(images, targets)`가 반환하는 loss dict를
  `{"loss", "loss_dict"}`로 변환한다. `loss = sum(loss_dict.values())`다.
- `forward`는 `self.model.eval()` 상태의 반환값을 그대로 쓴다. torchvision이 이미 `{boxes, scores, labels}`
  형식을 반환하므로 변환이 필요 없다.
- torchvision detection 모델은 **`N=0` 타깃을 이미 지원한다.** 빈 `boxes (0,4)`를 넘겨도 동작한다.

#### 4.3.3. 모델 에이전트 3 — YOLOv8n 어댑터

가장 오프라인 위반 위험이 큰 모델이다.

- `ultralytics.YOLO(...)` 고수준 API와 `model.train()` trainer를 **사용하지 않는다.** ultralytics trainer는
  데이터셋 YAML·다운로드·자체 로깅을 전제하므로 `CON-01`(pure-PyTorch)과 `NFR-07`(오프라인)에 어긋난다.
- 사용 범위는 `ultralytics.nn.tasks.DetectionModel`(모델 정의), `ultralytics.utils.loss.v8DetectionLoss`,
  `ultralytics.utils.ops.non_max_suppression`으로 한정한다.
- 로컬 `/mnt/d/backbones/yolov8n.pt`를 `torch.load`로 읽어 `nc=2`(전경 클래스 수)로 재구성한 모델에
  head를 제외한 가중치를 주입한다. `strict=False`가 되므로 **누락·초과 키 목록을 로그에 남기고**,
  backbone/neck 키가 하나라도 누락되면 오류로 승격한다(`PLAN-P1 §9.2`).
- 타깃 변환은 래퍼 안에서 수행한다. 공통 규약 `{boxes(xyxy, 절대), labels(1-based)}`을 ultralytics의
  `{batch_idx, cls(0-based), bboxes(cxcywh, 정규화)}`로 바꾸고, 추론 결과는 역방향으로 되돌린다.
  이 변환 코드가 래퍼 밖으로 새어나가면 안 된다.
- ultralytics는 임포트 시 설정 파일 생성과 버전 확인을 시도할 수 있다. `PLAN-P1 §9.1`의 오프라인 가드가
  이를 차단하며, 차단으로 임포트가 실패하면 필요한 환경 변수(`YOLO_OFFLINE`, `ULTRALYTICS_OFFLINE`)를
  가드에 추가한다. 가드를 끄는 방식으로 해결하지 않는다.

## 5. Loss

Detection의 loss는 모델에 내장된다(§4.1). 따라서 config의 `loss` 블록은 다음과 같이 선언한다.

```yaml
loss: {name: none, params: {}}
```

`@LOSSES.register("none")`은 호출 시 `RuntimeError`를 던지는 placeholder다. 어댑터가 loss를 계산하지
않는다는 사실을 config 수준에서 명시하고, 실수로 호출되면 조용히 통과하지 않게 한다.
`loss`는 통제 필드이므로 3 split이 모두 `none`이어야 한다. 한 모델만 외부 loss를 쓰면 통제 검사에서 걸린다.

## 6. Metric (`FR-19`)

| registry key | 구현 |
|---|---|
| `map_50_95` | `torchmetrics.detection.MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="faster_coco_eval")` |

- 백엔드는 `faster-coco-eval`이다(`CON-08`, 2026-08-18 사용자 승인). **2026-08-18 기준 `pytorch_env`에
  미설치 상태이므로 P1.1에서 설치하고 `requirements.txt`에 고정한다.**
- `update(preds, targets)`의 입력은 list-of-dict다. `preds`는 `{boxes, scores, labels}`,
  `targets`는 `{boxes, labels}`이며 모두 CPU 텐서로 detach해서 넘긴다.
- `compute()`는 dict를 반환한다. leaderboard에는 `map`(=mAP@0.5:0.95), `map_50`, `map_75`를 기록하고
  `train.monitor`는 `{metric: map_50_95, mode: max}`로 `map` 값을 사용한다.
- `MeanAveragePrecision`은 `N=0` 타깃과 `M=0` 예측을 모두 허용한다. P4.1에서 이를 fixture로 확인한다.
- metric 리셋 누락 시 이전 배치가 누적되어 값이 조용히 왜곡된다. 엔진이 강제하지만(`PLAN-P1 §7.2`)
  P4.1 검증 항목에 명시적으로 포함한다.

## 7. TaskAdapter와 Postprocess

```python
@ADAPTERS.register("detection")
class DetectionAdapter(TaskAdapter):
    def train_step(self, model, batch, device):
        images, targets = self.to_device(batch, device)
        return model.train_step(images, targets)       # loss lives in the model (§4.1)

    def eval_step(self, model, batch, device):
        images, targets = self.to_device(batch, device)
        predictions = model(images)
        return {"loss": None, "outputs": {"preds": predictions, "targets": targets}}

    def update_metrics(self, outputs):
        preds = [{k: v.detach().cpu() for k, v in p.items()} for p in outputs["outputs"]["preds"]]
        tgts  = [{k: v.detach().cpu() for k, v in t.items()} for t in outputs["outputs"]["targets"]]
        for metric in self.metrics.values():
            metric.update(preds, tgts)

    def batch_size(self, batch):
        return len(batch[0])

    def collate_fn(self):
        return detection_collate
```

- `to_device`는 list-of-Tensor와 list-of-dict를 각각 순회하며 옮긴다. `batch.to(device)`가 통하지 않는다.
- **`eval_step`의 `loss`는 `None`이다.** torchvision detection 모델은 `eval()` 상태에서 loss를 반환하지
  않으며, 세 모델의 loss 정의가 서로 달라 valid loss를 비교 지표로 쓸 수도 없다. 모델 선택은
  `map_50_95` 기준으로 한다(`train.monitor`). `metrics_epoch.csv`의 valid loss 열은 비어 있다.
- 후처리(NMS·score threshold)는 모델 `forward` 안에서 이미 적용된 상태다(§4.1). 어댑터가 다시 NMS를
  걸지 않는다. 이중 NMS는 모델에 따라 결과를 다르게 바꿔 비교를 깨뜨린다.
- `predict_step`은 `{"stem", "boxes", "scores", "labels", "class_names"}` 목록을 반환한다.
- 훅은 사용하지 않는다.

### 7.1. Visualization (`FR-22`)

- 예측 박스를 이미지에 그려 저장한다. 클래스별 색상은 고정(`cat = (0, 160, 255)`, `dog = (255, 160, 0)`)이며
  각 박스에 `<class> <score:.2f>`를 표기한다.
- GT 박스는 흰색 점선으로 함께 그린다. `score_thresh` 미만은 그리지 않는다.
- 검출이 0건인 이미지도 저장한다. 비어 있는 결과를 눈으로 확인할 수 있어야 한다.

## 8. Config 구성

### 8.1. 파일 목록

| 파일 | 역할 |
|---|---|
| `configs/detection/_base.yaml` | 통제 조건 |
| `configs/detection/custom_fcos.yaml` | `_base` 상속 + `model` |
| `configs/detection/fasterrcnn_r50_fpn.yaml` | `_base` 상속 + `model` |
| `configs/detection/yolov8n.yaml` | `_base` 상속 + `model` |
| `configs/benchmarks/detection_baseline.yaml` | 3 split 벤치마크 (`FR-28`) |

### 8.2. `_base.yaml` 통제 조건

```yaml
meta: {task_name: detection}
runtime: {seed: 42, device: cuda, amp: false, deterministic: warn, allow_network: false}
data:
  name: oxford_pets_det
  root: /mnt/d/datasets/oxford_pets
  params: {class_names: [cat, dog], drop_difficult: false}
  image_size: [512, 512]
  batch_size: 4
  num_workers: 4
  drop_last: false
  split: {mode: file, path: configs/splits/oxford_pets_subset_det.json}
  transform:
    train: {name: det_train, params: {}}
    eval:  {name: det_eval,  params: {}}
loss: {name: none, params: {}}
metrics:
  - {name: map_50_95, params: {box_format: xyxy}}
adapter: {name: detection, params: {}}
optim:
  optimizer: {name: sgd,    params: {lr: 0.005, momentum: 0.9, weight_decay: 0.0005}}
  scheduler: {name: cosine, params: {t_max: 5, eta_min: 0.00005}}
train: {epochs: 5, grad_clip: 10.0, monitor: {metric: map_50_95, mode: max}, log_interval: 10, save_last: true}
```

Cls/Seg와 달리 SGD를 쓴다. Faster R-CNN과 YOLO의 표준 학습 설정이 SGD 기반이고, AdamW로는 5 epoch
규격에서 세 모델 모두 mAP가 0에 가깝게 나올 위험이 크다. `grad_clip = 10.0`은 초기 학습의 loss 폭주를
막기 위한 값이며 세 split에 동일하게 적용된다.

### 8.3. 벤치마크 config와 통제 필드 확장

`PLAN-P1 §10.2`의 통제 필드에 P4 한정으로 다음 3개를 **추가**한다.

| 추가 통제 필드 | 값 |
|---|---|
| `model.params.score_thresh` | 0.05 |
| `model.params.nms_iou` | 0.5 |
| `model.params.max_det` | 100 |

`model`은 자유 축이지만 이 세 필드는 후처리 통제 조건이므로 예외적으로 검사한다(§4.2).
벤치마크 config의 `control.extra_fields`에 선언하고 `src/bench/control.py`가 통제 필드 목록에
합쳐 검사한다. 이는 계약 확장(등급 B, `PLAN.md` 5.2)이며 `PLAN-P1 §16`에 기록한다.

`control.exceptions`는 비어 있다. `batch_size = 4`에서 세 모델 모두 11.8GB에 들어가는 것을 P4.2에서 확인한다.

## 9. 하위 단계와 수용 기준

### 9.1. 템플릿 편차 (`PLAN.md` 2.3)

- P4.1이 다른 태스크보다 무겁다. VOC XML 파서, 자체 split 생성, 가변 N `collate_fn`, mAP 백엔드 설치·연결,
  N=0 fixture가 모두 여기에 들어간다.
- P4.2의 부하가 불균등하다. 모델 에이전트 1을 먼저 착수시키고 §4.3.1의 축소안을 준비해 둔다.

### 9.2. 단계별 수용 기준

| 단계 | 주체 | 산출물 | 수용 기준 |
|---|---|---|---|
| P4.1 | task 에이전트 | XML 파서, split JSON 2종, transform 2종, `detection_collate`, mAP metric, adapter, visualize, `_base.yaml`, §4 모델 인터페이스 명세, N=0 fixture | **`AC-09` fixture 통과**: `N=0`, `N=1`, `N>1`과 cat·dog가 한 배치에 섞인 입력이 collate → adapter → metric까지 오류 없이 완주. `assert_disjoint` 통과. `faster-coco-eval` 설치 확인 |
| P4.2 | 모델 에이전트 3 (동시) | 모델 3종과 config 3종 | 각 모델이 `N=0` 포함 더미 배치에서 `train_step` loss 산출과 `forward` 추론을 모두 통과. 반환 형식이 §4.1과 일치. pretrained 2종은 로컬 가중치 로드 성공 |
| P4.3 | task 에이전트 | 3모델 통합 | subset 5 epoch 학습·평가·추론 3모델 완주. 검출 0건 이미지 포함 시각화 생성 |
| P4.4 | task 에이전트 | 벤치마크 config, leaderboard | 통제 검사(확장 필드 3개 포함) 통과, leaderboard 생성. 1-stage 대 2-stage 비교 축 확인 |
| P4.5 | master | **적대적 검증(필수)**, 보고 | 미해결 Critical 없음. `reviews/A4.md` 기록 |

- 실행 시간 상한은 모델 1종당 25분이다(`PLAN-P1 §13`).
- 5 epoch·296장 규격에서 절대 mAP는 낮게 나온다. v0.1의 목적은 파이프라인 검증이며 절대 성능이 아니다
  (`PLAN.md` 9). mAP가 0에 가깝더라도 파이프라인 완주와 통제 검사 통과가 수용 기준이다.

## 10. 적대적 검증 초점

P4는 적대적 검증 **필수** Phase다(`PLAN.md` 3.1). `backlog.json`의 P4 `adversarialFocus`가 이 표를 사용한다.

| 축 | 공격 내용 | 대응 조항 |
|---|---|---|
| Detection 일반성 | `N=0`에서 파이프라인이 깨지는가. 클래스 수가 하드코딩되었는가. `oxford_pets`가 N=1이라는 전제가 코드에 스며들었는가 | §2.1, §3.4, §3.5, §3.6, §4.1 |
| target 규약 | 박스가 절대 `xyxy`인가. 라벨이 1-based이고 배경 0이 예약되었는가. 정규화 좌표나 `cxcywh`가 새어나오는가 | §2, §3.4, §4.1 |
| 외부 형식 격리 | ultralytics 형식 변환이 모델 래퍼 밖으로 나갔는가. Dataset·어댑터·엔진이 외부 형식을 아는가 | §4.1, §4.3.3, §7 |
| 공정 비교 통제 | 세 모델의 `score_thresh`/`nms_iou`/`max_det`가 동일한가. 이중 NMS가 걸리는가. optimizer·해상도·epoch이 동일한가 | §4.2, §7, §8.2, §8.3 |
| split 무결성 | Detection 자체 분할의 train/valid/test가 상호 배타적인가. 정본 분할을 못 쓰는 사유가 기록되었는가 | §3.2, §3.3 |
| 오프라인 | ultralytics 임포트·로드 경로에 네트워크 접근이 남아 있는가. 가드를 끄는 방식으로 우회하지 않았는가 | §4.3.3, `PLAN-P1 §9.1` |
| 평가 무결성 | metric 리셋이 이루어지는가. valid loss가 `None`인 것이 모델 선택을 망가뜨리지 않는가 | §6, §7 |
| 엔진 순수성 | 태스크 구현이 `core`·`bench`를 수정했는가. 통제 필드 확장이 등급 B 절차를 거쳤는가 | §8.3, `PLAN-P1 §7.4` |

## 11. 조항 개정 이력

| 일자 | 조항 | 등급 | 변경 내용 | 요청자 | 승인 |
|---|---|---|---|---|---|
| 2026-08-18 | 전체 | — | 최초 작성 | master | — |

---

*작성일: 2026-08-18 · 버전: v0.1 · 상위 문서: PLAN.md · 다음 단계: plans/PLAN-P5-anomaly.md*
