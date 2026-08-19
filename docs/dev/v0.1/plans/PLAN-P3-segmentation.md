# PLAN-P3 — Segmentation

## 1. 문서 목적

이 문서는 `PLAN.md` 3.4의 P3(Segmentation)를 구현 가능한 수준으로 상세화한다. 절 번호(`PLAN-P3 §4.1` 형식)가
적대적 검증과 `backlog.json`이 참조하는 조항 ID다.

- 범위: `oxford_pets` trimap 3-class 분할 파이프라인, 모델 3종, 벤치마크 config, 하위 단계 수용 기준
- 비범위: 공통 계약 → `PLAN-P1`. 분할 정본 생성 → `PLAN-P2 §3.2`
- 상위 문서: `PLAN.md` 3.4, `PLAN-P1`, `PLAN-P2`(정본 split), `PRD.md`

`PLAN.md` 2.3 템플릿을 따르며, 편차는 P3.1에 **이미지와 마스크의 동시 변환**이 추가되는 것뿐이다.

관련 요구사항: `FR-11`, `FR-13`, `FR-18`~`FR-22`, `AC-05`, `CON-10`.

## 2. 태스크 정의

| 항목 | 값 |
|---|---|
| 입력 | RGB 이미지 |
| 출력 | 픽셀별 클래스 인덱스 (3-class) |
| target 규약 | `Tensor(H, W) long`, 값 범위 `0..2` (`PLAN-P1 §6.1`) |
| 데이터셋 | `/mnt/d/datasets/oxford_pets/annotations/trimaps` |
| Loss | Cross Entropy (픽셀 단위) |
| Metric | mIoU, Dice (`FR-19`) |
| 비교 모델 | Custom U-Net류 / DeepLabV3-ResNet50 / FCN-ResNet50 (`PRD.md` 2.1) |

비교 축은 **head 구조(ASPP 유무) 효과**다. 따라서 torchvision 2종은 backbone을 ResNet50으로 통일한다
(`BRIEF.md` 4). backbone이 다르면 head 비교가 성립하지 않는다.

## 3. 데이터

### 3.1. 원본 구조와 라벨 매핑

| 항목 | 값 |
|---|---|
| `annotations/trimaps/` | 7,390 파일 (`.png`) |
| 픽셀 값 | `{1, 2, 3}` (`PLAN.md` 6.4 실측) |
| 파일명 | 이미지와 1:1 대응 (`<stem>.png`) |

Oxford-IIIT Pet trimap의 원본 의미와 이 프로젝트의 클래스 인덱스 매핑을 다음으로 확정한다.

| trimap 원본 값 | 의미 | 클래스 인덱스 | 클래스 이름 |
|---|---|---|---|
| 2 | Background | 0 | `background` |
| 1 | Foreground (pet) | 1 | `pet` |
| 3 | Not classified (경계) | 2 | `boundary` |

- 매핑은 lookup 테이블로 수행하고, 위 3개 외의 값이 나오면 조용히 무시하지 않고 오류로 보고한다(`NFR-11`).
- 경계(3)를 `ignore_index`로 버리지 않고 **독립 클래스로 학습·평가한다**. `CON-10`이 "trimap 3-class"로
  확정했기 때문이며, 세 모델이 동일 기준으로 평가되므로 비교의 공정성에도 문제가 없다.
- `PIL.Image.open(...)`으로 읽고 `numpy` 정수 배열을 거쳐 `torch.long` 텐서로 만든다. 마스크에
  보간이나 정규화를 적용하지 않는다.

### 3.2. split (`FR-17`, `AC-10`)

`PLAN-P2 §3.2`에서 생성한 **정본 분할 `configs/splits/oxford_pets.json`을 그대로 사용한다.**
7,349개 ID 전부가 trimap을 보유하므로 필터링이 필요 없다.

| split | 샘플 수 |
|---|---|
| `train` | 3,128 |
| `valid` | 552 |
| `test` | 3,669 |

같은 이미지가 Classification과 Segmentation에서 서로 다른 split에 속하는 일이 없다. 태스크 간 비교는
하지 않지만(`OUT-01`), 분할을 공유하면 향후 동일 이미지에 대한 태스크별 산출물을 대조하기 쉽다.

### 3.3. subset 규격 (`PLAN-P1 §13`)

| split | 규격 | 샘플 수 |
|---|---|---|
| `train` | 품종당 8장 | 296 |
| `valid` | 품종당 2장 | 74 |
| `test` | 품종당 2장 | 74 |

- 추출 방식은 `PLAN-P2 §3.3`과 동일하다. 정본 split 안에서 ID 사전순 정렬 후 품종별로 앞에서부터 취한다.
  무작위성이 없으므로 항상 동일하다.
- 결과는 `configs/splits/oxford_pets_subset_seg.json`으로 커밋한다.
- Classification subset(품종당 10/2/2)과 train 수만 다르다. 해상도가 224에서 256으로 커지고 마스크
  디코딩 비용이 붙어 15분 상한(`PLAN-P1 §13`)에 맞추기 위한 조정이다.

### 3.4. Dataset 계약 (`FR-11`, `FR-13`, `NFR-06`)

```python
@DATASETS.register("oxford_pets_seg")
class OxfordPetsSegmentation(Dataset):
    """Returns (image, mask) where mask is a LongTensor of shape (H, W) with values in [0, 2]."""

    def __init__(self, root, split, transform=None, split_path=None):
        ...

    def __getitem__(self, index):
        # returns (Tensor(3, H, W) float, Tensor(H, W) long)
```

- 이미지 경로는 `os.path.join(root, "images", stem + ".jpg")`, 마스크 경로는
  `os.path.join(root, "annotations", "trimaps", stem + ".png")`다.
- 이미지는 `.convert("RGB")`를 거친다(`PLAN-P2 §3.1`의 인코딩 문제 동일 적용).
- `self.classes = ["background", "pet", "boundary"]`를 노출한다.
- 새 분할 데이터셋을 붙일 때 이 계약만 지키면 모델·Loss·Metric·엔진을 수정하지 않는다(`NFR-06`).

### 3.5. Transform — 이미지·마스크 동시 변환 (`FR-21`)

P3.1의 유일한 템플릿 편차다. `torchvision.transforms.v2`가 이미지와 마스크를 함께 변환하도록
마스크를 `torchvision.tv_tensors.Mask`로 감싼다.

```python
from torchvision import tv_tensors

image = tv_tensors.Image(image)
mask = tv_tensors.Mask(mask)          # v2 applies NEAREST interpolation automatically
image, mask = transform(image, mask)
```

```python
@TRANSFORMS.register("seg_train")
# v2.Compose([
#     v2.RandomResizedCrop(image_size, scale=(0.7, 1.0), antialias=True),
#     v2.RandomHorizontalFlip(p=0.5),
#     v2.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64}, scale=True),
#     v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
# ])

@TRANSFORMS.register("seg_eval")
# v2.Compose([
#     v2.Resize(image_size, antialias=True),
#     v2.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64}, scale=True),
#     v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
# ])
```

필수 확인 사항이다. 여기서 실수하면 라벨이 조용히 망가지고 metric만 낮게 나온다.

- 마스크에 **bilinear 보간이 적용되면 안 된다.** `tv_tensors.Mask`로 감싸야 v2가 NEAREST를 선택한다.
- 마스크에 **`scale=True`가 적용되면 안 된다.** `ToDtype`에 dict를 넘겨 `Mask`는 `int64`로만 캐스팅하고
  255로 나누지 않는다.
- 마스크에 **Normalize가 적용되면 안 된다.** `v2.Normalize`는 `Image`에만 작용한다.
- 변환 후 마스크 값 집합이 `{0, 1, 2}`의 부분집합인지 P3.1 검증에서 확인한다.
- `eval`은 `Resize(image_size)`만 사용하고 CenterCrop을 쓰지 않는다. 잘라내면 GT의 일부가 평가에서
  사라져 mIoU가 왜곡된다.
- `image_size`는 `[256, 256]`으로 고정한다.

## 4. 모델 인터페이스 명세

P3.1의 필수 산출물이다. 모델 에이전트 3개가 이 명세만 보고 동시에 구현한다.

### 4.1. 시그니처

```python
@MODELS.register("<model_name>")
class <ModelName>(nn.Module):
    def __init__(self, num_classes=3, backbone_name=None, weights_path=None, **params):
        ...

    def forward(self, images):
        """images: Tensor(B, 3, H, W) float, normalized.
        returns: Tensor(B, num_classes, H, W) raw logits at input resolution."""
```

- 반환은 **입력 해상도와 동일한 크기의 logits 텐서**다. `OrderedDict`나 tuple을 반환하지 않는다.
- torchvision segmentation 모델은 `{"out": Tensor, "aux": Tensor}` 형태를 반환하므로, **모델 래퍼가
  `["out"]`을 꺼내 텐서로 정규화한다.** 이 변환을 어댑터나 엔진에 두지 않는다.
- 출력 해상도가 입력보다 작으면 래퍼 안에서 `F.interpolate(..., mode="bilinear", align_corners=False)`로
  올린다. Loss와 Metric은 항상 입력 해상도에서 계산된다.
- `model.train_step`을 정의하지 않는다. 표준 경로를 쓴다.

### 4.2. aux_classifier 처리 (`NFR-02`)

DeepLabV3와 FCN의 COCO 체크포인트는 `aux_classifier` 가중치를 포함한다. 다음 순서를 지킨다.

1. `torchvision.models.segmentation.<arch>(weights=None, weights_backbone=None, num_classes=21, aux_loss=True)`
   로 아키텍처를 만든다. `aux_loss=True`여야 체크포인트 키와 맞아 `strict=True` 로드가 성립한다.
2. `load_local_weights(model, weights_path, strict=True)`로 COCO 가중치를 주입한다.
3. `model.classifier`의 마지막 `Conv2d`를 `out_channels=3`으로 교체한다.
4. **`model.aux_classifier = None`으로 제거한다.**

4번이 필수다. aux head를 남기면 모델별로 loss 구성이 달라져 `NFR-02`(공정 비교 통제)를 위반한다.
세 모델 모두 단일 출력·단일 loss로 통일한다. Custom U-Net도 deep supervision을 사용하지 않는다.

`weights_path`가 없으면 `LocalAssetError`로 실패하며 무작위 초기화로 폴백하지 않는다(`PLAN-P1 §9.2`).

### 4.3. 모델 3종 배정 (`FR-18`)

| 에이전트 | registry key | 파일 | config | 내용 |
|---|---|---|---|---|
| 모델 에이전트 1 | `custom_unet_seg` | `src/tasks/segmentation/models/custom_unet.py` | `configs/segmentation/custom_unet.yaml` | from scratch U-Net. 공통 backbone(§4.3.1)을 encoder로, 대칭 decoder + skip connection. pretrained 없음 |
| 모델 에이전트 2 | `deeplabv3_resnet50_seg` | `.../models/deeplabv3_resnet50.py` | `configs/segmentation/deeplabv3_resnet50.yaml` | 로컬 `deeplabv3_resnet50_coco-cd0a2569.pth`, ASPP head |
| 모델 에이전트 3 | `fcn_resnet50_seg` | `.../models/fcn_resnet50.py` | `configs/segmentation/fcn_resnet50.yaml` | 로컬 `fcn_resnet50_coco-1167a1af.pth`, FCN head |

각 에이전트는 위 2개 파일만 생성·수정한다. Custom U-Net의 파라미터 규모는 10M 이하를 목표로 한다.

#### 4.3.1. Custom U-Net encoder/decoder

encoder는 `PLAN.md §3.1.1`에 정의한 공통 backbone을 그대로 쓴다(`custom_cnn_cls`, `custom_fcos_det`과
공유하는 `ConvBlock = Conv2d(bias=False) → BatchNorm2d → ReLU(inplace=True)` 1종, stage 5단, 채널
`3→32→64→128→256→512`(C1~C5), 누적 stride 2/4/8/16/32).

decoder는 encoder를 대칭으로 뒤집는다.

| 구성 | 기준안 |
|---|---|
| 기본 블록 | `DeconvBlock = interpolate(nearest, scale_factor=2) → ConvBlock` |
| 업샘플 단수 | 5단. `C5(stride32)`부터 시작해 `C4, C3, C2, C1`과 순서대로 concat skip 후 업샘플, 마지막 1단은 skip 없이 `stride2 → stride1`로 복귀 |
| 채널 진행(업샘플 방향) | `512 → 256 → 128 → 64 → 32 → 32` |
| head | `Conv2d(32, num_classes, kernel_size=1)` — 입력 해상도에서 raw logits (`§4.1` 규약) |

`aux` 분기를 두지 않는다. torchvision 2종의 `aux_classifier`를 제거해 단일 출력으로 맞추는 것(§4.2)과
동일한 이유로, custom U-Net도 deep supervision 없이 단일 출력만 낸다.

## 5. Loss와 Metric

### 5.1. Loss

```python
@LOSSES.register("seg_cross_entropy")
# nn.CrossEntropyLoss(weight=params.get("class_weight"), ignore_index=params.get("ignore_index", -100))
```

- 입력은 `logits (B, C, H, W)`와 `target (B, H, W)`다. `nn.CrossEntropyLoss`가 이 형태를 그대로 받는다.
- 기본값은 `class_weight=None`, `ignore_index=-100`(비활성)이다. 경계 클래스를 버리지 않는다(§3.1).
- `loss`는 통제 필드이므로 3 split이 동일 설정을 쓴다.

### 5.2. Metric (`FR-19`)

| registry key | 구현 | 비고 |
|---|---|---|
| `miou` | `torchmetrics.classification.MulticlassJaccardIndex(num_classes=3, average="macro")` | 클래스별 IoU의 평균 |
| `dice` | `torchmetrics.classification.MulticlassF1Score(num_classes=3, average="macro", multidim_average="global")` | Dice 계수는 클래스별 F1과 동일하다 |

- metric에 넘기는 예측은 `logits.argmax(dim=1)`, 형태는 `(B, H, W)`다. 확률을 넘기지 않는다.
- `multidim_average="global"`이어야 배치 내 모든 픽셀이 하나의 혼동행렬로 누적된다. 이미지별 평균을
  내면 작은 객체가 과대평가된다.
- `train.monitor`는 `{metric: miou, mode: max}`다.

## 6. TaskAdapter

```python
@ADAPTERS.register("segmentation")
class SegmentationAdapter(TaskAdapter):
    def train_step(self, model, batch, device):
        images, masks = batch[0].to(device), batch[1].to(device)
        logits = model(images)
        loss = self.loss_fn(logits, masks)
        return {"loss": loss, "loss_dict": {"loss": float(loss.detach())}}

    def eval_step(self, model, batch, device):
        images, masks = batch[0].to(device), batch[1].to(device)
        logits = model(images)
        loss = self.loss_fn(logits, masks)
        return {"loss": loss, "outputs": {"preds": logits.argmax(dim=1), "targets": masks}}

    def update_metrics(self, outputs):
        for metric in self.metrics.values():
            metric.update(outputs["outputs"]["preds"], outputs["outputs"]["targets"])

    def batch_size(self, batch):
        return batch[0].shape[0]

    def collate_fn(self):
        return None      # dense targets have fixed shape after transform
```

- `collate_fn`은 `None`이다. transform 이후 모든 마스크가 `(256, 256)`으로 같으므로 기본 collate로 충분하다.
- 훅은 사용하지 않는다.
- `predict_step`은 `{"stem", "mask_path", "class_pixel_ratio"}` 목록을 반환하고, 예측 마스크는
  PNG(값 `0..2`)로 `predictions/`에 저장한다.

## 7. Postprocess와 Visualization

### 7.1. Postprocess (`FR-20`)

`argmax(dim=1)` 외의 후처리를 두지 않는다. CRF나 morphological 보정은 v0.1 비범위이며, 도입하면
모델 간 비교 축이 흐려진다.

### 7.2. Visualization (`FR-22`)

- 원본 이미지 / GT 마스크 / 예측 마스크를 가로로 이어붙인 비교 이미지를 저장한다.
- 색상 팔레트는 고정한다. `background = (0, 0, 0)`, `pet = (0, 200, 0)`, `boundary = (200, 0, 0)`.
- 최대 장수는 `output.max_visualizations`(기본 16)를 따른다.

## 8. Config 구성

### 8.1. 파일 목록

| 파일 | 역할 |
|---|---|
| `configs/segmentation/_base.yaml` | 통제 조건 |
| `configs/segmentation/custom_unet.yaml` | `_base` 상속 + `model` |
| `configs/segmentation/deeplabv3_resnet50.yaml` | `_base` 상속 + `model` |
| `configs/segmentation/fcn_resnet50.yaml` | `_base` 상속 + `model` |
| `configs/benchmarks/segmentation_baseline.yaml` | 3 split 벤치마크 (`FR-28`) |

### 8.2. `_base.yaml` 통제 조건

```yaml
meta: {task_name: segmentation}
runtime: {seed: 42, device: cuda, amp: false, deterministic: warn, allow_network: false}
data:
  name: oxford_pets_seg
  root: /mnt/d/datasets/oxford_pets
  image_size: [256, 256]
  batch_size: 8
  num_workers: 4
  drop_last: false
  split: {mode: file, path: configs/splits/oxford_pets_subset_seg.json}
  transform:
    train: {name: seg_train, params: {}}
    eval:  {name: seg_eval,  params: {}}
loss: {name: seg_cross_entropy, params: {ignore_index: -100}}
metrics:
  - {name: miou, params: {num_classes: 3}}
  - {name: dice, params: {num_classes: 3}}
adapter: {name: segmentation, params: {}}
optim:
  optimizer: {name: adamw,  params: {lr: 0.0003, weight_decay: 0.0001}}
  scheduler: {name: cosine, params: {t_max: 5, eta_min: 0.000001}}
train: {epochs: 5, grad_clip: null, monitor: {metric: miou, mode: max}, log_interval: 10, save_last: true}
```

`batch_size = 8`은 256x256 해상도에서 ResNet50 기반 2종이 11.8GB에 들어가도록 정한 값이다.

### 8.3. 벤치마크 config

- 자유 축은 `model`뿐이다.
- `control.exceptions`는 비어 있다. OOM이 발생하면 예외를 추가하지 않고 `_base.yaml`의 `batch_size`를
  세 split 공통으로 낮춘다(`PLAN-P2 §8.3`과 동일 원칙).
- `deterministic: warn`이 필요하다. segmentation의 `interpolate` backward는 CUDA 결정 커널이 없어
  `strict`에서는 실행되지 않는다(`PLAN-P1 §5.1`).

## 9. 하위 단계와 수용 기준

| 단계 | 주체 | 산출물 | 수용 기준 |
|---|---|---|---|
| P3.1 | task 에이전트 | Dataset, subset split JSON, transform 2종, loss, metric 2종, adapter, visualize, `_base.yaml`, §4 모델 인터페이스 명세 | 더미 배치가 Dataset → collate → adapter → metric까지 통과. **변환 후 마스크 값이 `{0,1,2}` 부분집합이고 dtype이 `int64`임을 검증** (§3.5). `assert_disjoint` 통과 |
| P3.2 | 모델 에이전트 3 (동시) | 모델 3종과 config 3종 | 각 모델이 `(2, 3, 256, 256)` 입력에서 `(2, 3, 256, 256)` logits 반환. pretrained 2종은 로컬 가중치 `strict=True` 로드 성공 후 head 교체, `aux_classifier is None` |
| P3.3 | task 에이전트 | 3모델 통합 | subset 5 epoch 학습·평가·추론 3모델 완주. 예측 마스크 PNG와 비교 이미지 생성 |
| P3.4 | task 에이전트 | 벤치마크 config, leaderboard | 통제 검사 통과, leaderboard 생성. ASPP 유무 비교 축이 표에서 확인됨 |
| P3.5 | master | 적대적 검증, 보고 | 미해결 Critical 없음. `reviews/A3.md` 기록 |

실행 시간 상한은 모델 1종당 15분이다(`PLAN-P1 §13`).

## 10. 적대적 검증 초점

P3는 적대적 검증 선택 Phase다(`PLAN.md` 3.1). 수행 시 공격 축은 다음과 같다.

| 축 | 공격 내용 | 대응 조항 |
|---|---|---|
| 라벨 무결성 | trimap `{1,2,3}` → `{0,1,2}` 매핑이 정확한가. 마스크에 보간·정규화·`scale=True`가 적용되지 않는가 | §3.1, §3.5 |
| target 규약 | Dataset이 `(H, W) long`을 반환하는가. one-hot이나 float를 흘리지 않는가 | §3.4, `PLAN-P1 §6.1` |
| 모델 출력 정규화 | torchvision의 `OrderedDict`가 어댑터나 엔진으로 새어나가지 않는가. 출력 해상도가 입력과 같은가 | §4.1 |
| 공정 비교 통제 | aux head가 한 모델에만 남아 loss 구성이 다르지 않은가. backbone이 ResNet50으로 통일되었는가 | §4.2, §8.2 |
| metric 정확성 | `multidim_average`가 `global`인가. argmax 결과를 넘기는가 | §5.2 |
| 평가 무결성 | eval transform이 GT를 잘라내지 않는가. metric 리셋이 이루어지는가 | §3.5, §6 |
| 엔진 순수성 | 태스크 구현이 `core`·`bench`를 수정하지 않았는가 | §9, `PLAN-P1 §7.4` |

## 11. 조항 개정 이력

| 일자 | 조항 | 등급 | 변경 내용 | 요청자 | 승인 |
|---|---|---|---|---|---|
| 2026-08-18 | 전체 | — | 최초 작성 | master | — |

---

*작성일: 2026-08-18 · 버전: v0.1 · 상위 문서: PLAN.md · 다음 단계: plans/PLAN-P4-detection.md*
