# PLAN-P2 — Classification

## 1. 문서 목적

이 문서는 `PLAN.md` 3.3의 P2(Classification)를 구현 가능한 수준으로 상세화한다. 절 번호(`PLAN-P2 §4.1` 형식)가
적대적 검증과 `backlog.json`이 참조하는 조항 ID다.

- 범위: `oxford_pets` 37 breeds 분류 파이프라인, 모델 3종, 벤치마크 config, 하위 단계 수용 기준
- 비범위: 공통 계약(엔진·Config·Registry·벤치마크·CLI) → `PLAN-P1`
- 상위 문서: `PLAN.md` 3.3, `PLAN-P1`(공통 계약), `PRD.md`

P2는 4개 태스크 중 가장 가볍고, `PLAN.md` 2.3 하위 단계 템플릿의 실효성을 처음 검증하는 Phase다.
템플릿과의 편차는 없다.

관련 요구사항: `FR-11`, `FR-12`, `FR-17`~`FR-22`, `FR-30`~`FR-32`, `NFR-05`, `AC-05`, `CON-10`.

## 2. 태스크 정의

| 항목 | 값 |
|---|---|
| 입력 | RGB 이미지 |
| 출력 | 37개 품종 중 1개 (다중 클래스 단일 라벨) |
| target 규약 | `Tensor(scalar, long)`, 값 범위 `0..36` (`PLAN-P1 §6.1`) |
| 데이터셋 | `/mnt/d/datasets/oxford_pets` |
| Loss | Cross Entropy |
| Metric | Top-1 Accuracy, Macro F1 (`FR-19`) |
| 비교 모델 | Custom CNN / ResNet50 / EfficientNet-B0 (`PRD.md` 2.1) |

## 3. 데이터

### 3.1. 원본 구조 실측 (2026-08-18)

| 항목 | 값 |
|---|---|
| `images/` | 7,390 파일 |
| `annotations/list.txt` | 주석 4행 + 항목 7,349행 |
| `list.txt` 형식 | `Image CLASS-ID SPECIES BREED-ID` (공백 구분) |
| `CLASS-ID` | `1..37` (37 품종) |
| `SPECIES` | `1=Cat` 2,371건 / `2=Dog` 4,978건 |
| 품종당 샘플 수 | 최소 184, 최대 200 |
| `annotations/trainval.txt` | 3,680행 |
| `annotations/test.txt` | 3,669행 |

`3,680 + 3,669 = 7,349`로 `list.txt`와 일치한다. 이미지 디렉토리(7,390)와 41장 차이가 나므로
**샘플 인덱스의 기준은 `list.txt`이며 `images/` 디렉토리 스캔을 사용하지 않는다**(`PLAN.md` 6.4).

구현 시 유의사항:

- `CLASS-ID`는 1-based다. Dataset은 `label = class_id - 1`로 0-based 변환해 반환한다(`CON-10`).
- `oxford_pets`의 일부 `.jpg`는 실제 인코딩이 PNG이거나 CMYK다. `PIL.Image.open(...).convert("RGB")`를
  반드시 거친다. 변환 실패 파일이 발견되면 조용히 건너뛰지 않고 오류로 보고한다(`NFR-11`).
- `#`로 시작하는 주석 행을 건너뛴다.

### 3.2. split 확정 (`FR-17`, `AC-10`)

공식 분할을 기준으로 삼는다. `test.txt`를 그대로 test로 쓰고, `trainval.txt`를 train/valid로 나눈다.

| split | 출처 | 샘플 수 |
|---|---|---|
| `train` | `trainval.txt`의 85% | 3,128 |
| `valid` | `trainval.txt`의 15% | 552 |
| `test` | `test.txt` 전량 | 3,669 |

- `trainval.txt` 분할은 `CLASS-ID` 층화 추출이며 seed는 `42`로 고정한다.
- 결과는 `configs/splits/oxford_pets.json`으로 **materialize하여 커밋한다**(`PLAN-P1 §8.2` file 모드).
  비율 재계산에 의존하지 않으므로 라이브러리 버전이 바뀌어도 분할이 변하지 않는다. 이 파일은 Cls와 Seg가 공유하는 **정본 분할**이다. Detection은 `test.txt`에 XML 주석이 하나도 없어 이 분할을 사용할 수 없다. 사유와 대체 분할은 `PLAN-P4 §3.2`에 있다.
- 샘플 ID는 `list.txt`의 `Image` 컬럼(확장자 없는 stem)이다. Seg·Det과 동일한 ID 체계이므로
  P3·P4가 같은 split 파일 생성 로직을 재사용한다.
- `PLAN-P1 §8.3`의 `assert_disjoint`가 Dataset 생성 시마다 검증한다.

### 3.3. subset 규격 (`NFR-12`, `PLAN-P1 §13`)

스모크와 leaderboard는 축소 subset으로 수행한다(`PLAN.md` 4).

| split | 규격 | 샘플 수 |
|---|---|---|
| `train` | 품종당 10장 | 370 |
| `valid` | 품종당 2장 | 74 |
| `test` | 품종당 2장 | 74 |

- 각 split의 전체 목록에서 ID 사전순 정렬 후 앞에서부터 품종별로 필요한 수만큼 취한다. 무작위 추출을
  쓰지 않으므로 seed와 무관하게 항상 동일하다.
- 결과는 `configs/splits/oxford_pets_subset_cls.json`으로 커밋한다.
- 원본 split의 부분집합이므로 상호 배타성이 자동으로 유지된다. 그래도 `assert_disjoint`를 우회하지 않는다.

### 3.4. Dataset 계약 (`FR-11`, `FR-12`, `NFR-06`)

```python
@DATASETS.register("oxford_pets_cls")
class OxfordPetsClassification(Dataset):
    """Returns (image, label) where label is a 0-based breed index in [0, 36]."""

    def __init__(self, root, split, transform=None, split_path=None):
        ...

    def __getitem__(self, index):
        # returns (Tensor(3, H, W) float, Tensor(scalar) long)
```

- `root`는 `/mnt/d/datasets/oxford_pets`, 이미지 경로는 `os.path.join(root, "images", stem + ".jpg")`다.
- `transform`은 `(image, target)`을 함께 받아 함께 반환한다. Classification은 target을 변형하지 않지만
  시그니처는 4태스크 공통으로 유지한다.
- 클래스 이름 목록(`self.classes`, 길이 37)을 노출한다. 시각화와 예측 저장에 사용한다.
- 데이터 다운로드를 하지 않는다(`CON-03`). `root` 또는 `list.txt`가 없으면 `LocalAssetError`로 실패한다.
- 새 분류 데이터셋을 추가할 때는 이 Dataset 계약만 지키면 되고 모델·Loss·Metric·엔진은 수정하지 않는다(`NFR-06`).

### 3.5. Transform (`FR-21`, `NFR-02`)

`torchvision.transforms.v2`를 사용한다. 통제 필드이므로 3개 모델이 동일한 값을 쓴다(`PLAN-P1 §10.2`).

```python
@TRANSFORMS.register("cls_train")
# v2.Compose([
#     v2.RandomResizedCrop(image_size, scale=(0.7, 1.0), antialias=True),
#     v2.RandomHorizontalFlip(p=0.5),
#     v2.ToImage(),
#     v2.ToDtype(torch.float32, scale=True),
#     v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
# ])

@TRANSFORMS.register("cls_eval")
# v2.Compose([
#     v2.Resize(int(image_size[0] * 256 / 224), antialias=True),
#     v2.CenterCrop(image_size),
#     v2.ToImage(),
#     v2.ToDtype(torch.float32, scale=True),
#     v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
# ])
```

- `IMAGENET_MEAN = [0.485, 0.456, 0.406]`, `IMAGENET_STD = [0.229, 0.224, 0.225]`.
- Custom CNN도 동일한 정규화를 쓴다. 정규화 통계가 다르면 통제 조건 위반이다(`NFR-02`).
- `image_size`는 `[224, 224]`로 고정한다.

## 4. 모델 인터페이스 명세

`PLAN.md` 2.3에 따라 P2.1의 필수 산출물이다. 모델 에이전트 3개가 이 명세만 보고 동시에 구현한다.

### 4.1. 시그니처

```python
@MODELS.register("<model_name>")
class <ModelName>(nn.Module):
    def __init__(self, num_classes=37, backbone_name=None, weights_path=None, **params):
        ...

    def forward(self, images):
        """images: Tensor(B, 3, H, W) float, normalized.
        returns: Tensor(B, num_classes) raw logits."""
```

- 반환은 **logits**다. `softmax`나 `log_softmax`를 적용하지 않는다. 확률 변환은 후처리(§7)의 책임이다.
- `model.train_step`을 정의하지 않는다. Classification은 표준 경로를 쓰며 loss는 어댑터가 계산한다
  (`PLAN-P1 §6.3`).
- 모델은 metric·시각화·체크포인트를 알지 못한다.
- 모델 파일은 `src.core.registry`, `src.core.offline`, `torch`, `torchvision`만 임포트한다.
  다른 태스크 패키지나 `bench`를 임포트하지 않는다(`PLAN-P1 §2.2`).

### 4.2. 가중치 로드 규약 (`CON-04`, `PLAN-P1 §9.2`)

pretrained 모델은 다음 순서를 지킨다. 순서를 바꾸면 head 교체 후 키가 맞지 않는다.

1. `torchvision.models.<arch>(weights=None)`로 아키텍처를 만든다.
2. `load_local_weights(backbone, weights_path, strict=True)`로 ImageNet 가중치를 주입한다.
   이 시점에는 아직 원본 1000-class head가 붙어 있으므로 `strict=True`가 성립한다.
3. 마지막 분류 head를 `num_classes=37`로 교체한다.

- `weights_path`가 `None`이거나 파일이 없으면 `LocalAssetError`로 실패한다. 무작위 초기화로 폴백하지 않는다.
- 인터넷 접근이 발생하면 `PLAN-P1 §9.1`의 오프라인 가드가 즉시 실패시킨다.

### 4.3. 모델 3종 배정 (`FR-18`, `PRD.md` 2.1)

| 에이전트 | registry key | 파일 | config | 내용 |
|---|---|---|---|---|
| 모델 에이전트 1 | `custom_cnn_cls` | `src/tasks/classification/models/custom_cnn.py` | `configs/classification/custom_cnn.yaml` | from scratch CNN. conv-BN-ReLU 블록 5단 + GAP + Linear. pretrained 없음 |
| 모델 에이전트 2 | `resnet50_cls` | `.../models/resnet50.py` | `configs/classification/resnet50.yaml` | `torchvision.models.resnet50`, 로컬 `resnet50-0676ba61.pth`, `fc` 교체 |
| 모델 에이전트 3 | `efficientnet_b0_cls` | `.../models/efficientnet_b0.py` | `configs/classification/efficientnet_b0.yaml` | `torchvision.models.efficientnet_b0`, 로컬 `efficientnet_b0_rwightman-7f5810bc.pth`, `classifier[1]` 교체 |

각 에이전트는 위 2개 파일만 생성·수정한다. 다른 파일을 수정해야 하는 상황이 생기면 직접 고치지 않고
변경 요청을 반환한다(`PLAN.md` 5.1).

Custom CNN의 파라미터 규모는 5M 이하를 목표로 한다. 비교 축이 "정확도 대 파라미터/효율"이므로
(`PRD.md` 2.1) pretrained 대비 경량이어야 비교가 의미를 갖는다.

## 5. Loss와 Metric

### 5.1. Loss

```python
@LOSSES.register("cross_entropy")
# nn.CrossEntropyLoss(label_smoothing=params.get("label_smoothing", 0.0))
```

`label_smoothing` 기본값은 `0.0`이다. `loss`는 통제 필드이므로 3개 split이 동일 설정을 쓴다.

### 5.2. Metric (`FR-19`)

| registry key | 구현 | 비고 |
|---|---|---|
| `top1_accuracy` | `torchmetrics.classification.MulticlassAccuracy(num_classes=37, average="micro")` | Top-1 |
| `macro_f1` | `torchmetrics.classification.MulticlassF1Score(num_classes=37, average="macro")` | 품종 불균형(184~200) 보정 |

- metric 객체는 어댑터가 소유하며 device로 이동시킨다(`PLAN-P1 §6.4`).
- `compute_metrics()`는 `{"top1_accuracy": float, "macro_f1": float}`를 반환한다.
- `train.monitor`는 `{metric: top1_accuracy, mode: max}`다.

## 6. TaskAdapter

```python
@ADAPTERS.register("classification")
class ClassificationAdapter(TaskAdapter):
    def train_step(self, model, batch, device):
        images, targets = batch[0].to(device), batch[1].to(device)
        logits = model(images)
        loss = self.loss_fn(logits, targets)
        return {"loss": loss, "loss_dict": {"loss": float(loss.detach())}}

    def eval_step(self, model, batch, device):
        images, targets = batch[0].to(device), batch[1].to(device)
        logits = model(images)
        loss = self.loss_fn(logits, targets)
        return {"loss": loss, "outputs": {"logits": logits, "targets": targets}}

    def update_metrics(self, outputs):
        preds = outputs["outputs"]["logits"].argmax(dim=1)
        for metric in self.metrics.values():
            metric.update(preds, outputs["outputs"]["targets"])

    def batch_size(self, batch):
        return batch[0].shape[0]

    def collate_fn(self):
        return None      # torch default_collate is sufficient
```

- `collate_fn`은 `None`이다. 타깃이 스칼라이므로 기본 collate로 충분하다(`PLAN-P1 §6.1`).
- `on_fit_start` 등 훅은 사용하지 않는다(기본 no-op).
- `predict_step`은 `{"stem", "pred_index", "pred_class", "confidence", "topk"}` 목록을 반환한다.

## 7. Postprocess와 Visualization

### 7.1. Postprocess (`FR-20`)

Classification의 후처리는 `softmax` → `argmax` → top-k 추출이다. 어댑터의 `predict_step` 안에서
수행하며 별도 모듈을 두지 않는다. `k`는 config `adapter.params.topk`로 지정하고 기본값 5다.

### 7.2. Visualization (`FR-22`)

- `outputs/.../visualizations/`에 예측 그리드 이미지를 저장한다. 각 타일에 GT 품종명과 예측 품종명,
  confidence를 표기하고, 오분류는 제목에 `[X]` 표시를 붙인다.
- 최대 장수는 `output.max_visualizations`(기본 16)를 따른다.
- 이모지를 사용하지 않는다(`CON-14`).

## 8. Config 구성

### 8.1. 파일 목록

| 파일 | 역할 |
|---|---|
| `configs/classification/_base.yaml` | 데이터·transform·loss·metric·adapter·optim·train 통제 조건 |
| `configs/classification/custom_cnn.yaml` | `_base` 상속 + `model` 블록 |
| `configs/classification/resnet50.yaml` | `_base` 상속 + `model` 블록 |
| `configs/classification/efficientnet_b0.yaml` | `_base` 상속 + `model` 블록 |
| `configs/benchmarks/classification_baseline.yaml` | 3 split 벤치마크 (`FR-28`) |

### 8.2. `_base.yaml` 통제 조건

```yaml
meta: {task_name: classification}
runtime: {seed: 42, device: cuda, amp: false, deterministic: warn, allow_network: false}
data:
  name: oxford_pets_cls
  root: /mnt/d/datasets/oxford_pets
  image_size: [224, 224]
  batch_size: 32
  num_workers: 4
  drop_last: false
  split: {mode: file, path: configs/splits/oxford_pets_subset_cls.json}
  transform:
    train: {name: cls_train, params: {}}
    eval:  {name: cls_eval,  params: {}}
loss: {name: cross_entropy, params: {label_smoothing: 0.0}}
metrics:
  - {name: top1_accuracy, params: {num_classes: 37}}
  - {name: macro_f1,      params: {num_classes: 37}}
adapter: {name: classification, params: {topk: 5}}
optim:
  optimizer: {name: adamw,  params: {lr: 0.0003, weight_decay: 0.0001}}
  scheduler: {name: cosine, params: {t_max: 5, eta_min: 0.000001}}
train: {epochs: 5, grad_clip: null, monitor: {metric: top1_accuracy, mode: max}, log_interval: 10, save_last: true}
```

`lr = 3e-4` AdamW는 pretrained fine-tuning과 from-scratch 학습 모두에서 5 epoch 규격에 무리가 없는 값으로
선택했다. 모델별 lr 튜닝은 공정 비교 통제 위반이므로 하지 않는다(`NFR-02`, `OUT-11`).

### 8.3. 벤치마크 config (`FR-28`)

`PLAN-P1 §10.1` 형식을 따른다. `base`는 `configs/classification/_base.yaml`이고 split 3개는
`model` 블록만 override한다.

- 자유 축은 `model`뿐이다.
- `batch_size = 32`, `image_size = 224`에서 세 모델 모두 GTX 1080 Ti 11.8GB에 들어가므로
  `control.exceptions`는 **비어 있다**. P2.3에서 OOM이 발생하면 예외를 추가하지 않고 먼저
  `_base.yaml`의 `batch_size`를 세 split 공통으로 낮춘다. 통제를 유지하는 쪽이 우선이다.

## 9. 하위 단계와 수용 기준

`PLAN.md` 2.3 템플릿을 그대로 따른다.

| 단계 | 주체 | 산출물 | 수용 기준 |
|---|---|---|---|
| P2.1 | task 에이전트 | Dataset, split 생성 스크립트와 split JSON 2종, transform 2종, loss, metric 2종, adapter, postprocess, visualize, `_base.yaml`, §4 모델 인터페이스 명세 | 더미 배치가 Dataset → collate → adapter → metric까지 통과. `assert_disjoint` 통과. subset 샘플 수가 §3.3과 일치 |
| P2.2 | 모델 에이전트 3 (동시) | 모델 3종 파일과 config 3종 | 각 모델이 `(2, 3, 224, 224)` 더미 입력에서 `(2, 37)` logits 반환. pretrained 2종은 로컬 가중치 로드 성공, 누락 시 `LocalAssetError` |
| P2.3 | task 에이전트 | 3모델 통합 | subset 5 epoch 학습·평가·추론 3모델 완주. `predictions/`와 `visualizations/` 생성 |
| P2.4 | task 에이전트 | 벤치마크 config, leaderboard | 통제 검사 통과, `leaderboard.csv`/`.md` 생성. params/FLOPs/FPS 3열이 모두 채워짐 |
| P2.5 | master | 적대적 검증, 보고 | 미해결 Critical 없음. `reviews/A2.md` 기록 |

- 실행 시간 상한은 모델 1종당 10분이다(`PLAN-P1 §13`).
- 공통 코드 수정이 필요하면 `PLAN.md` 5.1의 변경 요청 절차를 따른다. task 에이전트가 `core`·`bench`를
  직접 수정하지 않는다.

## 10. 적대적 검증 초점

P2는 `PLAN.md` 3.1에서 적대적 검증 선택 Phase다. 수행 시 공격 축은 다음과 같으며,
`backlog.json`의 P2 `adversarialFocus`가 이 표를 사용한다.

| 축 | 공격 내용 | 대응 조항 |
|---|---|---|
| target 규약 | Dataset이 0-based 라벨을 반환하는가. `list.txt` 1-based를 그대로 흘리지 않는가 | §3.1, §3.4 |
| split 무결성 | train/valid/test가 상호 배타적인가. subset 추출이 결정적인가. valid만으로 모델을 선택하는가 | §3.2, §3.3, §5.2 |
| 공정 비교 통제 | 3 split의 해상도·정규화·augmentation·optimizer·epoch·seed가 동일한가. 모델별 lr 튜닝이 없는가 | §3.5, §8.2, §8.3 |
| 오프라인 | `weights=None` + 로컬 로드를 지키는가. 가중치 누락 시 무작위 초기화로 폴백하지 않는가 | §4.2 |
| 평가 무결성 | metric 리셋, `eval()`/`no_grad()`가 엔진 계약대로 적용되는가. test가 학습 경로에 노출되지 않는가 | §6, `PLAN-P1 §7.2` |
| 엔진 순수성 | 태스크 구현이 `core`·`bench`를 수정하지 않았는가 | §9, `PLAN-P1 §7.4` |

## 11. 조항 개정 이력

| 일자 | 조항 | 등급 | 변경 내용 | 요청자 | 승인 |
|---|---|---|---|---|---|
| 2026-08-18 | 전체 | — | 최초 작성 | master | — |

---

*작성일: 2026-08-18 · 버전: v0.1 · 상위 문서: PLAN.md · 다음 단계: plans/PLAN-P3-segmentation.md*
