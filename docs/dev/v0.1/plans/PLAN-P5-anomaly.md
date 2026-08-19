# PLAN-P5 — Anomaly Detection

## 1. 문서 목적

이 문서는 `PLAN.md` 3.6의 P5(Anomaly Detection)를 구현 가능한 수준으로 상세화한다. 절 번호
(`PLAN-P5 §4.1` 형식)가 적대적 검증과 `backlog.json`이 참조하는 조항 ID다.

- 범위: MVTec `bottle` 파이프라인, image/pixel AUROC, threshold 정책, 모델 3종의 훅 사용 명세, 벤치마크 config
- 비범위: 공통 계약 → `PLAN-P1`
- 상위 문서: `PLAN.md` 3.6, `PLAN-P1`, `PRD.md`

적대적 검증 **필수** Phase다(`PLAN.md` 3.1). `PLAN.md` 2.3 템플릿과의 편차는 §9.1에 정리한다.

관련 요구사항: `FR-10`, `FR-16`, `FR-18`, `FR-19`, `NFR-03`, `CON-02`, `CON-05`, `AC-06`.

## 2. 태스크 정의

| 항목 | 값 |
|---|---|
| 학습 | 정상(`good`) 이미지만 사용. 라벨 없음 |
| 평가 | image-level 이상 점수 + pixel-level 이상 맵 |
| target 규약 (train) | `{}` (빈 dict) |
| target 규약 (eval) | `{"label": Tensor(scalar) long, "mask": Tensor(H, W) long}` (`PLAN-P1 §6.1`) |
| 데이터셋 | `/mnt/d/datasets/mvtec/bottle` (`CON-09`) |
| Metric | image-AUROC, pixel-AUROC (`FR-19`, `FR-16`) |
| 비교 모델 | Custom AE / STFPM / EfficientAD (`PRD.md` 2.1) |

비교 축은 **속도 대 정확도**다. `bottle`은 GT mask를 제공하므로 pixel-level 산출이 **필수**다(`FR-16`).

### 2.1. 구현 원칙 (`CON-01`, `CON-02`)

- anomalib를 **설치하지 않는다.** PyTorch 모델 정의(`nn.Module`)만 저장소로 복사한다
  (2026-08-18 사용자 결정, `PLAN.md` 3.6). 런타임 의존성을 만들지 않는다.
- 복사한 파일 상단에 출처(anomalib 저장소 경로·리비전)와 라이선스(Apache-2.0) 주석을 영어로 남긴다.
- Lightning을 사용하지 않는다(`OUT-04`). 학습 루프는 `PLAN-P1 §7.1`의 공통 엔진을 그대로 쓴다.
- 외부 라이브러리 임포트를 최소화하고 PyTorch 수준에서 직접 작성한다. 참조 구현은
  `github.com/nampluskr/defectvad`다.

## 3. 데이터

### 3.1. 원본 구조 실측 (2026-08-18)

| 경로 | 개수 | 비고 |
|---|---|---|
| `train/good` | 209 | 정상만. 라벨 없음 |
| `test/good` | 20 | 정상 |
| `test/broken_large` | 20 | 이상 |
| `test/broken_small` | 22 | 이상 |
| `test/contamination` | 21 | 이상 |
| `ground_truth/broken_large` | 20 | mask |
| `ground_truth/broken_small` | 22 | mask |
| `ground_truth/contamination` | 21 | mask |

- 이미지 해상도는 900x900이다.
- mask 픽셀 값은 `{0, 255}`이며 파일명은 `<stem>_mask.png`다.
- `test/good`에는 대응하는 mask 파일이 **없다.** Dataset이 `(H, W)` 영행렬을 생성해 채운다.
  마스크가 없다는 이유로 pixel metric에서 정상 이미지를 제외하면 pixel-AUROC가 과대평가된다.
- mask는 `{0, 255}` → `{0, 1}`로 변환한다. 255를 그대로 두면 AUROC 라벨이 깨진다.

### 3.2. split 정책 (`FR-17`, `NFR-03`, `AC-10`)

MVTec에는 공식 valid split이 없다. `PLAN-P1 §13.1`이 확정한 정책을 수치로 구체화한다.

threshold 결정과 모델 선택은 valid만 사용해야 하므로(`NFR-03`) valid에 이상 샘플이 필요하다.
따라서 `test` 83장을 **결함 유형별 층화 추출**로 valid와 test로 나눈다.

| 원본 폴더 | 전체 | valid (40%) | test (60%) |
|---|---|---|---|
| `test/good` | 20 | 8 | 12 |
| `test/broken_large` | 20 | 8 | 12 |
| `test/broken_small` | 22 | 9 | 13 |
| `test/contamination` | 21 | 8 | 13 |
| 합계 | 83 | **33** | **50** |

- `train`은 `train/good` 209장 전량이다. 정상만 포함하므로 이상 샘플이 없다.
- 층화 추출 seed는 `42`, 반올림은 `round(n * 0.4)`다.
- 결과는 `configs/splits/mvtec_bottle.json`으로 materialize하여 커밋한다(`PLAN-P1 §8.2` file 모드).
  샘플 ID는 `<유형>/<stem>` 형식(예: `broken_large/000`)이며, `train/good`과 `test/*`는 원본 폴더가
  다르므로 ID 충돌이 없다.
- `PLAN-P1 §8.3`의 `assert_disjoint`가 세 split의 상호 배타성을 검증한다.
- **valid는 threshold 결정과 모델 선택 전용이다. test는 최종 평가에서만 사용한다.** 엔진이
  `allow_test_split` 플래그로 이를 강제한다(`PLAN-P1 §7.3`).

### 3.3. subset 규격 (`PLAN-P1 §13`)

`bottle`은 규모가 작아 **축소하지 않고 전량을 사용한다.**

| split | 샘플 수 |
|---|---|
| `train` | 209 |
| `valid` | 33 |
| `test` | 50 |

209장 5 epoch에 256x256 해상도이므로 15분 상한 안에 들어간다. 별도 subset split 파일을 만들지 않는다.

### 3.4. Dataset 계약 (`FR-11`, `FR-16`, `NFR-06`)

```python
@DATASETS.register("mvtec_anomaly")
class MVTecAnomaly(Dataset):
    """train split returns (image, {}).
    valid/test splits return (image, {"label": LongTensor scalar, "mask": LongTensor(H, W)})."""

    def __init__(self, root, split, transform=None, split_path=None, category="bottle"):
        ...
```

- `label`은 `0=normal`, `1=anomalous`다.
- `mask`는 항상 존재한다. 정상 이미지는 영행렬, 이상 이미지는 `ground_truth`에서 읽어 `{0,1}`로 변환한다.
- **train split의 target은 빈 dict `{}`다.** `None`을 반환하지 않는다. `collate_fn`이 list-of-dict를
  유지하므로 빈 dict가 그대로 배치에 들어간다(`PLAN-P1 §6.1`).
- 카테고리는 `data.params.category`로 지정한다. v0.1은 `bottle` 1개다(`CON-09`, `OUT-06`). 다른 카테고리로
  바꾸려면 config의 `category`만 수정하면 되고 코드는 수정하지 않는다.
- 자동 다운로드를 하지 않는다(`CON-03`).

### 3.5. Transform (`FR-21`)

**Anomaly에는 augmentation을 적용하지 않는다.** 좌우 반전이나 crop이 정상 패턴을 변형해 거짓 이상을
만들고, 정상 분포 학습이라는 전제를 흐린다. train과 eval이 동일한 결정적 변환을 쓴다.

```python
@TRANSFORMS.register("anomaly_default")
# v2.Compose([
#     v2.Resize(image_size, antialias=True),
#     v2.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64}, scale=True),
#     v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
# ])
```

- mask는 `tv_tensors.Mask`로 감싸 NEAREST 보간을 보장한다(`PLAN-P3 §3.5`와 동일 원칙).
- mask에 `scale=True`나 Normalize가 적용되면 안 된다.
- `image_size`는 `[256, 256]`으로 고정한다. anomaly map과 mask가 같은 해상도여야 pixel-AUROC가 성립한다.
- `data.transform.train`과 `data.transform.eval`에 같은 `anomaly_default`를 지정한다.

## 4. 모델 인터페이스 명세

P5.1의 필수 산출물이다. **세 모델의 학습 방식이 모두 다르므로**(재구성 / teacher-student /
teacher + autoencoder + quantile 정규화) 이 명세와 §5의 훅 사용 명세가 없으면 모델 에이전트 3개가
각자 다른 방식으로 우회한다(`PLAN.md` 3.6).

### 4.1. 시그니처

`PLAN-P1 §6.3`의 `model.train_step` 경로를 사용한다(`FR-10`).

```python
@MODELS.register("<model_name>")
class <ModelName>(nn.Module):
    def __init__(self, weights_path=None, **params):
        ...

    def train_step(self, images, targets):
        """images: Tensor(B, 3, H, W). targets: list of empty dict (unused).
        returns: {"loss": scalar Tensor with grad, "loss_dict": {str: float}}"""

    def forward(self, images):
        """Inference. images: Tensor(B, 3, H, W).
        returns: {"pred_score": Tensor(B,) float, "anomaly_map": Tensor(B, H, W) float}
        anomaly_map is at input resolution. Higher means more anomalous."""
```

- `anomaly_map`은 **입력 해상도 `(B, H, W)`**로 반환한다. `(B, 1, H, W)`나 축소 해상도를 반환하지 않는다.
  해상도 복원은 모델 내부에서 `F.interpolate(..., mode="bilinear", align_corners=False)`로 수행한다.
- `pred_score`는 이미지 단위 이상 점수다. 기본 정의는 `anomaly_map`의 최댓값이며, 모델이 다른 정의를
  쓰면 그 근거를 파일 docstring에 남긴다.
- 점수는 **정규화하지 않은 원시 값**이어도 된다. AUROC는 단조 변환에 불변이다. threshold는 §6에서
  valid로 결정한다.
- `targets`는 train에서 사용하지 않는다. 시그니처만 공통 규약에 맞춘다.

### 4.2. 고정 서브모듈의 train/eval 모드 (필수)

STFPM과 EfficientAD는 **고정된 teacher**를 갖는다. 엔진이 매 epoch `model.train()`을 호출하므로
(`PLAN-P1 §7.1`) 아무 조치가 없으면 **teacher의 BatchNorm running statistics가 갱신되고 dropout이 켜진다.**
teacher가 조용히 변하면 student가 쫓는 목표가 흔들려 비교가 무의미해진다.

모델은 `train()`을 오버라이드해 고정 서브모듈을 항상 eval로 유지한다.

```python
def train(self, mode=True):
    super().train(mode)
    self.teacher.eval()             # frozen module stays in eval mode
    return self
```

추가로 다음을 지킨다.

- teacher 파라미터는 `requires_grad_(False)`로 고정한다.
- optimizer에는 학습 대상 파라미터만 넘긴다. `PLAN-P1 §7.1`의 optimizer 빌더는
  `filter(lambda p: p.requires_grad, model.parameters())`를 사용한다.
- P5.2 수용 기준에 "학습 전후 teacher의 `state_dict` 해시가 동일함"을 포함한다.

### 4.3. 가중치 로드 (`CON-05`, `PLAN-P1 §9.2`)

| 모델 | 로컬 가중치 | 비고 |
|---|---|---|
| Custom AE | 없음 | from scratch |
| STFPM | `/mnt/d/backbones/resnet18-f37072fd.pth` | torchvision ResNet18 ImageNet. teacher 전용 |
| EfficientAD | `/mnt/d/backbones/efficientad_pretrained_weights/pretrained_teacher_small.pth` | PDN-Small teacher |

- EfficientAD teacher 가중치는 실측 확인 결과 `conv1..conv4`의 weight/bias 8개 키를 갖는 PDN-Small
  구조다(`pretrained_teacher_medium.pth`는 `conv1..conv6`의 12개 키). 복사한 모델 정의의 레이어 이름이
  이와 다르면 `key_map`으로 맞춘다. 이름이 안 맞는다고 `strict=False`로 넘기지 않는다.
- 파일이 없으면 `LocalAssetError`로 실패하며 무작위 초기화로 폴백하지 않는다. teacher가 무작위이면
  STFPM·EfficientAD가 학습은 되지만 성능이 무의미해지므로 조용한 폴백은 치명적이다.
- STFPM의 student는 teacher와 동일 구조를 무작위 초기화한다. student에 pretrained를 넣으면 안 된다.

### 4.4. 모델 3종 배정 (`FR-18`)

| 에이전트 | registry key | 파일 | config | 학습 방식 |
|---|---|---|---|---|
| 모델 에이전트 1 | `custom_ae_anomaly` | `src/tasks/anomaly/models/custom_ae.py` | `configs/anomaly/custom_ae.yaml` | 재구성 오차 |
| 모델 에이전트 2 | `stfpm_anomaly` | `.../models/stfpm.py` | `configs/anomaly/stfpm.yaml` | teacher-student feature matching |
| 모델 에이전트 3 | `efficientad_anomaly` | `.../models/efficientad.py` | `configs/anomaly/efficientad.yaml` | teacher + student + autoencoder |

각 에이전트는 위 2개 파일만 생성·수정한다.

- **Custom AE**: conv encoder-decoder. loss는 재구성 MSE. `anomaly_map`은 픽셀별 재구성 오차
  (채널 평균), `pred_score`는 그 최댓값. pretrained 없음.
- **STFPM**: teacher = torchvision ResNet18(로컬 가중치, 고정), student = 동일 구조 무작위 초기화.
  loss는 선택 레이어(`layer1`, `layer2`, `layer3`)의 L2 정규화 feature 간 MSE 합.
  `anomaly_map`은 레이어별 오차 맵을 입력 해상도로 올려 곱하거나 더한 값.
- **EfficientAD**: PDN-Small teacher(고정) + student + autoencoder. loss는 student-teacher 거리의
  하드 마이닝 항, autoencoder 재구성 항, student-autoencoder 항의 합.
  `on_fit_start`에서 채널 정규화 통계, `on_fit_end`에서 quantile 정규화 상수를 산출한다(§5).

## 5. 훅 사용 명세 (`FR-09`, `FR-10`)

P5.1의 필수 산출물이며 템플릿 편차다(`PLAN.md` 3.6). 세 모델이 `PLAN-P1 §6.5`의 훅을 어떻게 쓰는지
여기서 지정한다. 명세에 없는 훅을 모델이 임의로 사용하면 P5.3 통합에서 반려한다.

| 훅 | Custom AE | STFPM | EfficientAD |
|---|---|---|---|
| `on_fit_start` | 사용 안 함 | 사용 안 함 | **사용**. train 로더를 1회 순회해 teacher 출력의 채널별 평균·표준편차를 산출하고 모델에 저장 |
| `on_epoch_start` | 사용 안 함 | 사용 안 함 | 사용 안 함 |
| `on_epoch_end` | 사용 안 함 | 사용 안 함 | 사용 안 함 |
| `on_fit_end` | threshold 결정 | threshold 결정 | **quantile 정규화 상수 산출 + threshold 결정** |

- 훅은 **어댑터가 구현하고 모델에 위임한다.** 어댑터의 `on_fit_start`는 `hasattr(model, "on_fit_start")`를
  확인해 있으면 호출한다. 엔진은 어댑터만 호출한다(`PLAN-P1 §6.5`).
- `on_fit_start`와 `on_fit_end`에 전달되는 `loaders`에는 `train`과 `valid`만 있다. **test는 없다.**
  EfficientAD의 quantile 정규화가 test 분포를 보면 명백한 누수다(`NFR-03`).
- EfficientAD의 채널 정규화는 `no_grad`와 `model.eval()` 상태에서 수행한다. 통계 산출 중 BN이 갱신되면
  §4.2 위반이다.

## 6. threshold 정책 (`NFR-03`)

- threshold는 **valid split만으로 결정한다.** `on_fit_end` 훅에서 valid의 image score와 label을 모아
  F1이 최대가 되는 지점을 선택한다.
- 결정된 threshold는 `metrics_final.json`의 `image_threshold` 필드와 체크포인트에 저장한다.
- pixel threshold도 같은 방식으로 valid의 pixel score와 mask에서 결정하고 `pixel_threshold`로 저장한다.
- **threshold는 AUROC 계산에 사용하지 않는다.** AUROC는 threshold-free 지표다. threshold는 시각화의
  이진 마스크 표시와 참고용 F1 기록에만 쓴다.
- test 데이터로 threshold를 조정하는 경로가 존재하면 `AC-10` 위반이다.

## 7. Metric, TaskAdapter, 시각화

### 7.1. Metric (`FR-19`, `FR-16`, `AC-06`)

| registry key | 구현 | 입력 |
|---|---|---|
| `image_auroc` | `torchmetrics.classification.BinaryAUROC()` | `pred_score (B,)` 와 `label (B,)` |
| `pixel_auroc` | `torchmetrics.classification.BinaryAUROC()` | `anomaly_map.flatten()` 과 `mask.flatten()` |

- 두 metric을 **모두 산출한다.** `bottle`은 GT mask를 보유하므로 pixel-level이 필수다(`FR-16`).
- pixel-AUROC 누적 규모는 `50장 x 256 x 256 = 3.3M` 원소로 메모리에 무리가 없다.
- 정상 이미지도 pixel metric에 포함한다(§3.1).
- `train.monitor`는 `{metric: image_auroc, mode: max}`다.

### 7.2. TaskAdapter

```python
@ADAPTERS.register("anomaly")
class AnomalyAdapter(TaskAdapter):
    def train_step(self, model, batch, device):
        images = batch[0].to(device)
        return model.train_step(images, batch[1])       # targets are empty dicts

    def eval_step(self, model, batch, device):
        images = batch[0].to(device)
        outputs = model(images)
        labels = torch.stack([t["label"] for t in batch[1]]).to(device)
        masks  = torch.stack([t["mask"]  for t in batch[1]]).to(device)
        return {"loss": None, "outputs": {"scores": outputs["pred_score"],
                                          "maps": outputs["anomaly_map"],
                                          "labels": labels, "masks": masks}}

    def on_fit_start(self, model, loaders, device):
        if hasattr(model, "on_fit_start"):
            model.on_fit_start(loaders["train"], device)

    def on_fit_end(self, model, loaders, device):
        if hasattr(model, "on_fit_end"):
            model.on_fit_end(loaders["valid"], device)
        self.image_threshold, self.pixel_threshold = compute_thresholds(model, loaders["valid"], device)

    def batch_size(self, batch):
        return batch[0].shape[0]

    def collate_fn(self):
        return anomaly_collate       # (Tensor(B,3,H,W), list[dict])
```

- `anomaly_collate`는 이미지를 스택하고 타깃은 list-of-dict로 유지한다. train의 빈 dict도 그대로 통과한다.
- `eval_step`의 `loss`는 `None`이다. 세 모델의 loss 정의가 달라 valid loss를 비교할 수 없다.
  모델 선택은 `image_auroc`로 한다.
- config의 `loss` 블록은 `{name: none}`이다(`PLAN-P4 §5`와 동일). loss는 모델에 내장된다.

### 7.3. Postprocess와 Visualization (`FR-20`, `FR-22`)

- 후처리는 `anomaly_map`에 대한 gaussian smoothing(`sigma`는 `adapter.params.smooth_sigma`, 기본 4.0)이다.
  세 모델에 동일하게 적용한다. 모델별로 다르면 통제 위반이다.
- 시각화는 `원본 / GT mask / anomaly map heatmap / threshold 이진화` 4분할 이미지를 저장한다.
- 정상 이미지와 이상 이미지를 모두 포함하도록 저장 대상을 선택한다.

## 8. Config 구성

### 8.1. `_base.yaml` 통제 조건

```yaml
meta: {task_name: anomaly}
runtime: {seed: 42, device: cuda, amp: false, deterministic: warn, allow_network: false}
data:
  name: mvtec_anomaly
  root: /mnt/d/datasets/mvtec
  params: {category: bottle}
  image_size: [256, 256]
  batch_size: 8
  num_workers: 4
  drop_last: false
  split: {mode: file, path: configs/splits/mvtec_bottle.json}
  transform:
    train: {name: anomaly_default, params: {}}
    eval:  {name: anomaly_default, params: {}}
loss: {name: none, params: {}}
metrics:
  - {name: image_auroc, params: {}}
  - {name: pixel_auroc, params: {}}
adapter: {name: anomaly, params: {smooth_sigma: 4.0}}
optim:
  optimizer: {name: adamw,  params: {lr: 0.0001, weight_decay: 0.00001}}
  scheduler: {name: cosine, params: {t_max: 5, eta_min: 0.000001}}
train: {epochs: 5, grad_clip: null, monitor: {metric: image_auroc, mode: max}, log_interval: 10, save_last: true}
```

파일 목록은 `_base.yaml`, `custom_ae.yaml`, `stfpm.yaml`, `efficientad.yaml`,
`configs/benchmarks/anomaly_baseline.yaml`이다.

### 8.2. 벤치마크 config

- 자유 축은 `model`뿐이다. `control.exceptions`는 비어 있다.
- 5 epoch은 EfficientAD의 표준 학습량(수만 스텝)에 크게 못 미친다. 절대 성능이 낮게 나오는 것은
  예상된 결과이며 v0.1의 수용 기준이 아니다(`PLAN.md` 9). leaderboard 각주에 명시한다.

## 9. 하위 단계와 수용 기준

### 9.1. 템플릿 편차 (`PLAN.md` 2.3, 3.6)

- P5.1에 **threshold 결정 정책**(§6)이 추가된다.
- P5.1의 필수 산출물에 **훅 사용 명세**(§5)가 포함된다.

### 9.2. 단계별 수용 기준

| 단계 | 주체 | 산출물 | 수용 기준 |
|---|---|---|---|
| P5.1 | task 에이전트 | Dataset, split JSON, transform, metric 2종, adapter, `anomaly_collate`, threshold 로직, visualize, `_base.yaml`, §4 인터페이스 명세, §5 훅 명세 | 더미 배치가 Dataset → collate → adapter → metric까지 통과. train 타깃이 빈 dict임을 확인. mask 값이 `{0,1}`이고 정상 이미지에 영행렬 mask가 생성됨. `assert_disjoint` 통과 |
| P5.2 | 모델 에이전트 3 (동시) | 모델 3종과 config 3종 | 각 모델이 `(2, 3, 256, 256)`에서 `train_step` loss와 `forward` 출력(`pred_score (2,)`, `anomaly_map (2, 256, 256)`)을 반환. **학습 전후 teacher `state_dict` 해시 동일**(§4.2). 로컬 가중치 로드 성공, 누락 시 `LocalAssetError` |
| P5.3 | task 에이전트 | 3모델 통합 | 5 epoch 학습·평가·추론 3모델 완주. **image-AUROC와 pixel-AUROC가 모두 산출**(`AC-06`). 4분할 시각화 생성 |
| P5.4 | task 에이전트 | 벤치마크 config, leaderboard | 통제 검사 통과, leaderboard 생성. 속도 대 정확도 비교 축이 FPS 열과 함께 확인됨 |
| P5.5 | master | **적대적 검증(필수)**, 보고 | 미해결 Critical 없음. `reviews/A5.md` 기록 |

실행 시간 상한은 모델 1종당 15분이다(`PLAN-P1 §13`).

## 10. 적대적 검증 초점

`backlog.json`의 P5 `adversarialFocus`가 이 표를 사용한다.

| 축 | 공격 내용 | 대응 조항 |
|---|---|---|
| 학습/평가 누수 | threshold나 quantile 정규화가 test를 보는가. `on_fit_end`에 test loader가 전달되는가. 모델 선택이 valid 기준인가 | §3.2, §5, §6 |
| 고정 모듈 무결성 | teacher가 `model.train()`으로 BN이 갱신되는가. teacher 파라미터가 optimizer에 들어가는가 | §4.2 |
| 오프라인 | teacher 가중치가 없을 때 무작위 초기화로 폴백하는가. anomalib를 런타임 임포트하는가 | §2.1, §4.3 |
| target 규약 | train 타깃이 빈 dict인가. mask가 `{0,1}`인가. 정상 이미지 mask가 생성되는가 | §3.1, §3.4 |
| pixel metric 정확성 | `anomaly_map`이 입력 해상도인가. 정상 이미지가 pixel-AUROC에서 제외되는가 | §4.1, §7.1 |
| 공정 비교 통제 | smoothing이 모델별로 다른가. augmentation이 한 모델에만 적용되는가. optimizer·epoch이 동일한가 | §3.5, §7.3, §8.1 |
| 엔진 순수성 | 모델별 학습 방식 분기가 엔진이 아니라 모델·어댑터에 있는가. 훅 명세 밖의 우회가 있는가 | §4.1, §5, `PLAN-P1 §7.4` |

## 11. 조항 개정 이력

| 일자 | 조항 | 등급 | 변경 내용 | 요청자 | 승인 |
|---|---|---|---|---|---|
| 2026-08-18 | 전체 | — | 최초 작성 | master | — |
| 2026-08-19 | §2.1, §4.4 | — | `github.com/nampluskr/defectvad`에 오프라인 샌드박스에서 접근 불가해 `stfpm_anomaly`/`efficientad_anomaly`를 원 논문과 §4 인터페이스 계약으로부터 직접 구현(`ISS-05`). 계약·수용 기준 변경 없음 | master | master (2026-08-19) |
| 2026-08-19 | §6 | A | `best_f1_threshold`의 tie 처리 버그를 P5 Codex 적대적 검증(`reviews/A5.md` Major #1)에서 발견해 수정. 동일 정렬 점수 구간의 마지막 위치만 유효 threshold 후보로 마스킹. image/pixel threshold 계산 결과가 달라질 수 있으나 §6의 정책(valid-only, F1 최대화) 자체는 불변 | Codex A5 재검토 반영 | master (2026-08-19) |

---

*작성일: 2026-08-18 · 버전: v0.1 · 상위 문서: PLAN.md · 다음 단계: plans/PLAN-P6-integration.md*
