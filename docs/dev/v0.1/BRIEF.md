# BRIEF — PyTorch CV Boilerplate v0.1

## 1. 문서 목적

이 문서는 PyTorch 기반 Computer Vision boilerplate 프로젝트의 **요구사항과 의도**를 정리한다.
구현 방법(아키텍처 상세, 파일 단위 설계, 작업 순서)은 이 문서를 기반으로 별도 `PLAN.md`에서 다룬다.

- 범위: 요구사항 정의 (What / Why)
- 비범위: 구현 설계 (How) → `PLAN.md`

## 2. 프로젝트 목표

하나의 공통 학습 뼈대(engine) 위에서 **4개 CV 태스크**를 다루고,
각 태스크마다 **custom 모델과 pretrained 모델을 비교**하여 성능 차이를 정량적으로 확인하는
**재사용 가능한 boilerplate**를 만든다.

**모델은 향후 계속 추가될 예정**이며, 이 boilerplate는 **다양한 모델과 학습조건을 한꺼번에
학습·평가·추론하여 결과를 비교하는 벤치마크 도구**로서 기능해야 한다.

핵심 의도:

1. **학습 루프는 공통, 데이터·모델·손실·평가는 태스크별** — 엔진은 태스크를 몰라야 한다(task-agnostic).
2. **custom vs pretrained 비교** — 태스크별로 직접 설계한 CNN과 사전학습 모델의 정확도/효율 차이를 같은 조건에서 측정한다.
3. **확장성 검증** — 4개 태스크를 순차 통합하며 공통 엔진의 일반성과 한계선을 실증한다.
4. **벤치마크 도구화** — 모델을 지속적으로 추가할 수 있고, 여러 모델·조건을 일괄 학습·평가·추론해 결과를 한 표로 비교한다(7.1).

## 3. 대상 태스크

| 태스크 | 설명 |
|---|---|
| Classification | 이미지 → 클래스 |
| Segmentation | 이미지 → 픽셀 단위 마스크 |
| Object Detection | 이미지 → 박스 + 클래스 |
| Anomaly Detection | 정상 이미지 학습 → 이상 탐지 |

## 4. 태스크별 비교 모델 (custom 1 + pretrained 2)

각 태스크당 3종 모델을 **동일 조건**에서 비교한다.

| 태스크 | Custom | Pretrained 1 | Pretrained 2 | 비교 축 |
|---|---|---|---|---|
| Classification | Custom CNN | ResNet50 (torchvision) | EfficientNet-B0 (torchvision) | 정확도 vs 파라미터/효율 |
| Segmentation | Custom (U-Net류) | DeepLabV3-ResNet50 (torchvision) | FCN-ResNet50 (torchvision) | head 구조(ASPP 유무) 효과 |
| Detection | Custom (1-stage 권장) | YOLO (ultralytics, v8/v11) | Faster R-CNN ResNet50-FPN (torchvision) | 1-stage vs 2-stage |
| Anomaly Detection | Custom (Reconstruction/AE류) | STFPM (anomalib, 모델만) | EfficientAD (anomalib, 모델만) | 속도 vs 정확도 |

비고:

- anomalib는 **모델(nn.Module)만** 가져와 공통 어댑터에 감싼다. **Lightning 등 anomalib의 학습 프레임워크는 사용하지 않고**, 학습 루프는 `github.com/nampluskr/defectvad`에 현재 구현된 pure-PyTorch 학습 boilerplate를 참조·사용한다. STFPM/EfficientAD처럼 학습 방식이 다른 경우는 `train_step`을 모델별로 분기한다.
- Anomaly 구현은 **외부 라이브러리 임포트를 최소화하고 PyTorch 수준에서 직접 작성**하는 것을 원칙으로 한다(참조: `github.com/nampluskr/defectvad`). anomalib는 모델 정의 차용에 한정한다.
- Segmentation의 torchvision 2종은 backbone(ResNet50)을 통일해 head 차이만 비교한다. 경량축이 필요하면 FCN 대신 LRASPP-MobileNetV3로 교체 가능.
- Detection의 torchvision 모델은 2-stage 대표로 Faster R-CNN 선정. 1-stage 통일을 원하면 RetinaNet/FCOS로 교체 가능. custom 모델 포함 3종 모두 **multi-class / multi-object 출력**을 전제로 설계한다(6.1 규약).

## 5. 데이터셋 — 도메인 통일 확정

로컬 `/mnt/d/datasets` 검토 결과를 반영한다.
**Classification·Segmentation·Detection을 하나의 이미지셋(`oxford_pets`)으로 통일**하고,
Anomaly Detection만 `mvtec`을 사용한다.

> **전제**: 데이터셋과 백본 가중치는 프로젝트가 자동 다운로드하지 않는다. **사용자가 별도로 내려받아 로컬 경로에 저장**하며(데이터셋 → `/mnt/d/datasets`, 백본 가중치 → `/mnt/d/backbones`), 프로젝트는 해당 로컬 경로만 참조한다.

| 태스크 | 데이터셋 (로컬 경로) | 라벨 형태 | 규모 |
|---|---|---|---|
| Classification | `/mnt/d/datasets/oxford_pets` (images) | 37 breeds (또는 cat/dog 이진) | 7,390장 |
| Segmentation | `/mnt/d/datasets/oxford_pets` (annotations/trimaps) | trimap 3-class(전경/경계/배경) | 7,390 마스크 |
| Detection | `/mnt/d/datasets/oxford_pets` (annotations/xmls) | Pascal VOC XML bbox | 3,686 bbox |
| Anomaly Detection | `/mnt/d/datasets/mvtec` | train=good only, test + GT mask | 15 카테고리(초기엔 `bottle` 1개) |

### 5.1 도메인 통일 선택 의도

- **하나의 이미지셋으로 3개 태스크**를 수행 → 데이터 로딩·전처리 코드 재사용 극대화.
- "같은 데이터에서 태스크가 어떻게 달라지는가"(라벨 형태·모델 head·손실·평가)가 명확히 드러나 **교육적·구조적 가치**가 높다.
- `oxford_pets`는 images/trimaps/xmls를 모두 보유해 3태스크를 즉시 커버한다(파일명 1:1 매칭).

### 5.2 유의사항 / 제약

- **Detection 구현 요구사항**: 검출 파이프라인(Dataset·모델·Loss·Metric·Postprocess)은 **multi-class / multi-object를 일반적으로 지원**하도록 구현한다. 즉 target은 이미지당 임의 개수 N의 박스와 클래스를 담는 표준 형태여야 한다.
  - `oxford_pets`는 이 일반 구현 위에서 **N=1(이미지당 객체 1개, pet head), 소수 클래스**인 **특수 케이스**로 동작한다. 데이터가 단일 객체라는 이유로 파이프라인을 단일 객체 전용으로 축약하지 않는다.
  - `oxford_pets` xmls는 3,686장만 bbox 보유, 이미지당 객체 1개이므로 **실제 다객체 mAP 검증은 제한적**. 다객체·다클래스 성능 비교가 필요해지면 동일 파이프라인에 Pascal VOC 2007 / COCO128을 그대로 연결할 수 있어야 한다(현재 로컬에 없음, 데이터만 교체).
- **Anomaly 범위**: MVTec 전체(~5GB)를 초기에 전부 돌리지 않고 `bottle` 등 1~2개 카테고리로 파이프라인 검증 후 확장.

### 5.3 대안 자산 (참고, v0.1 비범위)

| 데이터셋 | 태스크 | 규모 | 용량(대략) | 용도 / 비고 |
|---|---|---|---|---|
| `visa` | Anomaly | 12 카테고리 | ~9 GB | anomaly 벤치마크 확장 |
| `btad` | Anomaly | 3 카테고리 | ~1 GB | anomaly 벤치마크 확장 |
| `imagenette2` | Classification | 10 classes | ~1.5 GB | 분류 대안(실해상도 객체) |
| `cifar10` | Classification | 10 classes / 6만장 | ~0.2 GB | 분류 대안(저해상도 객체 벤치) |
| COCO128 (ultralytics) | Detection | 128 images / 80 classes | ~7 MB | 파이프라인 스모크 테스트, multi-object/class 즉시 검증 (로컬 미보유) |
| Pascal VOC 2007 | Detection | ~9,963 images / 20 classes | ~870 MB | 표준 mAP@0.5 벤치마크, 다객체·다클래스 실검증 (로컬 미보유) |

## 6. 아키텍처 방향성 (요구 수준)

> **가장 중요한 지침 — 원점에서 재설계**: 이 boilerplate는 `260712_roi-corner-detection-ver2`의
> **CLI 사용 패턴만 참고**한다. 즉 "`scripts/`(config·train·evaluate·predict)와 `experiments/`(configs·run)를
> CLI에서 이렇게 실행하고 싶다"는 **사용 방식(사용자 의도)** 만 차용할 뿐이며, **기존 코드·폴더구조·파일을 재사용하거나 그대로 확장하지 않는다.**
> 폴더구조·파일 구성·모듈 경계는 이 문서의 목적/요구사항에서 출발해 **원점에서 최적의 형태로 재검토·재설계**한다. (roi 코드는 사용 패턴 이해를 위한 읽기 참고용일 뿐, 설계의 기준이 아니다.)

상세 설계는 `PLAN.md`에서 다루되, BRIEF 수준의 방향성은 다음과 같다.

- **공통(task-agnostic)**: Config 시스템, Trainer/Engine 루프, Registry(플러그인), Logger/Checkpoint, Optimizer/Scheduler 빌더, Device/AMP, 단일 CLI 진입점.
- **태스크별(task-specific)**: Dataset/라벨 파싱, 모델 head, Loss, Metric, Transform(augmentation), Postprocess(NMS/threshold), Visualization.
- **이음새**: Task 어댑터가 "한 배치를 어떻게 forward하고 loss/preds를 뽑는가"를 캡슐화하여 엔진과 태스크를 분리한다.
- 엔진은 4개 태스크에서 **수정 없이** 동작하는 것을 목표로 한다. Anomaly의 memory-bank류(예: PatchCore)처럼 표준 gradient 루프를 벗어나는 경우는 별도 훅/Trainer로 처리하며, 이것이 공통 엔진의 한계선으로 문서화된다.

### 6.1 라벨/타겟 형태 규약 (공통 적용 원칙)

각 Dataset이 반환하는 **target(라벨) 형태는 태스크별로 하나의 공통 규약**을 따른다.
같은 태스크 안에서는 어떤 데이터셋·어떤 모델을 쓰든 동일한 target 형태를 반환해야 하며,
모델·Loss·Metric은 그 규약에만 의존한다(특정 데이터셋 구조에 결합하지 않는다).
torchvision 표준 형태를 기본 규약으로 채택한다.

| 태스크 | target 형태 (공통 규약) | 비고 |
|---|---|---|
| Classification | `label: int (LongTensor scalar)` | 클래스 인덱스 |
| Segmentation | `mask: LongTensor (H, W)` | 픽셀별 클래스 인덱스 |
| Detection | `{"boxes": FloatTensor (N, 4), "labels": LongTensor (N)}` | **N은 가변(multi-object), 클래스 다수(multi-class)**. oxford_pets는 N=1 특수 케이스 |
| Anomaly Detection | 학습: 없음(정상만) / 평가: `{"label": int, "mask": LongTensor (H, W) 선택}` | image-level + 선택적 pixel-level |

- 이 규약 덕분에 Detection은 데이터만 교체(oxford_pets → VOC/COCO)해도 파이프라인이 그대로 동작한다.
- 가변 개수 target(Detection)은 공통 규약을 유지하기 위해 태스크별 `collate_fn`으로 배치를 구성한다(예: Detection은 list-of-dict).

## 7. 공정 비교를 위한 통제 조건

모델 간 성능 차이를 유의미하게 보려면 태스크 내에서 다음을 고정한다.

- 입력 해상도 / 정규화 / augmentation
- optimizer · scheduler · epoch · batch (모델 크기로 batch만 조정 시 명시)
- 평가 metric — Cls: Top-1/F1, Seg: mIoU/Dice, Det: mAP@0.5:0.95, Anomaly: image/pixel-AUROC
- 리포트 축 — 정확도뿐 아니라 **params / FLOPs / inference FPS**를 함께 기록 (custom 모델의 존재 의의를 드러냄)

### 7.1 벤치마크 산출물 (필수 요구사항)

벤치마크는 **하나의 태스크 안에서만** 수행하며, **태스크 간 비교는 하지 않는다**.
태스크는 최초에 한 번 정의(고정)되고, 그 안에서 **모델·학습조건에 대한 다양한 split(변형) 조건을 순차적으로 적용·비교**할 수 있어야 한다.

- **태스크 고정 + 변형 순차 적용**: 태스크(데이터·target 규약·평가 metric)를 고정한 상태에서, 모델(custom/pretrained)과 학습조건(해상도·optimizer·scheduler·epoch·augmentation 등)의 조합을 하나의 split으로 정의하여 순차 실행한다.
- **비교 리포트 자동 생성**: 같은 태스크 내 여러 split의 결과(표준 metric + params / FLOPs / inference FPS)를 한 표(leaderboard)로 집계해 나란히 비교한다.
- **재현성**: 각 split의 config·seed·결과를 기록하여 동일 결과를 재현할 수 있어야 한다.
- **최소 요구 split**: 각 태스크의 custom 1 + pretrained 2 = 3개 모델(동일 학습조건)을 기본 split으로 포함한다.

## 8. 기술 스택 (예정)

| 용도 | 라이브러리 |
|---|---|
| Config | YAML 파일 |
| Metric | torchmetrics (4태스크 공통) |
| Augmentation | torchvision.transforms.v2 (seg/det 라벨 동시 변환) |
| Backbone | torchvision.models (로컬 가중치) — timm은 선택(향후 확장용) |
| Segmentation | torchvision.models |
| Detection | torchvision.models.detection + ultralytics(YOLO) |
| Anomaly | anomalib (모델 정의만, Lightning 미사용) — 학습은 pure-PyTorch, 참조: `github.com/nampluskr/defectvad` |
| 실험 관리 | 별도 도구 미사용 — config·metric 기록 + leaderboard 표(CSV/Markdown) |

### 8.1 실행 환경

- **실행 환경**: Python 실행과 검증은 conda 환경 `pytorch_env`를 사용한다. 코드 실행·`python -c` 검증·스크립트 실행 전에 먼저 활성화한다.

  ```bash
  conda activate pytorch_env
  ```

- **백본 가중치(오프라인)**: pretrained 가중치는 인터넷 다운로드 없이 로컬 `/mnt/d/backbones`에서 로드한다. v0.1 pretrained 모델은 모두 `torchvision.models`에 존재하므로 `weights=None`으로 아키텍처를 만든 뒤 로컬 `.pth`를 `load_state_dict`로 주입한다. **timm은 필수가 아니며**, torchvision에 없는 백본(예: deit/cait/dinov2)이 필요해질 때만 선택적으로 도입한다.

- **산출물 경로·import 등 코드 구조**: 폴더구조·산출물 경로 규칙·import 방식은 기존 프로젝트를 따르지 않고 6장 지침대로 **원점에서 재설계**하여 `PLAN.md`에서 확정한다.

## 9. 성공 기준 (v0.1)

1. 공통 엔진 위에서 4개 태스크가 동일 CLI/Config 흐름으로 학습·평가된다.
2. 각 태스크에서 custom / pretrained 2종 = 3개 모델이 동일 조건으로 학습·비교된다.
3. 태스크별 표준 metric + params/FLOPs/FPS가 함께 리포트된다.
4. 태스크를 고정한 상태에서 모델·학습조건 split을 순차 적용하고, 그 결과를 한 표(leaderboard)로 비교하는 벤치마크 리포트가 생성된다(태스크 간 비교는 비범위, 7.1).
5. `oxford_pets` 단일 이미지셋이 Classification/Segmentation/Detection 3태스크에 재사용된다.
6. Anomaly는 MVTec `bottle`에서 custom/STFPM/EfficientAD 비교가 동작한다.

## 10. 결정 사항

기존 미결정 항목은 기본안으로 확정한다.

- Classification 라벨 정의: **37 breeds 다중분류**로 확정.
- Segmentation 클래스 수: **trimap 3-class**로 확정.
- Detection 클래스 정의: **이원(cat/dog)**으로 확정. 단, 파이프라인은 클래스 수와 무관하게 multi-class를 지원(6.1 규약)하므로 이 결정은 데이터 라벨링 선택일 뿐 구현 범위를 바꾸지 않는다.
- 실험 관리: **별도 실험관리 도구(tensorboard/wandb)는 사용하지 않는다.** 벤치마크 결과는 7.1의 config·metric 기록과 leaderboard 표(CSV/Markdown)로 관리한다.

---

*작성일: 2026-08-18 · 버전: v0.1 · 다음 단계: PLAN.md (구현 설계)*
