# PLAN — 교육용 Jupyter 노트북 v0.2

## 1. 문서 목적

이 문서는 v0.1에서 완성된 PyTorch CV boilerplate를 대상으로 하는 **교육용 Jupyter 노트북의
요구사항, 의도, 작업 계획**을 정리한다. v0.2의 단일 사양 문서이며, 이 문서의 절 번호가 적대적
검증과 `backlog.json`이 참조하는 조항 ID다.

- 범위: 노트북 요구사항(What / Why)과 작업 계획·검증 절차(15장)
- 전제: v0.1은 P1~P6 전 Phase 완료 상태이며(`docs/dev/v0.1/backlog.json`), v0.1의 코드와 문서는
  참조 전용이다.

### 1.1 문서 체인 축약

CLAUDE.md의 기본 문서 체인은 `BRIEF.md → PRD.md → PLAN.md → plans/PLAN-P{n}-*.md → backlog.json`
이지만, v0.2는 **`BRIEF.md`, `PRD.md`, Phase별 `plans/PLAN-P{n}-*.md`를 두지 않고
`PLAN.md → backlog.json`으로 축약한다.**

사유:

- v0.2는 공통 계약을 새로 정의하지 않는다. v0.1에서 확정·검증된 계약을 설명하는 작업이므로 요구사항
  발굴(BRIEF)과 요구사항 정제(PRD) 단계에서 새로 결정할 사항이 사실상 없다. 이 문서가 요구사항과
  구현 설계를 함께 담는다.
- 노트북은 태스크마다 독립적이고 공통 계약을 바꾸지 않으므로, Phase마다 별도 사양 문서를 두는 대신
  이 문서의 절 번호를 조항 ID로 사용한다.
- Phase 완료 상태는 CLAUDE.md 규정대로 `docs/dev/v0.2/backlog.json`에서만 관리한다.

v0.2 진행 중 공통 계약 변경이 필요해지면(예: 노트북 작성을 위해 `src/` 수정이 요구되는 경우) 이
축약을 유지하지 않고 사용자 승인 아래 필요한 설계 문서를 추가한다.

## 2. 프로젝트 목표

v0.1이 만든 것은 **동작하는 boilerplate**이지만, 그 안에서 무엇이 왜 그렇게 설계되었는지는
코드와 개발 문서를 함께 읽어야만 파악된다. v0.2는 이 간극을 메운다.

**v0.1의 코드를 그대로 import하여 실행하면서, 공통 엔진의 동작 원리와 12개 모델의 설계 의도를
단계적으로 확인할 수 있는 노트북 세트**를 만든다.

핵심 의도:

1. **읽고 실행하며 이해한다** — 설명 텍스트만 있는 문서가 아니라, 셀을 순서대로 실행하면 shape,
   배치 구조, 예측 결과가 눈앞에 출력되는 형태여야 한다.
2. **v0.1 코드를 재구현하지 않는다** — 노트북은 `src/`의 Dataset·Transform·Adapter·Model·Metric을
   import해서 호출한다. 노트북 안에 학습 로직을 다시 쓰면 노트북과 실제 엔진이 갈라져 교육 자료가
   오히려 오정보가 된다.
3. **엔진 이해와 모델 이해를 분리한다** — "공통 루프가 어떻게 태스크를 모른 채 동작하는가"와
   "이 태스크의 이 모델이 무엇을 하는가"는 다른 질문이므로 노트북 세트를 둘로 나눈다(3장).
4. **v0.1의 검증 결과를 재활용한다** — 이미 학습·평가된 산출물(`outputs/benchmarks/`)을 소비하여
   재학습 없이 실제 수치와 시각화를 보여준다(7장).

### 2.1 대상 독자

- Python과 PyTorch 기본 문법은 알고 있으나, 이 boilerplate의 구조(task-agnostic 엔진, TaskAdapter
  계약, Registry, Config 상속)를 처음 접하는 학습자.
- 각 태스크의 target 규약과 모델 출력 형태가 왜 그렇게 정의되었는지 확인하려는 사용자.
- 이 boilerplate에 새 모델이나 새 태스크를 추가하기 전에 계약을 파악해야 하는 개발자.

## 3. 산출물 범위

노트북 9개를 두 디렉토리로 나눈다.

```text
notebooks/
├── common/                     # 노트북 공용 헬퍼 모듈 (15.1)
├── toy/                        # 합성 toy 데이터. 데이터셋·체크포인트 불필요
│   ├── 00_engine.ipynb
│   ├── 01_toy_cls.ipynb
│   ├── 02_toy_seg.ipynb
│   ├── 03_toy_det.ipynb
│   └── 04_toy_anomaly.ipynb
└── tasks/                      # 실데이터. 로컬 데이터셋과 v0.1 체크포인트 필요
    ├── 01_classification.ipynb
    ├── 02_segmentation.ipynb
    ├── 03_detection.ipynb
    └── 04_anomaly.ipynb
```

| 노트북 | 답하는 질문 |
|---|---|
| `toy/00_engine.ipynb` | 공통 Trainer 루프는 태스크를 모른 채 어떻게 학습을 진행하는가 |
| `toy/01_toy_cls.ipynb` | 가장 단순한 태스크에서 Dataset → adapter → loss → metric이 어떻게 이어지는가 |
| `toy/02_toy_seg.ipynb` | 픽셀 단위 타깃이 배치와 metric에서 어떻게 다뤄지는가 |
| `toy/03_toy_det.ipynb` | 이미지당 객체 수가 가변(N=0/1/>1)인 타깃을 어떻게 표현하고 통과시키는가 |
| `toy/04_toy_anomaly.ipynb` | 타깃 없는 학습과 모델별 학습 방식 차이를 어댑터·훅만으로 어떻게 처리하는가 |
| `tasks/01_classification.ipynb` | oxford_pets 37 breeds에서 custom CNN과 pretrained 2종은 무엇이 다른가 |
| `tasks/02_segmentation.ipynb` | trimap 3-class에서 head 구조(ASPP 유무)가 결과를 어떻게 바꾸는가 |
| `tasks/03_detection.ipynb` | 1-stage와 2-stage 검출기의 출력·후처리·성능은 어떻게 다른가 |
| `tasks/04_anomaly.ipynb` | 정상 이미지만으로 학습하는 세 가지 방식(AE·distillation·teacher+AE)은 무엇이 다른가 |

### 3.1 디렉토리 분리 근거

두 세트는 **실행 전제와 학습 목표가 다르다**.

- `toy/`는 합성 데이터를 코드로 생성하므로 로컬 데이터셋과 체크포인트가 없어도 완주한다. 진입
  장벽이 없고, 실행이 빠르며, 관심 대상이 데이터가 아니라 **엔진과 계약**이다.
- `tasks/`는 `/mnt/d/datasets`의 실데이터와 v0.1이 생성한 체크포인트를 요구한다. 관심 대상은
  **실제 데이터의 성질과 모델의 동작·성능**이다.

디렉토리 이름을 버전(`v0.1`/`v0.2`)이 아니라 내용(`toy`/`tasks`)으로 정한 이유는, 두 세트 모두
v0.2 사이클에서 작성되고 둘 다 v0.1 엔진을 설명하기 때문이다. 버전 이름을 쓰면
`docs/dev/v{major}.{minor}/`의 문서 버전 체계와 의미가 충돌한다.

## 4. `notebooks/toy` 요구사항

### 4.1 `00_engine.ipynb`

공통 계층만 다루고 특정 태스크의 데이터 성질에는 들어가지 않는다.

- Config 로더 — YAML 상속, `--set` override, 검증 실패 동작, `config.resolved.yaml`의 의미
- Registry — 모델·데이터셋·transform·metric·adapter가 이름으로 등록·조회되는 방식
- TaskAdapter 계약 — 엔진이 어댑터에 무엇을 요구하고, 어댑터가 태스크 차이를 어디서 흡수하는가
- Trainer 루프 — epoch 진행, `model.train()`/`model.eval()` 경계, `no_grad()` 범위, metric 리셋
  시점, checkpoint 저장, monitor 기반 best 선택
- 비표준 학습 훅 — `on_fit_start` / `on_fit_end`가 왜 필요하고 어디서 호출되는가
- 엔진 순수성 — 공통 루프에 태스크 이름 분기가 없다는 사실을 코드로 확인

### 4.2 toy 태스크 노트북 4개

각 노트북은 해당 toy 태스크가 **어떤 계약을 대표하는지** 보여준다. 실데이터 노트북과 같은 흐름을
따르되 EDA는 두지 않고(합성 데이터이므로) 계약 확인과 shape 추적에 집중한다.

| 노트북 | 대표하는 계약 |
|---|---|
| `01_toy_cls` | 고정 형태 타깃(정수 라벨), 가장 단순한 stacked 배치 |
| `02_toy_seg` | 픽셀 단위 타깃 `(H, W) long`, 마스크에 보간·정규화를 적용하지 않는 이유 |
| `03_toy_det` | 가변 길이 타깃, `collate_fn`이 list를 유지하는 이유, N=0 배치의 통과 |
| `04_toy_anomaly` | 학습 시 빈 타깃, 모델별 `train_step` 분기, 훅을 통한 threshold·정규화 상수 산출 |

### 4.3 실행 전제

`notebooks/toy`의 5개 노트북은 **로컬 데이터셋과 체크포인트 없이 완주해야 한다.** toy 데이터는
노트북 안에서 합성하고, 모델은 소규모 toy 모델을 즉석에서 학습한다(수 epoch, CPU 실행 가능).

## 5. `notebooks/tasks` 공통 템플릿

4개 태스크 노트북은 동일한 흐름을 공유한다. 각 태스크 노트북 안에서 모델 3종은 하위 섹션으로
반복된다.

1. **태스크 개요** — 입력과 출력, 이 태스크의 target 규약, 사용 metric
2. **데이터 섹션** — EDA와 Dataset/DataLoader 출력 검증 (6장)
3. **어댑터 계약** — 이 태스크의 TaskAdapter가 배치를 모델 입력으로 바꾸고 출력을 metric 입력으로
   바꾸는 지점
4. **모델별 섹션 (custom → pretrained 1 → pretrained 2)**
   - 구조 요약과 설계 의도
   - forward pass 단계별 shape 추적
   - 중간 feature map 시각화
   - v0.1 체크포인트 로드 후 추론, 예측 결과 오버레이 (입력 / GT / 예측)
   - 학습 곡선 (`metrics_epoch.csv`)
5. **3모델 비교** — v0.1 leaderboard 표, 정확도 대 params/FLOPs/FPS 트레이드오프, 통제 조건이
   동일했음을 `config.resolved.yaml`로 확인

### 5.1 내용 깊이

**코드 동작 설명 중심**으로 작성한다. 데이터 흐름, shape 변화, forward pass, loss와 metric 계산
방식을 실제 코드와 함께 따라가며 "이 boilerplate가 어떻게 동작하는가"를 이해시키는 것이 목표다.
모델의 수학적 배경 이론(예: FCOS의 anchor-free 할당 유도, EfficientAD의 손실 항 도출)은 구조를
이해하는 데 필요한 최소한으로만 언급하고, 원 논문 참조로 대체한다.

## 6. 데이터 섹션 요구사항

### 6.1 EDA 범위

EDA는 **전체 데이터셋 기준**으로 수행한다. v0.1의 학습·평가는 축소 subset으로 진행되었으나,
데이터가 실제로 어떻게 생겼는지 보여주는 것이 EDA의 목적이므로 전장을 대상으로 통계를 낸다.
라벨 통계는 이미지 픽셀을 로드하지 않고 산출할 수 있어 실행 시간 부담이 작다.

공통 항목:

- split별 샘플 수와 상호 배타성 확인 결과
- 클래스·라벨 분포
- 원본 이미지 해상도 분포
- 전처리 전 raw 샘플 그리드

### 6.2 태스크별 EDA 항목

| 태스크 | 항목 |
|---|---|
| Classification | 37 breeds 클래스별 이미지 수 분포, cat/dog 상위 분류 비율 |
| Segmentation | trimap 3-class의 이미지 평균 픽셀 비율, 경계(class 2) 비중, 마스크와 이미지의 파일명 1:1 대응 |
| Detection | 박스 크기·종횡비 분포, **이미지당 객체 수 실측**(N=1 전제가 데이터의 성질일 뿐 파이프라인 제약이 아님을 확인), cat/dog 라벨 분포, `test.txt`에 XML 주석이 0건이라 자체 분할을 쓴 이유 |
| Anomaly | good 대 defect 이미지 수, 결함 종류별 분포, GT 마스크의 결함 면적 비율, 정상 이미지에 영행렬 마스크가 생성되는 것 |

### 6.3 Dataset / DataLoader 출력 검증

EDA가 "데이터가 어떻게 생겼는가"라면 이 절은 "코드가 무엇을 내보내는가"다. v0.1이 적대적 검증을
통해 확정한 계약들을 노트북에서 실측으로 재확인한다.

- `dataset[i]` 단일 샘플 — image tensor의 shape·dtype·값 범위, target의 정확한 구조를 target 공통
  규약과 대조
- transform 적용 전후 비교 시각화 — 이미지에는 보간·정규화가 적용되고 마스크에는 적용되지 않는
  것, Detection 박스가 리사이즈에 맞춰 함께 변환되는 것
- `collate_fn` 통과 후 배치 구조 — Cls/Seg는 stacked tensor, Detection은 가변 N의 list, Anomaly는
  학습 시 빈 dict
- `TaskAdapter`가 배치를 device로 옮기고 모델 입력 형태로 변환하는 지점까지 추적
- Detection 노트북은 N=0/1/>1이 혼재한 배치를 의도적으로 구성해 통과를 시연

확인 대상 계약의 예: Classification 라벨이 0-based(0..36), Segmentation trimap이 `{1,2,3}` →
`{0,1,2}` 매핑, Detection 박스가 절대 xyxy이고 라벨이 1-based(0은 배경 예약), Anomaly 마스크가
`{0,1}`.

### 6.4 EDA 중복 정책

`oxford_pets`를 Classification·Segmentation·Detection 3개 노트북이 공유하므로 이미지 관련 EDA
(해상도 분포, raw 샘플 그리드)가 3번 반복된다. 이는 **각 노트북의 단독 완결성을 위한 의도된
선택**이다. 학습자가 관심 있는 태스크의 노트북 하나만 열어도 데이터부터 결과까지 끊김 없이 따라갈
수 있어야 한다. 라벨 관련 EDA는 태스크마다 다르므로 실제 중복은 이미지 부분에 한정된다.

## 7. v0.1 결과물 재사용 정책

`notebooks/tasks`는 **재학습하지 않고** v0.1이 생성한 벤치마크 산출물을 소비한다.

### 7.1 소비 대상

체크포인트와 지표의 출처를 `outputs/benchmarks/{cls,seg,detection,anomaly}_baseline/`으로 고정한다.
이 디렉토리가 v0.1 leaderboard를 생성한 바로 그 실행 산출물이므로, 노트북이 보여주는 수치가 v0.1
문서의 수치와 정확히 일치한다.

```text
outputs/benchmarks/<task>_baseline/
├── leaderboard.csv / leaderboard.md    # 3모델 비교표와 통제 조건 덤프
├── control_report.json                 # 통제 필드 기계 검사 결과
└── splits/<model>/
    ├── checkpoints/best.pth            # 노트북 추론에 사용
    ├── config.resolved.yaml            # 통제 조건 제시에 사용
    ├── env.json
    ├── metrics_epoch.csv               # 학습 곡선에 사용
    ├── metrics_final.json              # 최종 지표 확인에 사용
    ├── train.log
    └── visualizations/                 # v0.1이 생성한 시각화. 노트북 결과와 대조
```

| 파일 | 노트북에서의 용도 |
|---|---|
| `best.pth` | 모델별 추론 섹션의 가중치 |
| `config.resolved.yaml` | 3모델이 동일 통제 조건으로 학습되었음을 제시 |
| `metrics_epoch.csv` | epoch별 학습 곡선 시각화 |
| `metrics_final.json` | 최종 지표 확인 |
| `leaderboard.csv` | 3모델 비교 섹션의 표 |
| `visualizations/` | 노트북에서 새로 만든 예측 시각화와의 대조 |

### 7.2 재학습 금지

노트북은 학습을 실행하지 않는다(`notebooks/toy`의 소규모 toy 학습은 예외). 재학습은 실행 시간을
크게 늘리고, 수치가 v0.1 leaderboard와 달라져 교육 자료로서의 일관성을 해친다. 학습 루프 자체는
`toy/00_engine.ipynb`가 toy 데이터로 이미 시연한다.

## 8. 대상 모델 12종과 교육적 비교 축

v0.1 leaderboard 실측값(5 epoch 축소 subset)을 각 태스크 비교 섹션의 근거로 사용한다.

| 태스크 | Custom | Pretrained 1 | Pretrained 2 | 비교 축 |
|---|---|---|---|---|
| Classification | `custom_cnn` (top1 0.081) | `resnet50` (0.797) | `efficientnet_b0` (0.757) | pretrained 효과, 정확도 대 params/FLOPs |
| Segmentation | `custom_unet` (mIoU 0.470) | `deeplabv3_resnet50` (0.794) | `fcn_resnet50` (0.793) | 동일 backbone에서 head 구조(ASPP 유무) 효과 |
| Detection | `custom_fcos` (mAP50-95 0.029) | `fasterrcnn_r50_fpn` (0.696) | `yolov8n` (0.410) | 1-stage 대 2-stage, 정확도 대 FPS |
| Anomaly | `custom_ae` (img 0.732 / px 0.259) | `stfpm` (0.162 / 0.857) | `efficientad` (0.539 / 0.449) | 학습 방식 차이, image-level 대 pixel-level 강점 |

### 8.1 낮은 수치를 다루는 방식

custom 모델과 일부 pretrained 모델의 수치는 낮다. v0.1은 **파이프라인 검증**이 목적이었고 5 epoch
축소 subset으로만 학습했기 때문이다. 노트북은 이 사실을 숨기지 않고 명시하며, 낮은 수치 자체를
교육 재료로 쓴다.

- from-scratch 모델이 짧은 학습에서 왜 불리한가 (`custom_cnn`, `custom_fcos`)
- STFPM의 image-AUROC가 pixel-AUROC보다 크게 낮은 것처럼, 지표 하나만 보면 모델을 오판할 수 있다
- 절대 성능이 목적이 아닐 때 벤치마크를 어떻게 읽어야 하는가

## 9. 전제 조건

- **실행 환경**: conda 환경 `pytorch_env`. 노트북 커널도 이 환경을 사용한다.
- **데이터셋**: `/mnt/d/datasets/oxford_pets`, `/mnt/d/datasets/mvtec` (자동 다운로드하지 않음)
- **백본 가중치**: `/mnt/d/backbones` (자동 다운로드하지 않음)
- **체크포인트**: `outputs/benchmarks/*_baseline/` (7.1)
- **오프라인 원칙 유지**: 노트북 실행 중 네트워크 접근이 발생하지 않아야 한다.

### 9.1 체크포인트 부재 시 동작

`outputs/`는 `.gitignore` 대상이므로 새 clone이나 다른 환경에서는 체크포인트가 없다. 이 경우
`notebooks/tasks`의 노트북은 **재생산 방법을 안내하고 명확하게 실패**한다. 필요한 체크포인트가
없다는 사실과 해당 벤치마크를 재생성하는 CLI 커맨드를 출력하는 셀을 노트북 앞부분에 둔다. 조용히
미학습 가중치로 진행하여 잘못된 결과를 보여주지 않는다.

`notebooks/toy`는 이 전제에 해당하지 않으며 어떤 환경에서도 완주해야 한다(4.3).

## 10. 제약 사항

- **v0.1 코드 수정 금지** — `src/`, `configs/`, `scripts/`는 읽기·import만 한다. 노트북을 위해
  v0.1 코드를 바꿔야 하는 상황이 생기면 변경 요청으로 보고하고 승인을 받는다.
- **v0.1 문서 수정 금지** — `docs/dev/v0.1/`은 참조 전용이다.
- **학습 로직 재구현 금지** — 노트북 안에 Trainer 루프나 Dataset을 다시 작성하지 않고 `src/`를
  import한다. 설명을 위해 코드 일부를 셀에 인용하는 것은 허용하되, 인용임을 명시한다.
- **실험 관리 도구 미도입** — tensorboard, wandb를 사용하지 않는다.
- **이모지 사용 금지.**
- **언어** — 노트북의 markdown 셀 설명은 한국어, 코드 셀의 주석은 영어로 작성한다. 코드, 명령어,
  파일 경로, 라이브러리 고유 이름은 원문 표기를 유지한다.
- **경로 표기** — `os.path` 방식을 사용하며 `pathlib.Path`를 사용하지 않는다.

## 11. 기술 스택 추가분

노트북 작성에 필요할 수 있는 신규 의존성 후보다. CLAUDE.md 규칙에 따라 **추가 전 사용자 승인이
필요하며**, 기존 스택으로 대체 가능한지 먼저 확인한다.

| 용도 | 후보 | 비고 |
|---|---|---|
| 노트북 실행 | `jupyter`, `ipykernel` | 필수. 완주 검증(15.3)의 `nbconvert`도 여기 포함된다 |
| 시각화 | `matplotlib` | 필수. 예측 오버레이, feature map, 분포 히스토그램, 학습 곡선 |
| 모델 구조 요약 | `torchinfo` | 선택. `print(model)`과 직접 순회로 대체 가능한지 검토 |
| 표 출력 | `pandas` | 선택. leaderboard CSV 읽기와 EDA 통계 집계 |

`PLAN.md` 단계에서 최소 집합으로 확정하고 `requirements.txt` 반영 여부를 결정한다.

## 12. 성공 기준 (v0.2)

1. 9개 노트북이 `pytorch_env` 커널에서 처음부터 끝까지 오류 없이 완주한다(판정 방법은 15.3).
2. `notebooks/toy`의 5개 노트북이 로컬 데이터셋과 체크포인트 없이 완주한다.
3. `notebooks/tasks`의 4개 노트북이 12모델 전부의 예측 시각화와 중간 feature map을 생성한다.
4. 각 태스크 노트북의 비교 섹션 수치가 v0.1 leaderboard와 일치한다.
5. EDA가 산출한 split별 샘플 수가 `configs/splits/` 정의와 일치한다.
6. 노트북이 `src/` 아래 파일을 수정하지 않고, 학습 로직을 재구현하지 않는다.
7. 체크포인트가 없는 환경에서 `notebooks/tasks`가 재생산 안내와 함께 명확히 실패한다.
8. 노트북을 대상으로 한 반대 벤더 적대적 검증에 미해결 Critical이 남지 않는다(15.4).

## 13. 비범위

- 신규 모델·태스크·데이터셋 추가
- 재학습, 하이퍼파라미터 튜닝, 성능 개선
- v0.1의 미해결 항목 수정 — ISS-01(v0.1 BRIEF 문구 정정), ISS-06(`Trainer.fit()` resume 분기 재설계)
- **문서 자체에 대한 적대적 검증** — 이 문서(`PLAN.md`)와 `backlog.json`은 반대 벤더 검토 대상이
  아니다. 적대적 검증은 노트북이 생성된 뒤 그 산출물을 대상으로만 수행한다(15.4)
- 노트북의 HTML·웹 문서화 및 배포
- v0.1 코드 리팩토링

## 14. 결정 사항

2026-08-19 사용자 문답으로 확정한 항목이다.

| 항목 | 결정 |
|---|---|
| 산출물 | 노트북 9개 — `toy/` 5개(엔진 1 + toy 태스크 4), `tasks/` 4개 |
| 디렉토리 | 프로젝트 루트 `notebooks/`, 내용 기준 이름(`toy`/`tasks`)으로 분리 |
| 태스크 노트북 구성 | 태스크당 1개. 공통 섹션 + 모델별 하위 섹션 3개 + 비교 섹션 |
| 가중치 | `outputs/benchmarks/*_baseline/splits/<model>/checkpoints/best.pth` 재사용, 재학습 없음 |
| 체크포인트 부재 시 | 재생산 커맨드 안내 후 명확히 실패 |
| 내용 깊이 | 코드 동작 설명 중심, 수학적 배경 이론은 최소화 |
| 시각화 | 예측 결과 오버레이 + 중간 feature map 포함 |
| EDA 범위 | 전체 데이터셋 기준 |
| EDA 중복 | oxford_pets 이미지 EDA 3중복은 단독 완결성을 위해 허용 |
| 언어 | markdown 셀 한국어, 코드 주석 영어 |
| 적대적 검증 | 노트북 생성 후 그 산출물을 대상으로 Codex(`gpt-5.6-sol`) 1회 (15.4). 문서는 대상 아님 |
| 문서 체인 | `BRIEF.md`·`PRD.md`·Phase별 PLAN 생략, `PLAN.md → backlog.json`으로 축약 (1.1) |
| 공유 헬퍼 | `notebooks/common/` 모듈로 분리 (15.1) |
| 완주 검증 | `jupyter nbconvert --execute` 헤드리스 실행 (15.3) |

## 15. 작업 계획

`BRIEF.md`·`PRD.md`를 생략하므로(1.1) 작업 계획과 검증 절차를 여기에 확정한다.

### 15.1 공유 헬퍼 모듈

9개 노트북이 반복해서 쓰는 시각화·집계 코드는 `notebooks/common/`에 모듈로 분리하고 각 노트북이
import한다. 노트북마다 복붙하면 수정 시 9곳을 고쳐야 하고, 설명해야 할 본론(데이터와 모델)보다
matplotlib 배관 코드가 지면을 더 차지한다.

분리 대상:

- feature map 추출(forward hook 등록·해제)과 그리드 시각화
- 태스크별 예측 오버레이 — Cls 라벨 표기, Seg 마스크 합성, Det 박스 렌더링, Anomaly heatmap
- `metrics_epoch.csv` 기반 학습 곡선
- `leaderboard.csv` 표 출력
- EDA 히스토그램·분포 플롯
- 체크포인트 존재 확인과 부재 시 안내 출력(9.1)

노트북에 남기는 것:

- `src/`의 Dataset·Transform·Adapter·Model·Metric 호출은 **모두 노트북 셀에 노출한다.** 이것이
  학습 대상이므로 헬퍼로 감싸 숨기지 않는다.
- shape·dtype·값 범위 출력처럼 한두 줄인 확인 코드

`notebooks/common/`은 시각화와 집계 유틸리티만 담는다. 학습·평가 로직이나 `src/`를 대신하는 래퍼를
두지 않는다(10장).

### 15.2 작업 순서

`backlog.json`의 Phase로 관리하며 순차 진행한다.

| Phase | 내용 | 적대적 검증 |
|---|---|---|
| N1 | `notebooks/common/` 헬퍼 모듈, 의존성 확정과 승인, 완주 검증 절차 | 없음 (노트북 산출물 없음) |
| N2 | `toy/00_engine.ipynb` | 없음 |
| N3 | `toy/01_toy_cls` ~ `toy/04_toy_anomaly` | 1회 (N2 결과 포함) |
| N4 | `tasks/01_classification.ipynb` | 1회 |
| N5 | `tasks/02_segmentation.ipynb`, `tasks/03_detection.ipynb`, `tasks/04_anomaly.ipynb` | 1회 |
| N6 | 전체 완주 검증, 성공 기준(12장) 대조, README 갱신 | 1회 (전체 노트북) |

N4를 단독 Phase로 두는 이유는 실데이터 노트북의 템플릿(5장)이 여기서 처음 실체화되기 때문이다.
N4에서 템플릿을 확정한 뒤 N5의 3개 노트북에 적용한다. 각 Phase 완료 시 변경 내용과 검증 결과를
보고하고 커밋 승인을 요청한다.

### 15.3 완주 검증

성공 기준 1번은 `jupyter nbconvert --execute`로 헤드리스 실행하여 기계적으로 판정한다.

- 각 노트북을 `pytorch_env`에서 처음부터 끝까지 실행하고, 예외 발생 시 실패로 간주한다.
- 판정은 Phase 완료 시점마다 해당 노트북에 대해 수행하고, N6에서 9개 전체를 다시 실행한다.
- 실행 산출물(변환 결과물)은 커밋하지 않는다. 노트북 파일 자체의 출력 셀 저장 여부는 N1에서
  결정한다.
- `notebooks/toy` 5개는 데이터셋·체크포인트가 없는 조건에서도 완주해야 하므로, 해당 경로를 참조할 수
  없는 상태를 가정한 실행으로 별도 확인한다.

### 15.4 적대적 검증

적대적 검증은 **노트북이 생성된 뒤 그 산출물을 대상으로만** 수행한다. 이 문서와 `backlog.json`은
검증 대상이 아니다(13장).

- **시점** — N3, N4, N5, N6의 각 Phase에서 해당 노트북 작성과 완주 검증(15.3)을 마친 직후.
  N1과 N2는 노트북 산출물이 없거나 단독 노트북 1개뿐이므로 N3의 검증에 포함해 함께 검토한다.
- **횟수** — 각 Phase당 **1회**로 제한한다. CLAUDE.md의 Verification Attempt Limit이 정한 3회를
  v0.2에서는 1회로 축약한다. 노트북은 공통 계약을 바꾸지 않고 검토 대상이 설명 정확성과 실행
  무결성에 한정되므로 반복 검토의 한계 효용이 낮다.
- **검토자** — 구현자가 Claude Code이므로 반대 벤더인 Codex CLI를 사용한다.

  ```bash
  codex exec --model gpt-5.6-sol --sandbox read-only \
    --cd "/mnt/d/projects/nampluskr/00_review/260818_cv-boilerplate"
  ```

  모델 접근이 거부되면 기본 모델로 조용히 폴백하지 않고 오류와 대체안을 사용자에게 보고한다.
  시간 제한은 10분으로 둔다.
- **검토 범위** — 대상 노트북 파일과 `notebooks/common/`, 그리고 노트북이 참조하는 `src/` 코드.
  검토자는 파일을 수정하지 않는다.
- **공격 축** — 노트북 성격에 맞춰 다음에 집중한다.
  1. **설명 정확성** — markdown 셀의 서술이 실제 코드 동작과 일치하는가. 잘못된 설명은 코드 버그보다
     교육 자료에서 더 해롭다.
  2. **계약 확인의 실효성** — 6.3절이 확인하겠다고 한 계약(0-based 라벨, trimap 매핑, 절대 xyxy,
     `{0,1}` 마스크, N=0 통과)을 노트북이 실제로 검증하는가, 아니면 통과를 가정하고 넘어가는가.
  3. **학습/평가 무결성 왜곡** — 노트북이 `model.eval()`·`no_grad()`를 빠뜨려 v0.1이 지킨 규율과
     다른 결과를 보여주지 않는가. test split을 부적절하게 노출하지 않는가.
  4. **v0.1 코드 우회** — 노트북이 `src/`를 import하는 대신 로직을 재구현해 실제 엔진과 다른 것을
     설명하고 있지 않은가(10장).
  5. **오프라인 위반** — 노트북 실행 경로에 네트워크 접근이 남아 있는가.
  6. **수치 일관성** — 노트북이 제시하는 지표가 v0.1 leaderboard와 일치하는가.
- **처리** — Critical 지적은 모두 수정하고 해당 노트북의 완주 검증을 재실행한다. Major와 Minor는
  처리 여부와 근거를 기록한다. 1회 제한이므로 수정 후 재검토는 하지 않고, 수정 내용과 남은 위험을
  기록한다.
- **기록** — `docs/dev/v0.2/reviews/N{n}.md`에 검토 모델, 대상 파일, 실행 일시, 심각도별 건수,
  지적·확인 조건·관련 PLAN 조항·처리 상태를 남긴다. 유효하지 않은 지적의 반박 근거도 기록한다.
- **진행 조건** — 미해결 Critical이 있으면 다음 Phase로 진행하지 않는다.

---

*작성일: 2026-08-19 · 버전: v0.2 · 다음 단계: backlog.json의 N1 착수*
