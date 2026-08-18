# PRD — PyTorch CV Boilerplate v0.1

## 1. 문서 목적

이 문서는 `BRIEF.md`에서 정의한 요구사항과 의도를 **검증 가능한 요구사항 목록**으로 구체화한다.
각 요구사항은 고유 ID를 가지며, 이후 `PLAN.md`(Phase 전체 정의)와 `plans/PLAN-P{n}-*.md`(Phase 상세 설계), `backlog.json`(실행 단위)에서
ID로 역참조한다.

- 범위: 무엇을 만들 것인가(요구사항) + 무엇을 만들지 않을 것인가(비구현 대상)
- 비범위: 모듈 경계·클래스 시그니처·파일 단위 설계 → `plans/PLAN-P{n}-*.md`
- 상위 문서: `BRIEF.md` (배경·의도·결정 사항)

ID 체계는 다음과 같다.

| 접두사 | 의미 |
|---|---|
| `FR-` | 기능 요구사항 (Functional Requirement) |
| `NFR-` | 비기능 요구사항 (Non-Functional Requirement) |
| `CON-` | 제약 사항 (Constraint) |
| `OUT-` | 비구현 대상 (Out of Scope) |
| `AC-` | 수용 기준 (Acceptance Criterion) |

우선순위 표기는 `MUST`(v0.1 완료 조건), `SHOULD`(가능하면 v0.1, 미달 시 문서화), `MAY`(선택)로 구분한다.

근거 표기는 `BRIEF n`(BRIEF.md 절 번호), `AGENTS`(AGENTS.md 규칙), `PRD`(상위 문서에 없고 이 문서에서
구체화한 항목)로 구분한다. `PRD` 표기 항목은 상위 문서의 확정 사항이 아니므로 필요 시 조정할 수 있다.

## 2. 제품 개요

이 제품은 하나의 task-agnostic 학습 엔진 위에서 Classification, Segmentation, Object Detection,
Anomaly Detection 4개 태스크를 동일한 CLI/Config 흐름으로 다루는 **재사용 가능한 벤치마크 boilerplate**다.

핵심 사용 시나리오는 다음 세 가지다.

| 시나리오 | 사용자 행위 | 기대 결과 |
|---|---|---|
| 단일 실험 | 하나의 config로 학습·평가·추론을 실행한다 | 체크포인트, metric 기록, 예측 시각화 |
| 벤치마크 비교 | 한 태스크 안에서 모델·학습조건 split 여러 개를 순차로 학습·평가·추론한다 | split별 산출물과 이를 집계한 leaderboard 표 |
| 모델 확장 | 새 모델을 registry에 등록하고 config만 추가한다 | 엔진·태스크 코드 수정 없이 동일 흐름으로 비교 가능 |

### 2.1. 확정 비교 모델 구성

각 태스크의 비교 대상은 `BRIEF.md` 4장에서 확정되었다. 이 구성은 `FR-18`의 대상이며 임의 교체할 수 없다.

| 태스크 | Custom | Pretrained 1 | Pretrained 2 | 비교 축 |
|---|---|---|---|---|
| Classification | Custom CNN | ResNet50 (torchvision) | EfficientNet-B0 (torchvision) | 정확도 대 파라미터/효율 |
| Segmentation | Custom U-Net류 | DeepLabV3-ResNet50 (torchvision) | FCN-ResNet50 (torchvision) | head 구조(ASPP 유무) 효과 |
| Detection | Custom 1-stage | YOLO (ultralytics, v8/v11) | Faster R-CNN ResNet50-FPN (torchvision) | 1-stage 대 2-stage |
| Anomaly Detection | Custom Reconstruction/AE류 | STFPM (anomalib 모델 정의) | EfficientAD (anomalib 모델 정의) | 속도 대 정확도 |

Segmentation의 torchvision 2종은 backbone(ResNet50)을 통일해 head 차이만 비교한다.
경량축이 필요하면 FCN 대신 LRASPP-MobileNetV3로, Detection 1-stage 통일이 필요하면 Faster R-CNN 대신
RetinaNet/FCOS로 교체할 수 있으나, 교체는 `BRIEF.md` 4장 비고에 근거한 **사용자 승인 후**에만 허용한다.

## 3. 기능 요구사항

기능 요구사항은 공통 엔진, 태스크 파이프라인, 벤치마크, CLI의 네 영역으로 나눈다.

### 3.1. 공통 엔진 및 인프라

엔진은 태스크를 알지 못한 채 학습·평가 루프를 수행해야 한다.

| ID | 요구사항 | 상세 | 우선순위 | 근거 |
|---|---|---|---|---|
| FR-01 | Config 시스템 | YAML 파일로 데이터·모델·손실·optimizer·scheduler·학습조건을 선언한다 | MUST | BRIEF 6, 8 |
| FR-02 | task-agnostic 엔진 | Trainer/Engine 루프는 태스크 이름으로 분기하지 않으며, 4개 태스크에서 수정 없이 동작한다 | MUST | BRIEF 6 |
| FR-03 | Task 어댑터 | "한 배치를 어떻게 forward하고 loss/preds를 뽑는가"를 어댑터가 캡슐화하여 엔진과 태스크를 분리한다 | MUST | BRIEF 6 |
| FR-04 | Registry | dataset·model·loss·metric·transform을 문자열 키로 등록·조회하여 config만으로 조합한다 | MUST | BRIEF 6 |
| FR-05 | Checkpoint | best/last 체크포인트를 저장하고, 저장 시 config와 epoch·metric을 함께 기록하며 재개 로드를 지원한다 | MUST | BRIEF 6 |
| FR-06 | Logger | epoch별 train/valid loss와 metric을 콘솔 및 파일(로그·CSV)에 기록한다 | MUST | BRIEF 6 |
| FR-07 | Optimizer/Scheduler 빌더 | config 선언만으로 optimizer와 scheduler를 생성한다 | MUST | BRIEF 6 |
| FR-08 | Device/AMP | CPU·CUDA 선택과 AMP(mixed precision) on/off를 config로 제어한다 | SHOULD | BRIEF 6 |
| FR-09 | 비표준 학습 훅 | 표준 gradient 루프를 벗어나는 모델(memory-bank류 등)을 별도 훅 또는 Trainer로 처리하고 그 한계선을 문서화한다 | SHOULD | BRIEF 6 |
| FR-10 | 모델별 train_step 분기 | STFPM/EfficientAD처럼 학습 방식이 다른 경우 어댑터 또는 모델 수준에서 `train_step`을 분기한다(엔진 분기 금지) | MUST | BRIEF 4 |
| FR-34 | CLI override | config 개별 항목을 CLI 인자로 override한다 | SHOULD | PRD |

### 3.2. 태스크 파이프라인

각 태스크는 Dataset·모델·Loss·Metric·Transform·Postprocess를 제공하며, target 형태는 공통 규약을 따른다.

| ID | 요구사항 | 상세 | 우선순위 | 근거 |
|---|---|---|---|---|
| FR-11 | target 규약 준수 | Dataset이 반환하는 target은 태스크별 공통 규약(BRIEF 6.1)을 따르며, 모델·Loss·Metric은 규약에만 의존한다 | MUST | BRIEF 6.1 |
| FR-12 | Classification 파이프라인 | `oxford_pets` 37 breeds 다중분류를 학습·평가한다 | MUST | BRIEF 5, 10 |
| FR-13 | Segmentation 파이프라인 | `oxford_pets` trimap 3-class 픽셀 분류를 학습·평가한다 | MUST | BRIEF 5, 10 |
| FR-14 | Detection 파이프라인 | Pascal VOC XML을 파싱하여 multi-class / multi-object를 일반적으로 지원한다. 이미지당 가변 개수 N의 박스를 처리하고 N=0을 허용한다 | MUST | BRIEF 5.2, 6.1 |
| FR-15 | 가변 길이 collate | Detection 등 가변 개수 target은 태스크별 `collate_fn`(list-of-dict 등)으로 배치를 구성한다. 한 배치 안에 N=0, N=1, N>1 샘플이 섞여도 Loss·Metric·Postprocess까지 완주한다 | MUST | BRIEF 6.1 |
| FR-16 | Anomaly 파이프라인 | 정상 이미지만 학습하고 평가 시 image-level 이상 점수를 산출한다. GT mask가 있는 데이터셋에서는 pixel-level 이상 맵도 산출한다. MVTec `bottle`은 mask를 제공하므로 pixel-level 산출이 필수다 | MUST | BRIEF 5, 6.1, 7 |
| FR-17 | 데이터 split | train/valid/test split을 재현 가능한 방식(고정 seed 또는 명시적 목록)으로 생성한다. 세 split의 샘플 ID 집합은 상호 배타적이어야 하며 동일 원본 이미지가 둘 이상의 split에 포함되지 않는다 | MUST | BRIEF 6, 7.1, AGENTS |
| FR-18 | 태스크별 확정 모델 3종 | 각 태스크에서 2.1의 확정 모델 매트릭스(custom 1 + pretrained 2)를 구현하고 동일 조건으로 학습·비교한다. 매트릭스 변경은 사용자 승인 후에만 허용한다 | MUST | BRIEF 4, 9 |
| FR-19 | Loss/Metric | Cls: Top-1/F1, Seg: mIoU/Dice, Det: mAP@0.5:0.95, Anomaly: image-AUROC 및 (GT mask 보유 시) pixel-AUROC를 산출한다 | MUST | BRIEF 7 |
| FR-20 | Postprocess | Detection의 NMS·score threshold, Anomaly의 score threshold 등 태스크별 후처리를 파이프라인에 포함한다 | MUST | BRIEF 6 |
| FR-21 | Transform | `torchvision.transforms.v2`로 이미지와 라벨(mask/box)을 동시 변환한다 | MUST | BRIEF 8 |
| FR-22 | Visualization | 태스크별 예측 결과(클래스·mask·box·anomaly map)를 이미지 파일로 저장한다 | SHOULD | BRIEF 6 |

### 3.3. 벤치마크 및 리포트

벤치마크는 한 태스크 내부에서만 수행하며, 여러 split의 결과를 하나의 표로 비교한다.

| ID | 요구사항 | 상세 | 우선순위 | 근거 |
|---|---|---|---|---|
| FR-23 | split 정의 | 태스크를 고정한 상태에서 모델과 학습조건(해상도·optimizer·scheduler·epoch·augmentation 등)의 조합을 하나의 split으로 선언한다 | MUST | BRIEF 7.1 |
| FR-24 | 순차 실행 | 선언된 여러 split을 한 번의 실행으로 순차 학습·평가·추론한다. 각 split은 학습 후 평가 metric과 추론 산출물을 남긴다 | MUST | BRIEF 2, 7.1 |
| FR-25 | leaderboard 생성 | 같은 태스크 내 split 결과를 CSV와 Markdown 표로 집계한다. 표에는 통제 조건과 승인된 예외를 함께 기록한다 | MUST | BRIEF 7.1, 8 |
| FR-26 | 효율 지표 | 표준 metric과 함께 params / FLOPs / inference FPS를 기록한다 | MUST | BRIEF 7 |
| FR-27 | 실행 기록 | split별 config·seed·metric·환경 정보를 산출물 경로에 저장하여 재현 가능하게 한다 | MUST | BRIEF 7.1 |
| FR-28 | 최소 split 세트 | 각 태스크에서 2.1 매트릭스의 3개 모델을 동일 학습조건으로 실행하는 기본 split을 제공한다 | MUST | BRIEF 7.1 |
| FR-35 | 통제 조건 검사 | 비교 그룹 split의 config에서 통제 대상 필드를 기계적으로 비교하고, 승인된 예외 외의 차이가 있으면 실행 또는 leaderboard 생성을 실패시킨다 | MUST | BRIEF 7, AGENTS |

### 3.4. CLI 및 실행

CLI는 단일 진입점을 통해 config 확인·학습·평가·추론을 제공한다.

| ID | 요구사항 | 상세 | 우선순위 | 근거 |
|---|---|---|---|---|
| FR-29 | 단일 CLI 진입점 | config 확인, 학습, 평가, 추론을 하나의 CLI 체계로 실행한다 | MUST | BRIEF 6 |
| FR-30 | 학습 실행 | config 경로를 인자로 받아 학습을 수행하고 체크포인트·로그를 산출한다 | MUST | BRIEF 6 |
| FR-31 | 평가 실행 | 체크포인트와 config로 test split을 평가하여 metric을 출력·저장한다 | MUST | BRIEF 6 |
| FR-32 | 추론 실행 | 체크포인트로 임의 이미지에 대해 예측을 수행하고 결과를 저장한다 | MUST | BRIEF 6 |
| FR-33 | config 검증 | 실행 전 config의 필수 키·경로 존재·registry 키 유효성을 확인하고 오류를 명확히 보고한다 | SHOULD | BRIEF 6 |

## 4. 비기능 요구사항

비기능 요구사항은 품질·재현성·확장성 기준을 정의한다.

| ID | 항목 | 요구사항 | 판정 기준 | 우선순위 | 근거 |
|---|---|---|---|---|---|
| NFR-01 | 재현성 | seed 고정 시 동일 config가 동일 결과를 산출한다. seed 적용 범위는 Python `random`, NumPy, PyTorch CPU/CUDA, DataLoader worker, sampler를 포함한다 | 동일 환경·동일 seed 2회 실행에서 각 metric이 `PLAN-P1`에 정의된 허용 오차 내로 일치. 허용 오차와 deterministic 설정 범위는 `plans/PLAN-P1-foundation.md`에서 수치로 확정한다 | MUST | BRIEF 7.1, AGENTS |
| NFR-02 | 공정 비교 통제 | 한 태스크 내 비교 split은 입력 해상도·정규화·augmentation·optimizer·scheduler·epoch·seed를 고정한다. batch만 모델 크기로 조정 시 명시한다 | `FR-35`의 기계적 검사가 통과하고, leaderboard에 통제 조건과 승인된 예외가 기록됨 | MUST | BRIEF 7, 7.1, AGENTS |
| NFR-03 | 평가 무결성 | 평가·추론 경로에서 `model.eval()`과 `torch.no_grad()`를 적용하고 metric을 매 평가마다 리셋한다. 모델 선택과 threshold 결정은 valid split만 사용하고 test split은 최종 평가 전용으로 유지한다 | 코드 검토에서 학습/평가 누수가 확인되지 않음 | MUST | AGENTS |
| NFR-04 | 엔진 일반성 | 태스크 추가 시 엔진 코드를 수정하지 않는다 | 4개 태스크가 동일 엔진에서 동작, 예외는 FR-09로 문서화 | MUST | BRIEF 6 |
| NFR-05 | 모델 확장성 | 새 모델 추가는 registry 등록과 config 추가만으로 가능하다 | 엔진·태스크 파이프라인 파일 수정 없이 신규 모델 split 실행 성공 | MUST | BRIEF 2 |
| NFR-06 | 데이터셋 교체성 | 새 데이터셋 추가는 target 규약을 따르는 Dataset/파서 어댑터 구현과 registry 등록, config 추가로 가능하며 엔진·모델·Loss·Metric은 수정하지 않는다. Detection은 데이터만 교체(oxford_pets → VOC/COCO)해도 파이프라인이 동작해야 한다 | 규약 준수 어댑터를 추가한 상태에서 엔진·Loss·Metric 무수정으로 실행 성공(합성 fixture로 검증 가능) | MUST | BRIEF 5.2, 6.1 |
| NFR-07 | 오프라인 동작 | 실행 중 네트워크 접근이 없어도 학습·평가·추론이 완료된다 | 네트워크 차단 상태에서 스모크 실행 성공 | MUST | BRIEF 5, 8.1 |
| NFR-08 | 가독성·일관성 | PEP8 네이밍을 따르고 멤버 변수 접두사와 세로 정렬을 사용하지 않으며, split은 `valid`, 백본은 `backbone_name`으로 표기한다 | 코드 검토 통과 | MUST | AGENTS |
| NFR-09 | 검증 가능성 | 각 Phase는 실제 학습·평가 스모크 실행으로 검증한다 | 소규모 subset·소 epoch 스모크가 오류 없이 완주 | MUST | AGENTS |
| NFR-10 | 산출물 구조 | 실행 산출물(체크포인트·로그·metric·시각화)은 실행 단위로 분리된 경로에 저장된다 | split별 산출물이 서로 덮어쓰지 않음 | MUST | BRIEF 7.1 |
| NFR-11 | 오류 보고 | 데이터 경로·가중치 파일 부재 등 환경 오류는 원인과 조치를 포함한 메시지로 즉시 실패한다 | 경로 누락 시 stack trace가 아닌 명시적 메시지 | SHOULD | PRD |
| NFR-12 | 스모크 실행 규격 | 태스크별 스모크 설정(샘플 수·해상도·batch·최대 실행 시간)을 `plans/PLAN-P1-foundation.md`에 수치로 정의하고 그 범위 안에서 완주한다. epoch은 전 Phase 공통 5로 고정한다 | 정의된 스모크 설정으로 실행 시 제한 시간 내 완주 | SHOULD | AGENTS |

## 5. 제약 사항

제약 사항은 설계·구현 선택을 구속하는 조건이다. 위반 시 해당 구현은 수용되지 않는다.

| ID | 구분 | 제약 | 근거 |
|---|---|---|---|
| CON-01 | 프레임워크 | 학습·평가·추론 로직은 pure-PyTorch로 작성한다. Lightning 등 상위 학습 프레임워크를 도입하지 않는다 | BRIEF 4, 8 |
| CON-02 | 외부 라이브러리 | Anomaly는 외부 라이브러리 임포트를 최소화하고 PyTorch 수준에서 직접 작성한다. anomalib는 모델 정의만 차용한다 | BRIEF 4 |
| CON-03 | 오프라인 | 데이터셋과 백본 가중치를 자동 다운로드하지 않는다. `/mnt/d/datasets`와 `/mnt/d/backbones`의 로컬 경로만 참조한다 | BRIEF 5, 8.1 |
| CON-04 | torchvision pretrained 로드 | torchvision pretrained는 `weights=None`으로 아키텍처를 만든 뒤 로컬 `.pth`를 `load_state_dict`로 주입한다 | BRIEF 8.1 |
| CON-05 | 비 torchvision 가중치 | ultralytics YOLO와 anomalib 기반 모델(STFPM/EfficientAD)도 자동 다운로드 없이 로컬 가중치 파일만 사용한다. 로컬 가중치가 없으면 명시적 오류로 실패한다 | BRIEF 4, 8, 8.1 |
| CON-06 | 실행 환경 | 대상 환경은 WSL2(Linux), 셸은 bash이며 Python 실행·검증 전 conda 환경 `pytorch_env`를 활성화한다 | AGENTS, BRIEF 8.1 |
| CON-07 | 경로 표기 | 경로 처리는 `os.path`를 사용하고 `pathlib.Path`를 사용하지 않는다 | AGENTS |
| CON-08 | 의존성 추가 | 새 의존성 추가 전 기존 스택(torch, torchvision, torchmetrics)으로 구현 가능한지 확인하고 사용자 승인을 받는다 | AGENTS |
| CON-09 | 데이터셋 고정 | Cls/Seg/Det은 `oxford_pets`, Anomaly는 `mvtec`을 사용한다. Anomaly 초기 범위는 `bottle` 1개 카테고리다 | BRIEF 5, 5.2 |
| CON-10 | 라벨 정의 | Classification은 37 breeds, Segmentation은 trimap 3-class, Detection은 cat/dog 이원 라벨로 확정한다 | BRIEF 10 |
| CON-11 | Detection 일반성 | `oxford_pets`가 단일 객체라는 이유로 파이프라인을 단일 객체 전용으로 축약하지 않는다 | BRIEF 5.2 |
| CON-12 | 원점 재설계 | 기존 `260712_roi-corner-detection-ver2`의 CLI 사용 패턴만 참고하며, 코드·폴더구조·파일을 재사용하거나 확장하지 않는다 | BRIEF 6 |
| CON-13 | 추상화 수준 | `BRIEF.md`에 확정된 범위·데이터셋·비교 모델·기술 스택을 임의로 변경하지 않고 불필요한 추상화 계층을 추가하지 않는다 | AGENTS |
| CON-14 | 문서·표기 | 이모지를 사용하지 않는다. 코드 주석은 영어, Markdown 문서는 한국어로 작성한다 | AGENTS |
| CON-15 | 커밋 범위 | 데이터셋·백본 가중치·체크포인트·실험 산출물은 커밋하지 않는다 | AGENTS |
| CON-16 | 문서 체인 | 개발 문서는 `docs/dev/v{major}.{minor}/`에 두고 `BRIEF.md → PRD.md → PLAN.md → plans/PLAN-P{n}-*.md → backlog.json` 순서를 유지한다. v0.1은 별도 `SPEC.md`를 두지 않는다 | AGENTS |

## 6. 비구현 대상

다음 항목은 v0.1 범위에서 제외한다. 제외는 "영구 배제"가 아니라 "이번 버전에서 구현하지 않음"을 의미한다.

| ID | 비구현 대상 | 제외 사유 | 향후 처리 |
|---|---|---|---|
| OUT-01 | 태스크 간 성능 비교 | 벤치마크는 한 태스크 내부 비교로 한정한다 | 영구 비범위 |
| OUT-02 | 실험 관리 도구(tensorboard, wandb) | config·metric 기록과 leaderboard 표로 충분하다 | 필요 시 별도 검토 |
| OUT-03 | 자동 데이터셋·가중치 다운로드 | 오프라인 원칙(CON-03, CON-05) | 영구 비범위 |
| OUT-04 | Lightning 기반 학습 루프 | pure-PyTorch 원칙(CON-01) | 영구 비범위 |
| OUT-05 | anomalib 학습 프레임워크 사용 | 모델 정의만 차용(CON-02) | 영구 비범위 |
| OUT-06 | MVTec 전체 15개 카테고리 학습 | 초기에는 `bottle`로 파이프라인 검증 | v0.2 이후 확장 |
| OUT-07 | Pascal VOC 2007 / COCO128 실제 데이터 연동 | 로컬 미보유. 파이프라인 교체 가능성은 NFR-06으로 보장 | 데이터 확보 시 추가 |
| OUT-08 | `visa`, `btad`, `imagenette2`, `cifar10` 등 대안 데이터셋 | v0.1 비범위(BRIEF 5.3) | v0.2 이후 확장 |
| OUT-09 | timm 백본 도입 | 2.1 확정 모델 매트릭스에 timm 기반 모델이 없다 | torchvision에 없는 백본이 필요해질 때 도입 |
| OUT-10 | 분산 학습(DDP), 다중 GPU | 단일 GPU 벤치마크로 충분 | 향후 검토 |
| OUT-11 | 하이퍼파라미터 자동 탐색(HPO) | 공정 비교 통제(NFR-02)와 목적이 다르다 | 향후 검토 |
| OUT-12 | 모델 배포·서빙(ONNX/TensorRT export, REST API) | boilerplate 목적 밖 | 향후 검토 |
| OUT-13 | Web UI, 대시보드 | CLI 및 표 산출물로 충분 | 향후 검토 |
| OUT-14 | memory-bank류 Anomaly 모델(PatchCore 등) | 표준 gradient 루프를 벗어나며 v0.1 비교 모델이 아니다 | 훅 구조(FR-09)로 확장 여지만 확보 |
| OUT-15 | 사전학습 백본의 재학습(pretraining) | 비교 축은 fine-tuning 조건 통제다 | 영구 비범위 |

## 7. 수용 기준

v0.1은 다음이 모두 충족될 때 완료로 판정한다. AC-01~AC-06은 `BRIEF.md` 9장 성공 기준에 대응하고,
AC-07~AC-10은 AGENTS.md의 공격 초점을 검증 가능한 형태로 옮긴 것이다.

| ID | 수용 기준 | 대응 요구사항 |
|---|---|---|
| AC-01 | 공통 엔진 위에서 4개 태스크가 동일 CLI/Config 흐름으로 학습·평가된다 | FR-02, FR-03, FR-29 |
| AC-02 | 각 태스크에서 2.1 매트릭스의 3개 모델이 동일 조건으로 학습·비교되고 해당 비교 축이 leaderboard에서 확인된다 | FR-18, FR-28, FR-35, NFR-02 |
| AC-03 | 태스크별 표준 metric과 params / FLOPs / FPS가 함께 리포트된다 | FR-19, FR-26 |
| AC-04 | 태스크 고정 상태에서 split을 순차 학습·평가·추론하고 결과를 leaderboard 한 표로 비교한다. split별 추론 산출물이 생성된다 | FR-23, FR-24, FR-25 |
| AC-05 | `oxford_pets` 단일 이미지셋이 Cls/Seg/Det 3태스크에 재사용된다 | FR-12, FR-13, FR-14 |
| AC-06 | Anomaly는 MVTec `bottle`에서 custom/STFPM/EfficientAD 비교가 동작하고 image-AUROC와 pixel-AUROC가 모두 산출된다 | FR-16, FR-19, FR-10 |
| AC-07 | 네트워크 차단 상태에서 전체 흐름이 완주한다 | NFR-07, CON-03, CON-04, CON-05 |
| AC-08 | 동일 seed·config 재실행이 `PLAN-P1`에 정의된 허용 오차 내로 동일 결과를 산출한다 | NFR-01, FR-27 |
| AC-09 | 합성 fixture 또는 스모크 테스트에서 N=0, N=1, N>1과 복수 클래스가 한 배치에 섞인 Detection 입력이 Loss·Metric·Postprocess까지 완주한다 | FR-14, FR-15, NFR-06 |
| AC-10 | train/valid/test split의 샘플 ID 집합이 상호 배타적임이 검사로 확인되고, test split이 모델 선택·threshold 결정에 사용되지 않는다 | FR-17, NFR-03 |

## 8. 미결정 사항

다음 항목의 확정 위치는 `PLAN.md` 7장에 정리되어 있다. 공통 계약 항목은 `plans/PLAN-P1-foundation.md`에서 확정한다.

| 항목 | 내용 | 확정 시점 |
|---|---|---|
| 폴더구조·산출물 경로 | 원점 재설계 원칙(CON-12)에 따른 모듈 경계와 경로 규칙 | PLAN-P1 |
| split 선언 형식 | 단일 config의 sweep 표현 방식 대 split별 config 파일 나열 방식 | PLAN-P1 |
| CLI 인터페이스 | 서브커맨드 구성과 `FR-34` override 문법 | PLAN-P1 |
| 재현성 수치 기준 | metric별 허용 오차, deterministic 설정 범위(`torch.use_deterministic_algorithms` 적용 여부 포함) | PLAN-P1 |
| 통제 필드 목록 | `FR-35`가 비교하는 config 필드 집합과 승인 예외 표기 방식 | PLAN-P1 |
| 스모크 규격 | 태스크별 샘플 수·해상도·batch·최대 실행 시간 (epoch은 5로 확정) | PLAN-P1 |
| FLOPs 측정 방법 | `torch.utils.flop_counter.FlopCounterMode`(torch 2.5 내장) 사용으로 확정. 신규 의존성 없음 | 확정 (PLAN.md 6.2) |
| Detection custom 모델 구조 | 1-stage 구성의 구체 설계 | PLAN-P4 |
| YOLO·anomalib 가중치 조달 | 로컬 조사 결과 `yolov8n.pt`, EfficientAD teacher, STFPM backbone 가중치가 모두 `/mnt/d/backbones`에 존재함을 확인해 해소되었다. `BRIEF.md` 8.1의 "v0.1 pretrained는 모두 torchvision에 존재" 서술은 부정확하나 조달 문제는 발생하지 않는다 | 확정 (PLAN.md 6.3) |
| anomalib 사용 방식 | anomalib를 설치하지 않고 PyTorch 모델 정의만 저장소로 복사한다(2026-08-18 사용자 결정) | 확정 (PLAN.md 3.6) |
| mAP 산출 백엔드 | `faster-coco-eval`을 추가하고 `torchmetrics.detection.MeanAveragePrecision`을 사용한다(2026-08-18 사용자 승인, CON-08) | 확정 (PLAN.md 6.2) |
| 원격 저장소 | `https://github.com/nampluskr/cv_boilerplate`로 확정. 초기화와 원격 연결은 P1에서 수행한다 | 확정 |

*작성일: 2026-08-18 · 갱신일: 2026-08-18 · 버전: v0.1 · 상위 문서: BRIEF.md · 다음 단계: PLAN.md (Phase 전체 정의)*
