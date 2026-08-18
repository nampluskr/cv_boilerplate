# PLAN — PyTorch CV Boilerplate v0.1

## 1. 문서 목적

이 문서는 `PRD.md`의 요구사항을 실행 단위인 Phase로 분해하고, 각 Phase의 범위·산출물·수용 기준·의존성과
에이전트 실행 구조를 정의한다. Phase별 상세 설계와 계약 조항은 `plans/PLAN-P{n}-*.md`에 둔다.

- 범위: Phase 분해, 실행 모델, 공통 검증 규격, 공통 계약 변경 정책
- 비범위: Phase 내부의 모듈 경계·클래스 시그니처·config 스키마 → 각 Phase PLAN 문서
- 상위 문서: `BRIEF.md`(의도), `PRD.md`(요구사항 ID)

### 1.1. 문서 체인 변경

v0.1은 별도 `SPEC.md`를 두지 않는다. 4개 태스크의 "무엇을 만드는가"와 "어떻게 만드는가"가 Phase 단위로
붙어 있어 문서를 두 벌로 나누면 정합성 유지 비용만 커진다는 판단이다(2026-08-18 사용자 결정).

```text
BRIEF.md → PRD.md → PLAN.md → plans/PLAN-P{n}-*.md → backlog.json
```

`PRD.md` 8장의 미결정 사항 중 공통 계약에 해당하는 항목은 `plans/PLAN-P1-foundation.md`에서 확정한다.
적대적 검증과 `backlog.json`이 참조하는 "위반 조항"은 각 PLAN 문서의 절 번호(예: `PLAN-P1 §4.2`)를 사용한다.

## 2. 실행 모델

### 2.1. 계층 구조

| 계층 | 주체 | 담당 |
|---|---|---|
| master | 메인 세션 | P1·P6 직접 구현, 태스크 Phase 위임·통합·검토·보고·커밋, 공통 코드 수정 권한 |
| task 에이전트 | Phase당 1개 | 태스크 기반(Dataset·split·transform·collate·metric·adapter·postprocess·시각화) 구현, 모델 에이전트 위임, 통합·스모크·leaderboard |
| 모델 에이전트 | 태스크당 3개 | 모델 1종의 `models/<name>.py`와 config 1개 구현 |

### 2.2. 실행 순서

- 태스크 Phase(P2~P5)는 **순차** 진행한다. 앞 Phase가 완료·검토·승인된 뒤 다음 Phase를 시작한다.
  GPU가 1장(GTX 1080 Ti)이므로 학습 실행이 겹치지 않고, 앞 태스크에서 드러난 계약 결함을 다음 태스크 전에 해소할 수 있다.
- 태스크 Phase 내부는 2단계다. (1) task 에이전트가 태스크 기반을 **직렬**로 확정해 모델이 붙잡을 인터페이스를 고정하고,
  (2) 모델 3종을 모델 에이전트 3개가 **동시** 구현한다.
- 모델 에이전트는 자기 모델 파일과 config 외의 파일을 수정하지 않는다. 3개 에이전트의 산출 파일은 서로 겹치지 않는다.

### 2.3. 태스크 Phase 하위 단계 템플릿

P2~P5는 실행 구조가 같으므로 동일한 5단계를 따른다. 각 Phase 절(3.3~3.6)에는 이 템플릿과 다른 편차만 적는다.
`n`은 Phase 번호로 대체한다.

| 단계 | 주체 | 내용 | 완료 조건 |
|---|---|---|---|
| Pn.1 | task 에이전트 | 태스크 기반. Dataset·split·transform·`collate_fn`·metric·TaskAdapter·postprocess·시각화, base config, 모델 인터페이스 명세 | 더미 배치가 Dataset → collate → adapter → metric까지 통과 |
| Pn.2 | 모델 에이전트 3개 (동시) | 모델 1종씩 `models/<name>.py`와 config 1개 | 각 모델이 더미 배치 forward·loss 통과 |
| Pn.3 | task 에이전트 | 3모델 통합, 축소 subset 5 epoch 학습·평가·추론 | 3모델 완주 |
| Pn.4 | task 에이전트 | split 선언, 통제 조건 검사 통과, leaderboard 생성 | 3모델을 한 표로 비교 |
| Pn.5 | master | 적대적 검증, 변경 요청 처리, 보고·커밋 승인 요청 | 미해결 Critical 없음 |

Pn.1의 필수 산출물은 **모델 인터페이스 명세**다. 모델 에이전트 3개가 동시에 작업하려면 모델이 무엇을 입력받아
무엇을 반환하는지와 loss를 누가 계산하는지가 Pn.1에서 확정되어야 한다. 이것이 없으면 3개 에이전트가 서로 다른
시그니처로 구현하고 Pn.3에서 전부 다시 맞춰야 한다.

Pn.5의 적대적 검증 결과는 `reviews/A{n}.md`에 기록하고, 변경 내용·검증 결과·남은 위험을 사용자에게 보고한 뒤
커밋·푸시 승인을 요청한다.

## 3. Phase 정의

### 3.1. 전체 구성

| ID | Phase | 주체 | 의존 | 적대적 검증 필수 |
|---|---|---|---|---|
| P1 | 공통 기반 | master | — | 필수 |
| P2 | Classification | task 에이전트 + 모델 3 | P1 | 선택 |
| P3 | Segmentation | task 에이전트 + 모델 3 | P1 | 선택 |
| P4 | Object Detection | task 에이전트 + 모델 3 | P1 | 필수 |
| P5 | Anomaly Detection | task 에이전트 + 모델 3 | P1 | 필수 |
| P6 | 통합 검증·마무리 | master | P2~P5 | 필수 |

P2~P5는 P1에만 의존하고 서로 의존하지 않는다. 순차 실행은 자원과 계약 안정성을 위한 운영 결정이며
설계상의 의존 관계가 아니다.

### 3.2. P1 — 공통 기반

전체 작업량의 가장 큰 비중을 차지하며, 여기서 고정한 계약이 P2~P5의 병렬 실행 가능 여부를 결정한다.

| 항목 | 내용 |
|---|---|
| 범위 | 패키지 스켈레톤, `requirements.txt`, `git init`·remote 연결, 오프라인 가드와 로컬 가중치 로더, Config 로더·검증·override, Registry, seed/device/AMP, run 디렉토리 및 산출물 경로 규칙, Logger, Checkpoint, Trainer 루프, `TaskAdapter` 계약, optimizer/scheduler/dataloader 빌더, 비표준 학습 훅, 벤치마크 러너, 통제 조건 기계 검사, params/FLOPs/FPS 프로파일, leaderboard 생성, CLI 전체 서브커맨드 |
| 확정 사항 | `PRD.md` 8장 미결정 중 폴더구조·산출물 경로, config 스키마, split 선언 형식, CLI 인터페이스와 override 문법, 재현성 허용 오차와 deterministic 범위, 통제 필드 목록, 스모크 규격, FLOPs 측정 방법 |
| 검증 | 합성 toy 태스크 4종(3.2.1)이 학습·평가·추론·leaderboard까지 완주 |
| 관련 요구사항 | FR-01~FR-10, FR-23~FR-29, FR-33~FR-35, NFR-01~NFR-04, NFR-07, NFR-10 |

벤치마크 러너와 leaderboard를 P1에 포함한다. 태스크 Phase에 두면 태스크 에이전트 4개가 각각 집계 코드를
만들게 되고 통제 기준이 갈라진다.

#### 3.2.1. toy 태스크 4종

P1은 실데이터 없이 합성 데이터로 4가지 타깃 형태를 모두 통과시켜야 한다. toy-det과 toy-anomaly가
P1에서 통과하지 않으면 P4·P5 에이전트가 공통 엔진을 수정하게 되고 병렬 구조가 무너진다.

| toy 케이스 | 검증 대상 |
|---|---|
| toy-cls | 스칼라 라벨, 기본 학습·평가 경로 |
| toy-seg | dense 라벨 `(H, W)`, 픽셀 단위 metric |
| toy-det | 가변 N(N=0, N=1, N>1 혼합 배치), list-of-dict 타깃, 태스크별 `collate_fn` |
| toy-anomaly | 학습 시 타깃 없음, 모델별 `train_step` 분기, fit 전 훅 |

#### 3.2.2. 하위 단계

P1은 "계약을 코드로 먼저 고정하고 엔진을 그 계약에 맞춰 구현한다"는 순서를 따른다.

| 단계 | 내용 | 검증 | 의존 |
|---|---|---|---|
| P1.1 | 스켈레톤·환경. 패키지 구조, `requirements.txt`(`faster-coco-eval` 포함), `git init`·remote 연결, 오프라인 가드, 로컬 가중치 로더, 로컬 자산 점검 커맨드 | 데이터셋·가중치 경로 전수 확인 통과 | — |
| P1.2 | Config·Registry. YAML 로더, 스키마 검증, `--set` override, Registry, run 디렉토리와 산출물 경로 규칙 | 합성 config의 resolve·검증·경로 생성 | P1.1 |
| P1.3 | 실행 컨텍스트. seed/determinism, device, AMP 스위치, Logger, Checkpoint, 환경 정보 기록 | 동일 seed 2회 실행이 동일 난수열, 체크포인트 저장·재개 | P1.2 |
| P1.4 | 계약 정의. `TaskAdapter` 인터페이스와 toy 4종 fixture(dataset·model·adapter·metric)를 엔진보다 먼저 작성 | toy 4종이 계약대로 배치를 구성 (엔진 없이 단독) | P1.2 |
| P1.5 | Trainer 엔진. 학습·평가 루프, optimizer/scheduler/dataloader 빌더, 비표준 학습 훅, 평가 무결성 | toy 4종 학습·평가 5 epoch 완주 | P1.3, P1.4 |
| P1.6 | CLI. `config` / `train` / `evaluate` / `predict` 서브커맨드 | toy로 4개 서브커맨드 완주 | P1.5 |
| P1.7 | 프로파일. params / FLOPs / FPS 측정 | toy 모델에서 3지표 산출 | P1.5 |
| P1.8 | 벤치마크. split 선언, 순차 실행, 통제 조건 기계 검사, leaderboard(CSV/Markdown) | toy 3모델 split의 leaderboard 생성, 통제 위반 config가 실패로 종료 | P1.6, P1.7 |
| P1.9 | 마감. 재현성 2회 실행으로 허용 오차 수치 확정, 적대적 검증, `PLAN-P1` 조항 확정, 커밋 | toy 전체 재실행이 허용 오차 내 일치 | P1.8 |

P1.4를 P1.5보다 앞에 둔다. toy 4종 fixture가 먼저 존재해야 엔진이 4가지 계약을 모두 만족시키도록 구현된다.
순서를 뒤집으면 엔진을 먼저 만들고 toy를 거기에 맞추게 되어, 실제로는 toy-cls·toy-seg만 통과하는 엔진이
남고 P4·P5에서 계약이 깨진다. P1.4에서 특히 확정할 대상은 toy-det의 가변 N `collate_fn` 계약과
toy-anomaly의 "타깃 없는 학습 + 모델별 `train_step` 분기 + fit 전 훅" 계약이다.

P1.7은 P1.5에만 의존하므로 P1.6과 병렬로 진행할 수 있고 나머지는 직렬이다.
P2 착수 조건은 P1.9 완료다. P1.8 없이 태스크 Phase를 시작하면 태스크 에이전트가 leaderboard를 만들 수단이
없어 각자 집계 코드를 작성하게 된다.

#### 3.2.3. 적대적 검증

P1은 적대적 검증 필수 Phase다. P1의 계약 결함은 등급 C 변경(5.2)을 유발해 완료된 태스크를 전부 재실행하게
만들며, 순차 진행이므로 후반에 발견될수록 비용이 커진다. master가 직접 구현하므로 검토자는 Codex CLI다.

검증은 P1.9에서 한 번에 수행한다. 실행 횟수 3회 한도가 검증 대상 단위로 적용되므로 중간 단계에서 회차를
소진하지 않는다.

P1에는 실데이터도 실모델도 없어 태스크 간 비교로 task-agnostic 위반을 잡을 수 없다. 따라서 공격 초점은
계약의 일반성에 둔다.

| 축 | 공격 내용 |
|---|---|
| 계약 일반성 | `TaskAdapter` 인터페이스로 N=0을 포함한 가변 N 검출 배치를 표현할 수 있는가. 타깃 없는 학습과 모델별 `train_step` 분기를 어댑터·훅만으로 처리할 수 있는가 |
| 엔진 순수성 | 공통 루프에 태스크 이름이나 타깃 형태로 분기하는 조건문이 있는가. toy 4종 통과를 위해 엔진에 특례를 넣지 않았는가 |
| 평가 무결성 | `model.eval()`·`torch.no_grad()` 적용 범위, metric 리셋 누락, valid/test 경로 혼용 |
| 재현성 | seed 적용 범위(Python·NumPy·torch CPU/CUDA·DataLoader worker·sampler), config 저장이 실제 재현을 보장하는가 |
| 통제 검사 | `FR-35`의 통제 필드 비교를 우회할 수 있는가. override로 통제 필드를 바꿔도 검사를 통과하는가 |
| 오프라인 | 네트워크 접근을 유발하는 경로가 남아 있는가 |

이 표가 `backlog.json`의 P1 `adversarialFocus`가 되고, `planRefs`는 `PLAN-P1`의 해당 조항 번호를 가리킨다.

### 3.3. P2 — Classification

| 항목 | 내용 |
|---|---|
| 태스크 기반 | `oxford_pets` 37 breeds Dataset, split 생성·상호 배타성 검사, transforms v2 파이프라인, Top-1/F1 metric, TaskAdapter, 예측 시각화 |
| 모델 에이전트 1 | Custom CNN (from scratch) |
| 모델 에이전트 2 | ResNet50 (torchvision, 로컬 `.pth` 주입) |
| 모델 에이전트 3 | EfficientNet-B0 (torchvision, 로컬 `.pth` 주입) |
| 검증 | 3모델 축소 subset 5 epoch 학습·평가·추론 완주, leaderboard 생성 |
| 관련 요구사항 | FR-11, FR-12, FR-17~FR-22, FR-30~FR-32, NFR-05 |

하위 단계는 2.3 템플릿을 그대로 따르며 편차가 없다. 4개 태스크 중 가장 가볍고, 여기서 템플릿의 실효성을
처음 검증한다.

### 3.4. P3 — Segmentation

| 항목 | 내용 |
|---|---|
| 태스크 기반 | trimap 3-class Dataset과 라벨 매핑, 이미지·마스크 동시 변환, mIoU/Dice metric, TaskAdapter, 마스크 시각화 |
| 모델 에이전트 1 | Custom U-Net류 |
| 모델 에이전트 2 | DeepLabV3-ResNet50 (로컬 `.pth`) |
| 모델 에이전트 3 | FCN-ResNet50 (로컬 `.pth`) |
| 검증 | 3모델 축소 subset 5 epoch 완주, leaderboard 생성 |
| 관련 요구사항 | FR-11, FR-13, FR-18~FR-22 |

torchvision 2종은 backbone을 ResNet50으로 통일해 head 차이만 비교한다.

하위 단계는 2.3 템플릿을 따른다. 편차는 P3.1에 이미지와 마스크의 동시 변환이 추가되는 정도다.

### 3.5. P4 — Object Detection

위험도가 가장 높은 Phase다. 모델 3종의 loss·postprocess 편차가 커서 모델 에이전트 간 부하가 균등하지 않다.

| 항목 | 내용 |
|---|---|
| 태스크 기반 | Pascal VOC XML 파서, 가변 N 타깃과 `collate_fn`(N=0 허용), mAP@0.5:0.95 metric, NMS·score threshold, 박스 시각화 |
| 모델 에이전트 1 | Custom 1-stage anchor-free 검출기(backbone·FPN·shared head·assigner·loss 포함) |
| 모델 에이전트 2 | Faster R-CNN ResNet50-FPN (로컬 `.pth`, box predictor 교체) |
| 모델 에이전트 3 | YOLOv8n 어댑터 (로컬 `yolov8n.pt`, 타깃 규약과 ultralytics 형식 변환을 모델 래퍼 안에 캡슐화) |
| 검증 | N=0·N=1·N>1과 복수 클래스가 섞인 fixture 완주, 3모델 축소 subset 5 epoch 완주, leaderboard 생성 |
| 관련 요구사항 | FR-14, FR-15, FR-18~FR-20, NFR-06, AC-09, CON-11 |

Custom 모델은 loss와 assigner까지 새로 작성해야 하므로 모델 에이전트 1의 분량이 다른 둘보다 크다.
YOLO 어댑터는 타깃 형식 변환을 모델 래퍼 안에 가두어 Dataset과 엔진이 ultralytics 형식을 알지 못하게 한다.

하위 단계는 2.3 템플릿을 따르되 두 가지 편차가 있다.

- P4.1이 다른 태스크보다 무겁다. VOC XML 파서, 가변 N `collate_fn`, mAP 백엔드 연결, NMS·score threshold가
  모두 여기에 들어간다.
- P4.2의 부하가 균등하지 않다. 모델 에이전트 1(custom 1-stage)은 assigner와 loss까지 작성해야 하므로 다른
  둘보다 크다. 착수를 먼저 시키고, 범위 축소안을 `PLAN-P4`에 미리 준비한다.

### 3.6. P5 — Anomaly Detection

| 항목 | 내용 |
|---|---|
| 태스크 기반 | MVTec `bottle` Dataset(train=good only, test+GT mask), image/pixel AUROC, threshold 결정은 valid만 사용, anomaly map 시각화 |
| 모델 에이전트 1 | Custom Reconstruction/AE |
| 모델 에이전트 2 | STFPM (anomalib의 PyTorch 모델 정의를 저장소로 복사, teacher는 로컬 백본 가중치) |
| 모델 에이전트 3 | EfficientAD (동일 방식, 로컬 teacher 가중치 사용) |
| 검증 | 3모델 축소 subset 5 epoch 완주, image-AUROC와 pixel-AUROC 동시 산출, leaderboard 생성 |
| 관련 요구사항 | FR-10, FR-16, FR-18, FR-19, NFR-03, CON-02, CON-05 |

anomalib는 설치하지 않는다. `github.com/nampluskr/defectvad`와 동일하게 anomalib의 PyTorch 모델 정의
(`nn.Module`)만 저장소에 복사해 사용하며 런타임 의존성을 만들지 않는다(2026-08-18 사용자 결정).
세 모델의 학습 방식(재구성, teacher-student, teacher + autoencoder + quantile 정규화)이 모두 다르므로
`train_step` 분기 지점은 P1의 어댑터 계약에서 확정한 형태를 따른다.

하위 단계는 2.3 템플릿을 따르되 두 가지 편차가 있다.

- P5.1에 threshold 결정 정책(valid split만 사용)이 추가된다.
- P5.1의 필수 산출물에 **훅 사용 명세**가 포함된다. 3모델의 학습 방식이 모두 다르므로, 각 모델이 어떤 훅과
  어떤 `train_step` 형태를 쓰는지 P5.1에서 지정하지 않으면 모델 에이전트 3개가 각자 다른 방식으로 우회한다.

### 3.7. P6 — 통합 검증·마무리

| 항목 | 내용 |
|---|---|
| 범위 | 4태스크 전체 흐름 재실행, 동일 seed 2회 재현성 검증, 네트워크 차단 상태 완주 검증, split 누수 검사, `AC-01`~`AC-10` 대조표, 공통 엔진 한계선 문서화, README 작성 |
| 검증 | 수용 기준 전 항목 통과 |
| 관련 요구사항 | AC-01~AC-10, FR-09, NFR-01, NFR-07, NFR-09 |

master 단독 수행이며 2.3 템플릿을 따르지 않는다. 검증 항목의 나열이다.

| 단계 | 내용 |
|---|---|
| P6.1 | 4태스크 전체 흐름 재실행 |
| P6.2 | 동일 seed 2회 재현성 검증 |
| P6.3 | 네트워크 차단 상태 완주 검증 |
| P6.4 | split 누수 검사 (상호 배타성, test split 미사용 확인) |
| P6.5 | `AC-01`~`AC-10` 대조표, 공통 엔진 한계선 문서화, README 작성 |
| P6.6 | 최종 적대적 검증, 보고·커밋 |

## 4. 공통 검증 규격

모든 Phase의 구현 검증은 toy 데이터 또는 축소 subset과 **5 epoch**로 수행한다(2026-08-18 사용자 결정).
전체 데이터 장시간 학습은 v0.1의 완료 조건이 아니며, leaderboard도 이 규격에서 생성한다.

| Phase | 데이터 | epoch |
|---|---|---|
| P1 | 합성 toy 4종 | 5 |
| P2~P5 | 실데이터 축소 subset | 5 |
| P6 | 실데이터 축소 subset (재현성 검증은 동일 조건 2회) | 5 |

subset 크기, 해상도, batch, 최대 실행 시간의 구체 수치는 `plans/PLAN-P1-foundation.md`에서 확정하고
태스크별 조정은 각 Phase PLAN에서 기록한다.

## 5. 공통 계약 변경 정책

P1에서 4태스크의 요구를 완전히 예측할 수 없으므로 P2~P5 진행 중 공통 코드 수정을 허용한다.
다만 태스크 에이전트가 각자 공통 코드를 고치면 엔진에 태스크 분기가 침투하므로(`FR-02`, `NFR-04` 위반)
다음 통제를 둔다.

### 5.1. 수정 권한

공통 코드(엔진·벤치마크·CLI 계층) 수정은 master만 수행한다. task 에이전트와 모델 에이전트는 직접 수정하지 않고
**변경 요청**을 반환한다. 요청에는 문제, 정확한 재현 조건, 최소 수정안, 영향받는 P1 조항 번호를 포함한다.
모델 에이전트의 요청은 task 에이전트를 거쳐 master로 올라온다.

### 5.2. 변경 등급

| 등급 | 내용 | 처리 |
|---|---|---|
| A | 계약 무변경. 공통 코드 버그 수정, 인터페이스 불변 | master가 수정 후 회귀 검증 |
| B | 계약 확장. 훅·옵션 추가이며 기본값 유지로 하위호환 | master 수정 + P1 조항에 항목 추가 |
| C | 계약 변경. 기존 인터페이스나 타깃 규약을 변경 | 완료된 태스크 전부 재실행 필요. 사용자 승인 후 진행 |

### 5.3. 필수 제약

등급과 무관하게 공통 루프에 태스크 이름으로 분기하는 수정은 허용하지 않는다. 그런 요구가 올라오면
어댑터 또는 훅으로 흡수하는 형태로 되돌린다.

### 5.4. 회귀 방어

완료된 태스크의 toy 4종과 스모크 실행이 회귀 테스트 세트가 된다. 공통 코드가 바뀌면 그 시점까지 완료된
모든 태스크의 스모크를 재실행하고 metric이 재현성 허용 오차 안에 있는지 확인한다. 순차 진행이므로
후반 Phase의 변경일수록 비용이 크며, 이것이 P1에서 toy-det과 toy-anomaly를 미리 통과시켜야 하는 이유다.

### 5.5. 기록

조항별 개정 이력은 `plans/PLAN-P1-foundation.md`에 남긴다. 변경 요청과 처리 결과는 해당 Phase의
`reviews/A{n}.md`에 기록한다. 이 정책은 `backlog.json`의 `commonContractChangePolicy`에 반영한다.

## 6. 환경 확정 사항

작업 시작 전 로컬 환경을 조사해 확인한 사실이다(2026-08-18). 이 결과로 `PRD.md` 8장의 미결정 항목 일부가 해소된다.

### 6.1. 실행 환경

| 항목 | 값 |
|---|---|
| conda 환경 | `pytorch_env` (`/home/nampl/anaconda3/envs/pytorch_env/bin/python`) |
| torch / torchvision | 2.5.1+cu121 / 0.20.1+cu121 |
| torchmetrics | 1.8.2 |
| ultralytics | 8.4.101 |
| GPU | GTX 1080 Ti, 11.8 GB |

GTX 1080 Ti는 Pascal 세대로 fp16 tensor core가 없어 AMP 이득이 없고 재현성에 불리하다. AMP 기본값은 off로 둔다.

### 6.2. 의존성 결정

| 항목 | 결정 |
|---|---|
| anomalib | 설치하지 않는다. PyTorch 모델 정의만 저장소로 복사한다 |
| FLOPs 측정 | `torch.utils.flop_counter.FlopCounterMode`(torch 2.5 내장)를 사용한다. 새 의존성 없음 |
| mAP 산출 | `faster-coco-eval`을 추가한다(2026-08-18 사용자 승인, `CON-08`). `torchmetrics.detection.MeanAveragePrecision`은 `pycocotools` 또는 `faster-coco-eval` 없이는 동작하지 않으며 둘 다 미설치 상태였다 |
| timm | 도입하지 않는다(`OUT-09`) |

### 6.3. 로컬 자산

`PRD.md` 8장의 "YOLO 가중치 조달" 미결정 항목은 해소된다. v0.1 pretrained 모델의 가중치가 모두 로컬에 존재한다.

| 모델 | 로컬 가중치 |
|---|---|
| ResNet50 | `/mnt/d/backbones/resnet50-0676ba61.pth` |
| EfficientNet-B0 | `/mnt/d/backbones/efficientnet_b0_rwightman-7f5810bc.pth` |
| DeepLabV3-ResNet50 | `/mnt/d/backbones/deeplabv3_resnet50_coco-cd0a2569.pth` |
| FCN-ResNet50 | `/mnt/d/backbones/fcn_resnet50_coco-1167a1af.pth` |
| Faster R-CNN R50-FPN | `/mnt/d/backbones/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth` |
| YOLOv8n | `/mnt/d/backbones/yolov8n.pt` |
| EfficientAD teacher | `/mnt/d/backbones/efficientad_pretrained_weights/pretrained_teacher_{small,medium}.pth` |
| STFPM teacher 후보 | `/mnt/d/backbones/resnet18-f37072fd.pth`, `wide_resnet50_2-95faca4d.pth` |

### 6.4. 데이터셋 실측

| 항목 | 값 |
|---|---|
| `oxford_pets/images` | 7,390장 (전부 jpg) |
| `oxford_pets/annotations/list.txt` | 7,349개 항목 |
| `oxford_pets/annotations/trainval.txt` / `test.txt` | 3,680 / 3,669 |
| `oxford_pets/annotations/trimaps` | 7,390개, 값은 `{1, 2, 3}` |
| `oxford_pets/annotations/xmls` | 3,686개 |
| `mvtec/bottle/train/good` | 209장, 900x900 |
| `mvtec/bottle/test` | `good`, `broken_large`, `broken_small`, `contamination` |
| `mvtec/bottle/ground_truth` | 결함 3종 마스크, 값은 `{0, 255}` |

이미지 수(7,390)와 `list.txt` 항목 수(7,349)가 일치하지 않는다. 샘플 인덱스의 기준은 이미지 디렉토리가 아니라
`list.txt`로 잡아야 하며, 공식 `trainval.txt` / `test.txt`를 고정 분할로 사용할 수 있다. 상세는
`plans/PLAN-P2-classification.md`에서 확정한다.

## 7. 미결정 사항의 확정 위치

`PRD.md` 8장 항목의 확정 위치는 다음과 같다.

| 항목 | 확정 위치 |
|---|---|
| 폴더구조·산출물 경로 | `PLAN-P1` |
| split 선언 형식 | `PLAN-P1` |
| CLI 인터페이스와 override 문법 | `PLAN-P1` |
| 재현성 수치 기준 | `PLAN-P1` |
| 통제 필드 목록 | `PLAN-P1` |
| 스모크 규격 | `PLAN-P1` (태스크별 조정은 각 Phase PLAN) |
| FLOPs 측정 방법 | 확정 (6.2) |
| Detection custom 모델 구조 | `PLAN-P4` |
| YOLO 가중치 조달 | 확정 (6.3) |
| 원격 저장소 | 확정. `https://github.com/nampluskr/cv_boilerplate` |

## 8. 문서 인덱스

Phase별 PLAN은 원래 해당 Phase 착수 직전에 작성할 계획이었으나, **2026-08-18 사용자 결정으로 P1~P6
전체를 착수 전에 일괄 작성했다.** 전체 계약을 미리 확정해 두는 편이 태스크 Phase의 병렬 실행 준비와
`backlog.json` 작성에 유리하다는 판단이다.

이에 따른 유지 비용은 다음 규칙으로 흡수한다. 앞 Phase에서 계약 조정이 발생하면 해당 조항을 `PLAN-P1 §16`의
개정 이력에 기록하고 영향받는 뒤쪽 Phase PLAN을 함께 갱신한다. 갱신 순서는
`plans/PLAN-P{n}-*.md → PLAN.md → backlog.json → PRD.md`다.

| 문서 | 내용 |
|---|---|
| `plans/PLAN-P1-foundation.md` | P1 상세 + 공통 계약 조항 |
| `plans/PLAN-P2-classification.md` | P2 상세 |
| `plans/PLAN-P3-segmentation.md` | P3 상세 |
| `plans/PLAN-P4-detection.md` | P4 상세 |
| `plans/PLAN-P5-anomaly.md` | P5 상세 |
| `plans/PLAN-P6-integration.md` | P6 상세 |

## 9. 위험

| 위험 | 영향 | 완화 |
|---|---|---|
| P1 계약 불완전 | P2~P5에서 공통 코드 수정이 반복되어 병렬 구조가 무너진다 | toy 4종(3.2.1)을 P1 수용 기준에 포함하고 P1을 적대적 검증 필수로 지정 |
| Detection 모델 3종의 부하 불균형 | 모델 에이전트 1(custom)이 병목이 된다 | P4 PLAN에서 custom 모델의 범위를 축소 가능한 형태로 설계하고 착수 전 분량을 재평가 |
| Anomaly 3모델의 학습 방식 상이 | 모델 에이전트가 각자 다른 방식으로 우회한다 | `train_step` 분기 지점을 P1 어댑터 계약에서 확정하고 P5 PLAN에서 3모델의 훅 사용을 명시 |
| anomalib 모델 정의 복사의 정합성 | 복사한 코드가 로컬 teacher 가중치와 키가 맞지 않을 수 있다 | 로컬 가중치 로드를 P5의 수용 기준에 포함하고 실패 시 조용한 폴백 없이 오류로 보고 |
| ultralytics의 네트워크 접근 | 오프라인 원칙(`NFR-07`) 위반 | 로컬 `.pt`만 사용하고 ultralytics trainer를 쓰지 않으며, P6에서 네트워크 차단 검증으로 확인 |
| 5 epoch 스모크의 낮은 성능 | 모델 간 성능 차이가 통계적으로 유의하지 않을 수 있다 | v0.1의 목적은 파이프라인 검증과 비교 도구 완성이며 절대 성능이 아님을 leaderboard에 명시 |

---

*작성일: 2026-08-18 · 갱신일: 2026-08-18 · 버전: v0.1 · 상위 문서: PRD.md · 다음 단계: P1 구현 (plans/PLAN-P1-foundation.md, backlog.json)*
