# PLAN-P1 — 공통 기반 (Foundation)

## 1. 문서 목적

이 문서는 `PLAN.md` 3.2의 P1(공통 기반)을 구현 가능한 수준으로 상세화하고, P2~P5가 의존하는
**공통 계약**을 확정한다. 이 문서의 절 번호(`PLAN-P1 §6.2` 형식)가 적대적 검증과 `backlog.json`이
참조하는 조항 ID다.

- 범위: 패키지 구조, Config, Registry, 실행 컨텍스트, `TaskAdapter` 계약, Trainer 엔진, 데이터 계층,
  오프라인 자산, 벤치마크, CLI, toy fixture, 스모크 규격, P1 하위 단계 수용 기준
- 비범위: 실데이터 Dataset 구현, 실모델 구현 → `plans/PLAN-P{2..5}-*.md`
- 상위 문서: `PLAN.md`(Phase 정의), `PRD.md`(요구사항 ID), `BRIEF.md`(의도)

이 문서는 `PRD.md` 8장 미결정 사항 중 다음을 확정한다.

| 미결정 항목 | 확정 절 |
|---|---|
| 폴더구조·산출물 경로 | §2 |
| config 스키마 | §3 |
| split 선언 형식 | §8.2, §10.1 |
| CLI 인터페이스와 override 문법 | §11, §3.3 |
| 재현성 수치 기준 | §5.2 |
| 통제 필드 목록 | §10.2 |
| 스모크 규격 | §13 |

## 2. 패키지 구조와 경로 규칙

### 2.1. 저장소 구조

`CON-12`(원점 재설계)에 따라 기존 프로젝트의 구조를 따르지 않는다. 참고하는 것은 "CLI에서 config를 지정해
학습·평가·추론을 실행한다"는 사용 방식뿐이다.

```text
cv_boilerplate/
├── README.md
├── AGENTS.md
├── CLAUDE.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── __main__.py               # python -m src 진입점
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── parser.py             # 서브커맨드 정의, --set 파싱
│   │   └── commands.py           # check_assets/config/train/evaluate/predict/benchmark/leaderboard
│   ├── core/                     # task-agnostic. 태스크 이름이 등장할 수 없다 (§7.4)
│   │   ├── __init__.py
│   │   ├── config.py             # 로드·상속·병합·override·검증
│   │   ├── registry.py
│   │   ├── context.py            # seed/determinism/device/AMP/run 디렉토리/환경 기록
│   │   ├── logger.py
│   │   ├── checkpoint.py
│   │   ├── builders.py           # optimizer/scheduler/dataloader 빌더
│   │   ├── adapter.py            # TaskAdapter 추상 클래스
│   │   ├── engine.py             # Trainer
│   │   ├── offline.py            # 오프라인 가드, 로컬 가중치 로더
│   │   └── errors.py             # ConfigError/RegistryError/LocalAssetError/OfflineViolationError/ControlViolationError
│   ├── bench/                    # task-agnostic
│   │   ├── __init__.py
│   │   ├── runner.py             # split 순차 실행
│   │   ├── control.py            # 통제 조건 기계 검사 (FR-35)
│   │   ├── profile.py            # params/FLOPs/FPS
│   │   └── leaderboard.py        # CSV/Markdown 집계
│   ├── data/                     # task-agnostic 데이터 유틸
│   │   ├── __init__.py
│   │   └── split.py              # split 로드·생성·상호 배타성 검사
│   ├── tasks/
│   │   ├── __init__.py           # 태스크 패키지 임포트로 registry 채움
│   │   ├── toy/                  # P1 fixture (§12)
│   │   ├── classification/       # P2
│   │   ├── segmentation/         # P3
│   │   ├── detection/            # P4
│   │   └── anomaly/              # P5
│   └── utils/
│       ├── __init__.py
│       ├── io.py                 # YAML/JSON/CSV 입출력
│       └── timing.py
├── configs/
│   ├── assets.yaml               # 로컬 데이터셋·가중치 목록 (§9.3)
│   ├── splits/                   # 생성된 split 파일 (커밋 대상)
│   ├── toy/
│   ├── classification/
│   ├── segmentation/
│   ├── detection/
│   ├── anomaly/
│   └── benchmarks/
├── docs/
│   └── dev/v0.1/...
└── outputs/                      # 실행 산출물, .gitignore 대상
```

각 태스크 패키지의 내부 구성은 동일하다. 이 구성은 P2~P5 태스크 에이전트가 그대로 따른다.

```text
src/tasks/<task>/
├── __init__.py        # register 모듈 임포트
├── dataset.py
├── transform.py
├── collate.py         # 기본 collate로 충분하면 생략
├── loss.py
├── metric.py
├── adapter.py
├── postprocess.py
├── visualize.py
└── models/
    ├── __init__.py
    └── <model_name>.py
```

모델 에이전트가 작업하는 파일은 `models/<model_name>.py` 하나와 `configs/<task>/<model_name>.yaml`
하나로 한정된다(`PLAN.md` 2.1). 이 구조가 그 제약을 물리적으로 보장한다.

### 2.2. 모듈 경계와 의존 방향

의존은 한 방향으로만 흐른다. 역방향 임포트는 금지한다.

```text
cli  →  bench  →  core  ←  tasks
                    ↑
                  data, utils
```

| 계층 | 임포트 가능 대상 | 금지 |
|---|---|---|
| `core` | `data`, `utils`, 표준 라이브러리, torch | `tasks`, `bench`, `cli` |
| `bench` | `core`, `data`, `utils` | `tasks`, `cli` |
| `data` | `utils` | `core`, `bench`, `tasks`, `cli` |
| `tasks` | `core`, `data`, `utils` | `bench`, `cli`, 다른 태스크 패키지 |
| `cli` | 전부 | — |

`tasks/<task>`가 다른 태스크 패키지를 임포트하는 것을 금지한다. 태스크 간 코드 공유가 필요해지면
`core` 또는 `utils`로 승격하며, 이는 등급 B 변경(`PLAN.md` 5.2)이다.

### 2.3. 산출물 경로 규칙

`NFR-10`(산출물 분리)을 경로 규칙으로 고정한다. 실행 단위 하나가 run 디렉토리 하나에 대응한다.

```text
outputs/
├── runs/<task_name>/<run_name>/
│   ├── config.resolved.yaml      # override까지 반영된 최종 config
│   ├── env.json                  # 환경 정보 (§5.6)
│   ├── train.log                 # 콘솔과 동일 내용
│   ├── metrics_epoch.csv         # epoch, split, loss, metric...
│   ├── metrics_final.json        # 최종 valid/test metric, profile
│   ├── checkpoints/
│   │   ├── best.pth
│   │   └── last.pth
│   ├── predictions/
│   └── visualizations/
└── benchmarks/<bench_name>/
    ├── splits/<split_name>/      # run 디렉토리와 동일 구조
    ├── control_report.json       # 통제 검사 결과 (§10.3)
    ├── leaderboard.csv
    └── leaderboard.md
```

- `<task_name>`은 `meta.task_name`(§3.1)이며 경로 그룹핑과 leaderboard 라벨에만 쓰인다. `core`와 `bench`는
  이 값을 읽지 않는다(§7.4).
- `<run_name>` 기본값은 `{config 파일 stem}__{YYYYmmdd-HHMMSS}`다. `output.run_name`으로 고정할 수 있다.
- 벤치마크 실행의 run 디렉토리는 `outputs/benchmarks/<bench_name>/splits/<split_name>/`이며 타임스탬프를
  붙이지 않는다. 재실행 시 덮어쓰기는 `--overwrite` 없이는 오류로 거부한다.
- `outputs/`, `*.pth`, `*.pt`, 데이터셋 경로는 `.gitignore`에 등록한다(`CON-15`).

### 2.4. 명명 규칙

`NFR-08`을 구체화한다.

- PEP8. 모듈·함수·변수는 `snake_case`, 클래스는 `PascalCase`, 상수는 `UPPER_SNAKE`.
- 멤버 변수에 `_` 접두사를 붙이지 않는다. 등호·콜론 세로 정렬을 하지 않는다.
- split 이름은 `train` / `valid` / `test`다. `val`, `validation`을 쓰지 않는다.
- 백본 식별자는 `backbone_name`이다. `backbone`, `arch`, `encoder_name`을 쓰지 않는다.
- 경로 처리는 `os.path`만 사용한다(`CON-07`). `pathlib.Path`를 임포트하지 않는다.
- 코드 주석과 docstring은 영어, 문서는 한국어다(`CON-14`). 이모지를 쓰지 않는다.

## 3. Config 계약

### 3.1. 스키마

config는 단일 YAML 파일이며 최상위 키는 다음 9개로 고정한다. 정의되지 않은 최상위 키는 검증 오류다.

```yaml
meta:
  task_name: classification        # 출력 경로 그룹·leaderboard 라벨 전용. 엔진은 읽지 않는다
  description: "ResNet50 baseline"

runtime:
  seed: 42
  device: cuda                     # cuda | cpu
  amp: false                       # Pascal 세대이므로 기본 off (PLAN.md 6.1)
  deterministic: warn              # strict | warn | off (§5.1)
  allow_network: false             # true 이면 오프라인 가드 해제 (§9.1)

data:
  name: oxford_pets_cls            # registry: dataset
  root: /mnt/d/datasets/oxford_pets
  params: {}
  image_size: [224, 224]           # [H, W]
  batch_size: 32
  num_workers: 4
  drop_last: false
  split:
    mode: file                     # file | ratio (§8.2)
    path: configs/splits/oxford_pets_cls.json
  transform:
    train: {name: cls_train, params: {}}
    eval:  {name: cls_eval,  params: {}}

model:
  name: resnet50_cls               # registry: model
  params:
    num_classes: 37
    backbone_name: resnet50
    weights_path: /mnt/d/backbones/resnet50-0676ba61.pth

loss:
  name: cross_entropy              # registry: loss
  params: {}

metrics:
  - {name: top1_accuracy, params: {num_classes: 37}}
  - {name: macro_f1,      params: {num_classes: 37}}

adapter:
  name: classification             # registry: adapter
  params: {}

optim:
  optimizer: {name: adamw,  params: {lr: 0.001, weight_decay: 0.0001}}
  scheduler: {name: cosine, params: {t_max: 5, eta_min: 0.00001}}

train:
  epochs: 5
  grad_clip: null                  # float 이면 clip_grad_norm_
  monitor: {metric: top1_accuracy, mode: max}   # valid 기준 모델 선택 (§7.3)
  log_interval: 10
  save_last: true

output:
  root: outputs
  run_name: null                   # null 이면 자동 생성 (§2.3)
  save_predictions: true
  save_visualizations: true
  max_visualizations: 16
```

`{name, params}` 쌍은 registry 조회 규격이며 전 계층에서 동일하다(§4).

### 3.2. 상속과 병합

`_base` 키로 다른 config를 상속한다. 값은 파일 경로 문자열 또는 경로 리스트다.

- 병합은 dict에 대해 재귀적(deep merge)이며, list와 스칼라는 자식 값이 부모 값을 통째로 대체한다.
- `metrics`는 list이므로 자식이 선언하면 전체 교체다. 부분 추가를 지원하지 않는다.
- 상속 깊이는 3단계로 제한하고 순환 참조는 오류다.
- 병합 순서는 `_base`(선언 순) → 자기 자신 → benchmark split override(§10.1) → CLI `--set`(§3.3)이다.
  뒤에 적용되는 것이 항상 이긴다.

### 3.3. override 문법 (`FR-34`)

```bash
python -m src train configs/classification/resnet50.yaml \
    --set train.epochs=5 \
    --set optim.optimizer.params.lr=0.0005 \
    --set data.image_size='[128, 128]' \
    --set model.params.weights_path=null
```

- 형식은 `--set <dotted.key>=<yaml_value>`이며 반복 지정할 수 있다.
- 값은 YAML 스칼라로 파싱한다. `null`, `true`, `3`, `0.001`, `[1, 2]`, `"문자열"`이 모두 유효하다.
- 리스트 원소는 숫자 인덱스로 지정한다. 예: `--set metrics.0.params.num_classes=10`.
- **존재하지 않는 키를 지정하면 오류다.** 오타로 인한 무음 실패를 막기 위한 규정이며, 키 신설이 필요하면
  config 파일을 수정한다.
- override는 `config.resolved.yaml`에 반영된 최종 상태로 저장되며, 통제 검사(§10.3)는 이 최종 상태를
  대상으로 수행한다. 따라서 `--set`으로 통제 필드를 바꿔도 검사를 우회할 수 없다.

### 3.4. 검증 (`FR-33`)

`config` 서브커맨드와 모든 실행 커맨드는 시작 전에 다음을 순서대로 검사하고, 실패 시 `ConfigError`로
원인과 조치를 담은 메시지를 출력하며 즉시 종료한다(`NFR-11`).

1. 최상위 키 집합이 §3.1과 일치하는가 (누락·미정의 키 검출)
2. 필수 하위 키가 존재하고 타입이 맞는가 (`data.image_size`는 길이 2의 정수 리스트 등)
3. `{name, params}` 항목의 `name`이 해당 registry에 등록되어 있는가
4. `data.root`, `data.split.path`, `model.params.weights_path` 등 경로 값이 실제로 존재하는가
5. `train.monitor.metric`이 `metrics` 목록의 이름 중 하나인가, `mode`가 `max` 또는 `min`인가
6. `runtime.device`가 `cuda`일 때 `torch.cuda.is_available()`가 참인가

## 4. Registry 계약 (`FR-04`)

네임스페이스는 다음 7개로 고정한다.

| 네임스페이스 | 등록 대상 | 빌드 시그니처 |
|---|---|---|
| `dataset` | `torch.utils.data.Dataset` 서브클래스 | `build(root, split, transform, **params)` |
| `transform` | 호출 가능 변환 객체 | `build(image_size, train, **params)` |
| `model` | `nn.Module` 서브클래스 | `build(**params)` |
| `loss` | 손실 함수 또는 `nn.Module` | `build(**params)` |
| `metric` | metric 객체 | `build(**params)` |
| `adapter` | `TaskAdapter` 서브클래스 | `build(loss_fn, metrics, **params)` |
| `builder` | optimizer/scheduler 팩토리 | `build(target, **params)` |

```python
from src.core.registry import DATASETS, MODELS

@MODELS.register("resnet50_cls")
class ResNet50Classifier(nn.Module):
    ...
```

- 키는 소문자 `snake_case` 문자열이다. 중복 등록은 `RegistryError`로 즉시 실패하며 조용한 덮어쓰기를 하지 않는다.
- 등록은 `src/tasks/__init__.py`가 태스크 패키지를 임포트할 때 이루어진다. 지연 로딩이나 파일 스캔을
  하지 않는다(오프라인·결정성 유지).
- 조회 실패 시 오류 메시지에 해당 네임스페이스의 등록된 키 목록을 포함한다.
- `NFR-05`(모델 확장성)의 판정 근거는 이 계약이다. 새 모델은 `models/<name>.py` 추가와 `@MODELS.register`,
  config 1개만으로 실행 가능해야 하며 `core`·`bench`·태스크 공통 파일을 수정하지 않는다.

## 5. 실행 컨텍스트

### 5.1. seed와 determinism (`NFR-01`)

`RunContext.setup_seed(seed)`는 다음을 모두 적용한다. 하나라도 누락하면 `NFR-01` 위반이다.

```python
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

추가로 DataLoader 계층에 다음을 적용한다(§8.4).

- `worker_init_fn`: worker마다 `seed + worker_id`로 `random`·`numpy`·`torch` 시드를 재설정
- `generator`: `torch.Generator().manual_seed(seed)`를 shuffle sampler에 주입

`runtime.deterministic` 3단계를 정의한다.

| 값 | 설정 | 용도 |
|---|---|---|
| `strict` | `cudnn.deterministic=True`, `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` | toy·CPU 검증. 비결정 연산 사용 시 오류로 실패 |
| `warn` (기본) | 위와 동일하되 `use_deterministic_algorithms(True, warn_only=True)` | 실데이터 기본값. 결정 커널이 없는 연산은 경고 후 진행 |
| `off` | `cudnn.benchmark=True` | 속도 측정 전용. 비교 실행에는 사용하지 않는다 |

기본값을 `warn`으로 두는 이유는 segmentation의 `interpolate` backward, detection의 산포 연산 등
CUDA 결정 커널이 없는 연산이 존재해 `strict`로는 실행 자체가 불가능하기 때문이다. `warn`에서 발생한
비결정 연산 경고는 `train.log`와 `env.json`에 수집해 기록한다.

### 5.2. 재현성 허용 오차 (`NFR-01`, `AC-08`)

동일 환경·동일 seed·동일 `config.resolved.yaml`로 2회 실행할 때 판정 기준은 다음과 같다.

| 대상 | 허용 오차 | 비고 |
|---|---|---|
| 비율형 metric (accuracy, F1, mIoU, Dice, AUROC, mAP) | 절대 오차 `<= 1e-3` | 값 범위가 [0, 1] |
| loss (epoch 평균) | 상대 오차 `<= 1e-2` | 0 근처에서는 절대 오차 `1e-4` 병용 |
| params | 완전 일치 | 정수 |
| FLOPs | 완전 일치 | 정수 |
| FPS | 판정 제외 | 시간 측정값이므로 재현성 대상이 아님 |
| `deterministic: strict` + `device: cpu` | 비트 단위 완전 일치 | toy 회귀 테스트 기준 |

- P1.9에서 toy 4종을 2회 실행해 실측 편차를 기록한다. 실측이 위 기준을 넘으면 기준을 임의로 완화하지 않고,
  원인(비결정 연산 목록)과 함께 조항을 개정하고 개정 이력(§16)에 남긴다.
- 재현성 판정은 `outputs/.../metrics_final.json` 두 개를 비교하는 스크립트로 수행하며 육안 비교하지 않는다.

### 5.3. device와 AMP (`FR-08`)

- `runtime.device`로 `cuda`/`cpu`를 선택한다. 다중 GPU는 비범위(`OUT-10`)이며 `cuda:0`만 사용한다.
- AMP 기본값은 `false`다. GTX 1080 Ti는 fp16 tensor core가 없어 이득이 없고 재현성에 불리하다
  (`PLAN.md` 6.1).
- AMP가 `true`일 때 `torch.amp.autocast("cuda")`와 `torch.amp.GradScaler`를 사용한다. AMP 상태는
  통제 필드(§10.2)이므로 비교 split 간에 달라질 수 없다.

### 5.4. Logger (`FR-06`)

- 콘솔과 `train.log`에 동일 내용을 출력한다. 포맷은 `[YYYY-mm-dd HH:MM:SS] [LEVEL] message`다.
- epoch 종료 시 `metrics_epoch.csv`에 한 행을 추가한다. 컬럼은
  `epoch, split, loss, <metric 이름들...>, lr, elapsed_sec`이며 split은 `train` 또는 `valid`다.
- 진행 표시는 `log_interval` 스텝마다 한 줄로 출력한다. 진행률 바 라이브러리를 도입하지 않는다.
- 실험 관리 도구(tensorboard, wandb)를 사용하지 않는다(`OUT-02`).

### 5.5. Checkpoint (`FR-05`)

저장 내용은 다음으로 고정한다.

```python
{
    "epoch": int,
    "model_state": OrderedDict,
    "optimizer_state": dict,
    "scheduler_state": dict | None,
    "scaler_state": dict | None,
    "best_metric": float,
    "monitor": {"metric": str, "mode": str},
    "config": dict,          # resolved config
    "env": dict,             # §5.6
    "rng_state": dict,       # python/numpy/torch/cuda RNG 상태
}
```

- `best.pth`는 `train.monitor` 기준 valid metric이 개선될 때만 갱신한다(§7.3).
- `last.pth`는 매 epoch 종료 시 갱신한다.
- 재개 로드는 `rng_state`까지 복원해 동일 seed 실행과 동일한 난수열을 잇는다.
- 체크포인트는 커밋 대상이 아니다(`CON-15`).

### 5.6. 환경 기록 (`FR-27`)

`env.json`에 다음을 기록한다. 재현 조건 확인과 leaderboard의 환경 열에 사용한다.

`python_version`, `torch_version`, `torchvision_version`, `torchmetrics_version`, `cuda_version`,
`gpu_name`, `driver_version`, `hostname`, `git_commit`, `git_dirty`, `command_line`, `seed`,
`deterministic_mode`, `amp`, `started_at`, `finished_at`, `nondeterministic_warnings`.

## 6. TaskAdapter 계약 (`FR-03`, `FR-10`, `FR-11`)

P1의 핵심 산출물이다. 엔진과 태스크의 유일한 접점이며, 이 계약이 불완전하면 P2~P5가 공통 코드를
수정하게 되어 병렬 구조가 무너진다.

### 6.1. 배치 계약

**엔진은 배치를 열어보지 않는다.** 엔진에게 배치는 불투명한 객체이며, 배치의 구조를 아는 것은
`collate_fn`과 `TaskAdapter`뿐이다. 이 규칙 하나가 `FR-02`(task-agnostic)의 실질적 근거다.

| 태스크 | Dataset이 반환하는 샘플 | collate 결과 |
|---|---|---|
| Classification | `(image: Tensor(3,H,W), target: Tensor(scalar, long))` | `(Tensor(B,3,H,W), Tensor(B))` |
| Segmentation | `(image, target: Tensor(H,W) long)` | `(Tensor(B,3,H,W), Tensor(B,H,W))` |
| Detection | `(image, target: {"boxes": Tensor(N,4) float, "labels": Tensor(N) long})` | `(list[Tensor], list[dict])` |
| Anomaly (train) | `(image, target: {})` | `(Tensor(B,3,H,W), list[dict])` |
| Anomaly (eval) | `(image, target: {"label": Tensor(scalar), "mask": Tensor(H,W) 선택})` | `(Tensor(B,3,H,W), list[dict])` |

- Detection의 `boxes`는 절대 좌표 `xyxy`이며 `N=0`일 때 `shape (0, 4)`의 float 텐서, `labels`는
  `shape (0,)`의 long 텐서다. `None`을 쓰지 않는다.
- Detection의 클래스 인덱스는 `0`을 배경으로 예약하고 전경 클래스는 `1`부터 시작한다
  (torchvision detection 관례). `oxford_pets`는 `1=cat`, `2=dog`다.
- 엔진이 배치 크기를 알아야 하는 지점(로그 평균)은 `adapter.batch_size(batch)`가 반환한다.
  엔진이 `len(batch[0])`를 직접 계산하지 않는다.

### 6.2. 인터페이스

```python
class TaskAdapter(ABC):
    """Encapsulates how one batch is forwarded, how loss and predictions are produced,
    and how metrics are updated. The engine knows nothing else about a task."""

    def __init__(self, loss_fn, metrics, **params):
        self.loss_fn = loss_fn
        self.metrics = metrics

    # --- required ---
    @abstractmethod
    def train_step(self, model, batch, device) -> dict:
        """Return {"loss": scalar Tensor with grad, "loss_dict": {str: float}}."""

    @abstractmethod
    def eval_step(self, model, batch, device) -> dict:
        """Return {"loss": scalar Tensor or None, "outputs": Any}."""

    @abstractmethod
    def update_metrics(self, outputs) -> None: ...

    @abstractmethod
    def compute_metrics(self) -> dict: ...

    @abstractmethod
    def reset_metrics(self) -> None: ...

    @abstractmethod
    def predict_step(self, model, batch, device) -> list:
        """Return one serializable prediction per sample in the batch."""

    @abstractmethod
    def batch_size(self, batch) -> int: ...

    # --- optional, default no-op ---
    def collate_fn(self): return None          # None means torch default_collate
    def to(self, device): return self
    def on_fit_start(self, model, loaders, device): pass
    def on_fit_end(self, model, loaders, device): pass
    def on_epoch_start(self, model, epoch): pass
    def on_epoch_end(self, model, epoch, results): pass
    def save_predictions(self, predictions, output_dir): pass
    def visualize(self, batch, predictions, output_dir, max_items): pass
```

### 6.3. 학습 스텝

`train_step`은 디바이스 전송, forward, loss 계산을 모두 책임진다. 엔진은 반환된 `loss`에 대해
backward와 optimizer step만 수행한다.

모델별 학습 방식 분기(`FR-10`)는 어댑터 안에서 처리한다. 규약은 다음과 같다.

```python
def train_step(self, model, batch, device):
    images, targets = self.to_device(batch, device)
    if hasattr(model, "train_step"):          # model-specific training (STFPM, EfficientAD, ...)
        return model.train_step(images, targets)
    outputs = model(images)
    loss = self.loss_fn(outputs, targets)
    return {"loss": loss, "loss_dict": {"loss": float(loss.detach())}}
```

- `model.train_step`을 정의한 모델은 `{"loss", "loss_dict"}` 규약을 그대로 지켜야 한다.
- torchvision detection 모델처럼 학습 모드에서 loss dict를 직접 반환하는 모델은 detection 어댑터가
  이 규약으로 변환한다. 엔진은 그 사실을 모른다.
- **엔진에는 어떤 형태의 모델 이름·태스크 이름 분기도 두지 않는다**(§7.4).

### 6.4. 평가 스텝과 metric 계약 (`NFR-03`)

- `eval_step`은 `model.eval()`과 `torch.no_grad()` 컨텍스트 **안에서 엔진이 호출한다.** 어댑터가
  자체적으로 `no_grad`를 다시 걸 필요는 없으나, 걸어도 무해하다.
- metric 객체는 어댑터가 소유한다. 엔진은 metric 객체를 보지 않고 `reset_metrics` /
  `update_metrics` / `compute_metrics`만 호출한다.
- 엔진은 **매 평가 루프 시작 시 반드시 `reset_metrics()`를 호출한다.** 누락은 `NFR-03` 위반이다.
- `compute_metrics()`는 `{"top1_accuracy": 0.83, ...}` 형태의 float dict를 반환한다. 텐서를 반환하지 않는다.
- metric 이름은 config `metrics[*].name`과 동일해야 한다. `train.monitor.metric` 조회가 이 이름으로
  이루어지기 때문이다.

### 6.5. 훅 (`FR-09`)

표준 gradient 루프를 벗어나는 처리를 흡수하는 지점이다. 엔진은 정해진 시점에 훅을 호출할 뿐
내용을 알지 못한다.

| 훅 | 호출 시점 | v0.1 사용처 |
|---|---|---|
| `on_fit_start(model, loaders, device)` | 첫 epoch 시작 전, optimizer 생성 후 | EfficientAD의 채널 정규화 통계 산출, teacher 가중치 고정 |
| `on_epoch_start(model, epoch)` | 각 epoch의 학습 루프 직전 | 모델별 스케줄 조정 |
| `on_epoch_end(model, epoch, results)` | valid 평가 완료 후 | 모델별 상태 갱신 |
| `on_fit_end(model, loaders, device)` | 마지막 epoch 종료 후, best 로드 전 | Anomaly threshold 결정 (valid만 사용) |

- `loaders`는 `{"train": DataLoader, "valid": DataLoader}`다. **`test`는 포함하지 않는다**(§8.3).
- `OUT-14`(PatchCore 등 memory-bank류)는 v0.1 비범위이나, "학습 없이 `on_fit_start`에서 뱅크를 채우고
  `train.epochs=0`으로 실행"하는 경로가 이 훅으로 표현 가능함을 P1에서 확인한다. 이것이 공통 엔진의
  한계선이며 P6에서 문서화한다.

### 6.6. 예측과 시각화 (`FR-22`, `FR-32`)

- `predict_step`은 배치 하나에 대해 샘플 수만큼의 직렬화 가능한 예측 목록을 반환한다.
- `save_predictions`는 `outputs/.../predictions/`에 JSON 또는 이미지로 저장한다. 형식은 태스크가 정한다.
- `visualize`는 `outputs/.../visualizations/`에 최대 `output.max_visualizations`장을 저장한다.
- 엔진은 저장 경로만 넘기고 내용에 관여하지 않는다.

### 6.7. 금지 사항

- 어댑터가 `optimizer.step()` 또는 `loss.backward()`를 호출하지 않는다. 엔진의 책임이다.
- 어댑터가 `train.epochs`, 체크포인트, 로그 파일에 접근하지 않는다.
- 어댑터가 test split DataLoader를 생성하거나 참조하지 않는다.
- 어댑터가 다른 태스크 패키지를 임포트하지 않는다.

## 7. Trainer 엔진 (`FR-02`, `NFR-04`)

### 7.1. 학습 루프

```python
class Trainer:
    def fit(self, model, adapter, train_loader, valid_loader, optimizer, scheduler, ctx):
        adapter.on_fit_start(model, {"train": train_loader, "valid": valid_loader}, ctx.device)
        for epoch in range(1, ctx.epochs + 1):
            adapter.on_epoch_start(model, epoch)
            self._train_epoch(model, adapter, train_loader, optimizer, ctx)
            if scheduler is not None:
                scheduler.step()
            results = self.evaluate(model, adapter, valid_loader, ctx)
            adapter.on_epoch_end(model, epoch, results)
            self._checkpoint(model, optimizer, scheduler, epoch, results, ctx)
        adapter.on_fit_end(model, {"train": train_loader, "valid": valid_loader}, ctx.device)
```

`_train_epoch` 본문은 다음 순서로 고정한다.

1. `model.train()`
2. `optimizer.zero_grad(set_to_none=True)`
3. `autocast(enabled=ctx.amp)` 안에서 `adapter.train_step(model, batch, ctx.device)`
4. `scaler.scale(loss).backward()`
5. `train.grad_clip`이 `null`이 아니면 `scaler.unscale_(optimizer)` 후 `clip_grad_norm_`
6. `scaler.step(optimizer)`, `scaler.update()`
7. `loss_dict`를 배치 크기 가중 평균으로 누적

scheduler는 epoch 단위로만 step한다. step 단위 scheduler는 v0.1 비범위다.

### 7.2. 평가 루프와 무결성 (`NFR-03`)

```python
def evaluate(self, model, adapter, loader, ctx):
    model.eval()
    adapter.reset_metrics()
    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            outputs = adapter.eval_step(model, batch, ctx.device)
            adapter.update_metrics(outputs)
            ...
    results = adapter.compute_metrics()
    return results
```

- `model.eval()`과 `torch.no_grad()`는 엔진이 강제하며 어댑터 구현에 맡기지 않는다.
- `reset_metrics()`는 루프 진입 전 무조건 호출한다.
- 평가 종료 후 호출자가 학습을 재개하는 경우 `model.train()`은 다음 `_train_epoch`가 다시 설정한다.

### 7.3. 모델 선택과 test 격리 (`AC-10`)

- `train.monitor.metric`은 **valid 결과에서만** 조회한다. `Trainer.fit`의 시그니처에는 test loader가 없다.
- test split은 `evaluate` 서브커맨드에서만 생성한다. `RunContext`는 `allow_test_split` 플래그를 갖고,
  `evaluate` 커맨드에서만 `True`로 설정된다. `False` 상태에서 `split="test"`로 DataLoader를 만들면
  `RuntimeError`로 실패한다(§8.3).
- Anomaly의 threshold 결정은 `on_fit_end` 훅에서 valid loader만으로 수행한다.

### 7.4. 엔진 순수성 규칙과 기계 검사 (`FR-02`, `NFR-04`)

`src/core/`와 `src/bench/`의 어떤 파일도 다음을 포함할 수 없다.

- 태스크 이름 문자열: `classification`, `segmentation`, `detection`, `anomaly`, `toy`
- 모델 이름 문자열: `resnet`, `yolo`, `efficientad`, `stfpm`, `unet` 등
- 배치 구조 가정: `batch[1]["boxes"]`, `isinstance(targets, list)` 등 타깃 형태 분기

이를 검사로 강제한다. `scripts/check_engine_purity.py`가 위 문자열 목록을 `src/core/`와
`src/bench/` 전체에서 검색하고, 하나라도 발견되면 비정상 종료한다. `meta.task_name`은 CLI와
경로 생성부(`src/cli/`)에서만 사용하므로 이 검사에 걸리지 않는다.

이 검사는 P1.9의 수용 기준이며 P2~P5의 공통 코드 변경 시마다 재실행한다(`PLAN.md` 5.4).

## 8. 데이터 계층

### 8.1. Dataset 계약 (`FR-11`)

- Dataset은 `__len__`과 `__getitem__(idx) -> (image, target)`만 제공한다.
- `image`는 transform 적용 후의 `Tensor(3, H, W)` float이며 정규화까지 완료된 상태다.
- `target`은 §6.1 표의 형태를 따른다. 태스크가 같으면 데이터셋이 달라도 형태가 같아야 한다(`NFR-06`).
- Dataset 생성자는 `(root, split, transform, **params)`를 받는다. `split`은 `"train"|"valid"|"test"`다.
- Dataset은 네트워크에 접근하지 않고 다운로드하지 않는다(`CON-03`).

### 8.2. split 선언 형식 (`FR-17`)

두 가지 모드를 지원한다. 비교 실행에서는 `file` 모드를 권장한다.

**`file` 모드** — 명시적 ID 목록. 재현성이 가장 확실하다.

```yaml
data:
  split:
    mode: file
    path: configs/splits/oxford_pets_cls.json
```

```json
{
  "dataset": "oxford_pets",
  "created_at": "2026-08-18",
  "source": "annotations/list.txt + trainval.txt/test.txt",
  "seed": 42,
  "train": ["Abyssinian_1", "Abyssinian_10", "..."],
  "valid": ["Abyssinian_100", "..."],
  "test":  ["Abyssinian_101", "..."]
}
```

**`ratio` 모드** — 비율과 seed로 생성한다. 생성 결과는 실행 시 run 디렉토리에 `split.json`으로 저장한다.

```yaml
data:
  split:
    mode: ratio
    ratio: {train: 0.7, valid: 0.15, test: 0.15}
    seed: 42
    stratify_by: label     # null 이면 무작위
```

- 샘플 ID는 원본 이미지 파일의 stem이다. 태스크가 달라도 같은 이미지면 같은 ID다.
- `configs/splits/*.json`은 커밋 대상이다(수 KB).

### 8.3. 상호 배타성과 test 격리 (`AC-10`)

`src/data/split.py`가 다음을 강제한다.

- `assert_disjoint(split_dict)`: `train`/`valid`/`test` ID 집합의 쌍별 교집합이 비어 있지 않으면
  교집합 샘플 수와 예시 ID 5개를 포함한 오류로 실패한다. Dataset 생성 시 매번 호출한다.
- 세 split의 합집합이 원본 인덱스의 부분집합인지 확인하고, 알 수 없는 ID가 있으면 실패한다.
- `build_dataloader(..., split="test")`는 `ctx.allow_test_split`가 `True`일 때만 허용한다(§7.3).

### 8.4. DataLoader 빌더

```python
DataLoader(
    dataset,
    batch_size=data.batch_size,
    shuffle=(split == "train"),
    num_workers=data.num_workers,
    collate_fn=adapter.collate_fn(),      # None 이면 default_collate
    worker_init_fn=make_worker_init_fn(seed),
    generator=torch.Generator().manual_seed(seed),
    drop_last=(data.drop_last and split == "train"),
    pin_memory=(device == "cuda"),
    persistent_workers=(num_workers > 0),
)
```

`shuffle`은 `train`에서만 참이다. `valid`/`test`는 항상 순서를 고정한다.

## 9. 오프라인과 로컬 자산

### 9.1. 오프라인 가드 (`NFR-07`, `AC-07`, `CON-03`)

`src/core/offline.py`의 `enable_offline_guard()`를 모든 서브커맨드 진입 직후 호출한다.

- `socket.socket.connect`를 래핑해 loopback(`127.0.0.1`, `::1`)과 AF_UNIX 외의 연결 시도에서
  `OfflineViolationError`를 발생시킨다. 어떤 코드 경로가 네트워크를 시도했는지 스택과 함께 보고한다.
- `TORCH_HOME`, `HF_HOME`, `HF_HUB_OFFLINE=1`, `YOLO_OFFLINE` 등 환경 변수를 로컬 값으로 설정한다.
- `runtime.allow_network: true`일 때만 가드를 해제한다. 기본값은 `false`이며 v0.1의 어떤 config에서도
  `true`로 두지 않는다.

이 가드가 `AC-07`(네트워크 차단 상태 완주)의 상시 검증 수단이다. P6에서 별도 차단 환경을 만들지 않아도
모든 실행이 이미 가드 아래에서 돌아간다.

### 9.2. 로컬 가중치 로더 (`CON-04`, `CON-05`)

```python
def load_local_weights(model, weights_path, strict=True, map_location="cpu", key_map=None):
    """Load a local .pth/.pt checkpoint into model. Never downloads."""
```

- `weights_path`가 없으면 `LocalAssetError`로 즉시 실패한다. 메시지에 기대 경로, 조치 방법
  (해당 파일을 `/mnt/d/backbones`에 두라는 안내)을 포함한다. 무작위 초기화로 조용히 폴백하지 않는다.
- torchvision 모델은 `weights=None`으로 생성한 뒤 이 함수로 주입한다.
- `strict=False`가 필요한 경우(분류 head 교체 등) 누락·초과 키 목록을 로그에 남긴다. 예상 밖의 키가
  누락되면 오류로 승격한다.
- ultralytics YOLO와 anomalib 유래 모델도 동일하게 로컬 파일만 사용한다(`CON-05`).

### 9.3. 자산 점검

`configs/assets.yaml`에 v0.1이 사용하는 로컬 자산을 선언하고, `check-assets` 서브커맨드가 전수 확인한다.

```yaml
datasets:
  oxford_pets:
    root: /mnt/d/datasets/oxford_pets
    require: [images, annotations/trimaps, annotations/xmls, annotations/list.txt]
  mvtec_bottle:
    root: /mnt/d/datasets/mvtec/bottle
    require: [train/good, test, ground_truth]
weights:
  resnet50: /mnt/d/backbones/resnet50-0676ba61.pth
  efficientnet_b0: /mnt/d/backbones/efficientnet_b0_rwightman-7f5810bc.pth
  deeplabv3_resnet50: /mnt/d/backbones/deeplabv3_resnet50_coco-cd0a2569.pth
  fcn_resnet50: /mnt/d/backbones/fcn_resnet50_coco-1167a1af.pth
  fasterrcnn_resnet50_fpn: /mnt/d/backbones/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
  yolov8n: /mnt/d/backbones/yolov8n.pt
  efficientad_teacher_small: /mnt/d/backbones/efficientad_pretrained_weights/pretrained_teacher_small.pth
  resnet18: /mnt/d/backbones/resnet18-f37072fd.pth
  wide_resnet50_2: /mnt/d/backbones/wide_resnet50_2-95faca4d.pth
```

`check-assets`는 존재 여부와 파일 크기를 표로 출력하고, 하나라도 없으면 비정상 종료한다.

## 10. 벤치마크

### 10.1. benchmark config (`FR-23`, `FR-28`)

split 선언 형식은 **base config + split별 override** 방식으로 확정한다. split마다 완전한 config 파일을
나열하는 방식은 통제 필드가 파일마다 흩어져 기계 검사와 유지가 어렵기 때문에 채택하지 않는다.

```yaml
# configs/benchmarks/classification_baseline.yaml
name: cls_baseline
task_name: classification
base: configs/classification/_base.yaml

control:
  exceptions:
    - split: efficientnet_b0
      field: data.batch_size
      value: 16
      reason: "GPU memory 11.8GB"
      approved_by: user
      approved_at: "2026-08-18"

splits:
  - name: custom_cnn
    override:
      model: {name: custom_cnn, params: {num_classes: 37}}
  - name: resnet50
    override:
      model:
        name: resnet50_cls
        params: {num_classes: 37, backbone_name: resnet50, weights_path: /mnt/d/backbones/resnet50-0676ba61.pth}
  - name: efficientnet_b0
    override:
      model:
        name: efficientnet_b0_cls
        params: {num_classes: 37, backbone_name: efficientnet_b0, weights_path: /mnt/d/backbones/efficientnet_b0_rwightman-7f5810bc.pth}
      data: {batch_size: 16}
```

각 split의 최종 config는 `base` 병합 → `override` 병합 → CLI `--set` 병합으로 만들어지며,
`outputs/benchmarks/<name>/splits/<split>/config.resolved.yaml`에 저장된다.

### 10.2. 통제 필드 목록 (`FR-35`, `NFR-02`)

비교 그룹의 모든 split에서 값이 동일해야 하는 필드는 다음으로 확정한다.

| 통제 필드 | 근거 |
|---|---|
| `runtime.seed` | BRIEF 7 |
| `runtime.amp` | 수치 정밀도가 결과에 영향 |
| `runtime.deterministic` | 재현 조건 통일 |
| `data.name`, `data.root`, `data.params` | 동일 데이터 |
| `data.split` | 동일 분할 (누수·난이도 차이 방지) |
| `data.image_size` | BRIEF 7 (입력 해상도) |
| `data.transform` | BRIEF 7 (정규화·augmentation) |
| `data.num_workers`, `data.drop_last` | worker 시드와 배치 수가 재현성에 영향 |
| `data.batch_size` | BRIEF 7. 모델 크기 사유의 예외만 허용 |
| `loss` | 동일 학습 목표 |
| `metrics` | 동일 평가 기준 |
| `adapter` | 동일 파이프라인 |
| `optim.optimizer`, `optim.scheduler` | BRIEF 7 |
| `train.epochs`, `train.monitor`, `train.grad_clip` | BRIEF 7 |

자유 축(비교 대상)은 `model`과 `meta`, `output`이다. 그 외 필드를 자유 축으로 두려면 `control.exceptions`에
`split`, `field`, `value`, `reason`, `approved_by`, `approved_at`를 명시해야 한다.

### 10.3. 통제 검사 절차

`src/bench/control.py`가 다음을 수행한다.

1. 모든 split의 resolved config를 만든다(학습 시작 **전**에 전부 만든다).
2. §10.2의 각 통제 필드를 split 간 깊은 비교한다.
3. 차이가 발견되면 `control.exceptions`에 `(split, field)`가 있고 선언된 `value`와 실제 값이 일치하는지 확인한다.
4. 승인되지 않은 차이가 하나라도 있으면 `ControlViolationError`로 **어떤 split도 실행하지 않고** 종료한다.
5. 검사 결과를 `control_report.json`에 기록한다. 필드별 값, 위반 여부, 적용된 예외를 포함한다.
6. `leaderboard` 서브커맨드는 실행 시 각 run 디렉토리의 `config.resolved.yaml`을 **다시 읽어** 2~4를
   재수행한다. 통과하지 못하면 leaderboard를 생성하지 않는다.

6번이 있으므로 벤치마크 실행 경로를 우회해 개별 `train`으로 만든 산출물을 모아도 통제 검사를 피할 수 없다.

### 10.4. 순차 실행 (`FR-24`)

`benchmark` 서브커맨드는 통제 검사 통과 후 split을 선언 순서대로 실행한다. split 하나의 처리 순서는
`학습 → valid 기준 best 선택 → test 평가 → 추론 산출물 생성 → 프로파일 측정`이다.

- split 하나가 실패해도 나머지를 계속 실행하고, 실패 사실을 `control_report.json`과 leaderboard의
  `status` 열에 기록한다. 실패한 split은 metric 열을 비운다.
- `--only <split1,split2>`로 일부만 재실행할 수 있다. 이때도 통제 검사는 전체 split에 대해 수행한다.

### 10.5. 프로파일 (`FR-26`)

| 지표 | 측정 방법 |
|---|---|
| `params_total`, `params_trainable` | `sum(p.numel() ...)`. 정수 |
| `flops_g` | `torch.utils.flop_counter.FlopCounterMode`, `model.eval()`, `no_grad`, batch=1, `data.image_size` 입력. 단위 GFLOPs |
| `fps` | `model.eval()`, `no_grad`, batch=1, warmup 10회 후 50회 측정. 각 반복 전후 `torch.cuda.synchronize()`. 중앙값 지연시간의 역수 |

- `fps`는 **모델 forward만** 측정하며 데이터 로딩과 후처리를 포함하지 않는다. leaderboard 각주에 명시한다.
- 측정 실패(예: FlopCounterMode 미지원 연산)는 `null`로 기록하고 사유를 로그에 남긴다. 전체 실행을
  중단하지 않는다.
- `fps`는 재현성 판정 대상이 아니다(§5.2).

### 10.6. leaderboard (`FR-25`)

`leaderboard.csv`와 `leaderboard.md`를 생성한다. 열 구성은 다음으로 고정한다.

```text
split, model, status, <task metrics...>, params_total, flops_g, fps,
best_epoch, train_time_sec, batch_size, image_size, seed, control_status, exceptions, run_dir
```

- 정렬 기준은 `train.monitor.metric`과 `mode`다.
- Markdown 표 아래에 통제 조건 요약(§10.2의 통제 필드 값)과 승인된 예외 목록, `fps` 측정 조건 각주,
  "v0.1의 목적은 파이프라인 검증이며 절대 성능이 아니다"라는 문구를 함께 출력한다.
- 태스크 간 비교는 하지 않는다(`OUT-01`). leaderboard는 벤치마크 1개 = 태스크 1개 단위로만 생성한다.

## 11. CLI (`FR-29` ~ `FR-33`)

단일 진입점은 `python -m src <subcommand>`다.

| 서브커맨드 | 형식 | 동작 |
|---|---|---|
| `check-assets` | `check-assets [--assets configs/assets.yaml]` | 로컬 데이터셋·가중치 전수 확인 (§9.3) |
| `config` | `config <config.yaml> [--set ...]` | resolve + 검증 후 최종 config를 출력. 실행하지 않음 |
| `train` | `train <config.yaml> [--set ...] [--resume <ckpt>]` | 학습. train/valid만 사용 |
| `evaluate` | `evaluate <config.yaml> --checkpoint <ckpt> [--split test]` | 평가. `allow_test_split=True` |
| `predict` | `predict <config.yaml> --checkpoint <ckpt> --input <path> [--output <dir>]` | 임의 이미지·디렉토리 추론 |
| `benchmark` | `benchmark <bench.yaml> [--set ...] [--only a,b] [--overwrite]` | 통제 검사 후 split 순차 실행 |
| `leaderboard` | `leaderboard <bench_output_dir>` | 통제 재검사 후 CSV/Markdown 생성 |

공통 옵션은 `--set`(반복), `--log-level`이다. 모든 서브커맨드는 진입 직후 오프라인 가드를 켠다(§9.1).

## 12. toy 태스크 4종 fixture (`PLAN.md` 3.2.1)

`src/tasks/toy/`에 합성 데이터 기반 4종을 구현한다. 실데이터·실모델 없이 4가지 타깃 형태를
모두 통과시키는 것이 목적이며, P2~P5의 회귀 테스트 세트가 된다(`PLAN.md` 5.4).

| fixture | Dataset | 타깃 | 모델 | metric | 검증 대상 |
|---|---|---|---|---|---|
| `toy_cls` | 3x32x32 합성 이미지, 4 클래스 | 스칼라 long | 3-layer CNN | accuracy | 스칼라 라벨, 기본 경로 |
| `toy_seg` | 3x32x32 | `(32,32)` long, 3 클래스 | 소형 encoder-decoder | mIoU | dense 라벨, 픽셀 metric |
| `toy_det` | 3x64x64 | `{"boxes": (N,4), "labels": (N,)}`, `N ∈ {0,1,2,3}` 순환, 클래스 `{1,2}` | 소형 anchor-free head | mAP@0.5:0.95 | 가변 N(N=0 포함), list-of-dict, 태스크별 `collate_fn` |
| `toy_anomaly` | train=정상만, test=정상+이상+mask | train `{}`, eval `{"label", "mask"}` | 소형 AE + `train_step` 보유 모델 1종 | image/pixel AUROC | 타깃 없는 학습, 모델별 `train_step`, `on_fit_start`/`on_fit_end` 훅 |

- 합성 데이터는 seed로부터 결정적으로 생성한다. 파일을 디스크에 만들지 않는다.
- `toy_det`은 반드시 한 배치 안에 `N=0`, `N=1`, `N>1` 샘플이 섞이도록 구성한다(`AC-09`).
- `toy_anomaly`는 모델 2종을 제공한다. 하나는 표준 loss 경로, 하나는 `model.train_step`을 정의한
  비표준 경로다. 두 경로가 같은 엔진에서 동작해야 `FR-10`이 검증된다.
- `configs/toy/`에 4종의 단일 실행 config와 벤치마크 config 1개(`toy_cls` 3모델)를 둔다.

## 13. 스모크 규격 (`NFR-12`)

epoch은 전 Phase 공통 5로 고정한다(`PLAN.md` 4). 최대 실행 시간은 모델 1종 기준이며 GTX 1080 Ti 단일
GPU를 전제한다. 이 시간을 초과하면 subset 크기를 줄이는 대신 원인을 기록한다.

| Phase | fixture / 데이터 | train / valid / test | image_size | batch | 최대 실행 시간 |
|---|---|---|---|---|---|
| P1 | toy_cls | 64 / 32 / 32 | 32x32 | 8 | 60초 |
| P1 | toy_seg | 64 / 32 / 32 | 32x32 | 8 | 60초 |
| P1 | toy_det | 64 / 32 / 32 | 64x64 | 4 | 120초 |
| P1 | toy_anomaly | 64 / 32 / 32 | 32x32 | 8 | 60초 |
| P2 | oxford_pets 37 breeds | 370 / 74 / 74 (breed당 10/2/2) | 224x224 | 32 | 10분 |
| P3 | oxford_pets trimap | 296 / 74 / 74 (breed당 8/2/2) | 256x256 | 8 | 15분 |
| P4 | oxford_pets xmls | 296 / 74 / 74 (breed당 8/2/2) | 512x512 | 4 | 25분 |
| P5 | mvtec bottle | train good 209 / valid·test는 §13.1 | 256x256 | 8 | 15분 |

toy 4종은 CPU에서도 완주해야 한다. `deterministic: strict` + `device: cpu` 조합이 P1의 재현성
회귀 테스트 기준이다(§5.2).

### 13.1. MVTec bottle의 split 정책

MVTec에는 공식 valid split이 없다. 실측 구성은 `train/good` 209장, `test` 83장(good 20,
broken_large 20, broken_small 22, contamination 21), `ground_truth` 63장이다.

- threshold 결정과 모델 선택은 valid만 사용해야 하므로(`NFR-03`), valid에는 이상 샘플이 필요하다.
- 따라서 `test`를 결함 유형별 층화 추출로 valid와 test로 나눈다. 비율과 seed는 `PLAN-P5`에서 확정하며,
  두 집합은 §8.3의 상호 배타성 검사를 통과해야 한다.
- `train/good`은 전량 학습에 사용한다.
- 이 정책은 P1에서 확정한 계약(valid 전용 threshold, 상호 배타 split)의 적용이며, 구체 수치만
  `PLAN-P5`로 위임한다.

## 14. 하위 단계와 수용 기준

`PLAN.md` 3.2.2의 P1.1~P1.9를 수용 기준과 함께 상세화한다.

| 단계 | 산출물 | 수용 기준 |
|---|---|---|
| P1.1 | 패키지 스켈레톤, `requirements.txt`, `.gitignore`, `git init` + remote 연결, `offline.py`, `load_local_weights`, `configs/assets.yaml`, `check-assets` | `check-assets`가 §9.3 전 항목 통과. `faster-coco-eval` 설치 확인 |
| P1.2 | `config.py`, `registry.py`, run 디렉토리 규칙 | 합성 config의 상속·`--set` override·검증·경로 생성 통과. 존재하지 않는 `--set` 키가 오류로 실패 |
| P1.3 | `context.py`, `logger.py`, `checkpoint.py` | 동일 seed 2회 실행이 동일 난수열. 체크포인트 저장·재개가 RNG 상태까지 복원 |
| P1.4 | `adapter.py`(추상), toy 4종 fixture(dataset·model·adapter·metric·collate) | 엔진 없이 toy 4종이 Dataset → collate → adapter → metric까지 통과. toy_det 배치에 N=0/1/>1 혼재 |
| P1.5 | `engine.py`, `builders.py`, 훅 | toy 4종 학습·평가 5 epoch 완주. `check_engine_purity.py` 통과 |
| P1.6 | `cli/` 전체 서브커맨드 | toy로 `config`/`train`/`evaluate`/`predict` 완주. `train` 경로에서 test split 접근이 오류로 차단됨 |
| P1.7 | `bench/profile.py` | toy 모델에서 params/FLOPs/FPS 3지표 산출 |
| P1.8 | `bench/runner.py`, `control.py`, `leaderboard.py` | toy 3모델 split의 leaderboard 생성. 통제 위반 config가 **학습 시작 전** 실패로 종료. `leaderboard` 재검사도 실패 |
| P1.9 | 재현성 실측, 적대적 검증, 조항 확정 | toy 4종 2회 실행이 §5.2 허용 오차 내 일치. Codex CLI 적대적 검증에서 미해결 Critical 없음 |

## 15. 적대적 검증 초점

`PLAN.md` 3.2.3의 공격 축을 이 문서의 조항에 대응시킨다. `backlog.json`의 P1 `adversarialFocus`와
`planRefs`가 이 표를 그대로 사용한다.

| 축 | 공격 내용 | 대응 조항 |
|---|---|---|
| 계약 일반성 | `TaskAdapter`로 N=0 포함 가변 N 배치를 표현할 수 있는가. 타깃 없는 학습과 모델별 `train_step` 분기를 어댑터·훅만으로 처리할 수 있는가 | §6.1, §6.3, §6.5, §12 |
| 엔진 순수성 | 공통 루프에 태스크 이름·타깃 형태 분기가 있는가. toy 통과를 위해 엔진에 특례를 넣었는가 | §6.1, §7.1, §7.4, §2.2 |
| 평가 무결성 | `model.eval()`·`no_grad()` 적용 범위, metric 리셋 누락, valid/test 경로 혼용 | §6.4, §7.2, §7.3, §8.3 |
| 재현성 | seed 적용 범위, config 저장이 실제 재현을 보장하는가, 허용 오차가 근거 있는가 | §5.1, §5.2, §5.5, §8.4 |
| 통제 검사 | 통제 필드 비교를 우회할 수 있는가. override로 통제 필드를 바꿔도 통과하는가 | §3.3, §10.2, §10.3 |
| 오프라인 | 네트워크 접근을 유발하는 경로가 남아 있는가 | §9.1, §9.2, §8.1 |

## 16. 조항 개정 이력

`PLAN.md` 5.5에 따라 공통 계약 조항의 변경을 이 절에 기록한다. 등급은 `PLAN.md` 5.2를 따른다.

| 일자 | 조항 | 등급 | 변경 내용 | 요청자 | 승인 |
|---|---|---|---|---|---|
| 2026-08-18 | 전체 | — | 최초 작성 | master | — |
| 2026-08-19 | `src/tasks/__init__.py` | A | `detection` 태스크 패키지를 임포트 목록에 추가 (`classification, detection, segmentation, toy`). classification/segmentation과 동일한 기계적 등록 패턴이며 다른 태스크의 동작에 영향 없음 | P4 task 에이전트 | master (2026-08-19) |
| 2026-08-19 | `src/bench/control.py` §10.2, `src/bench/runner.py`, `src/cli/commands.py` | B | `build_control_report`/`enforce_control`에 `extra_fields` 선택 인자를 추가해 통제 필드 목록을 `CONTROL_FIELDS + extra_fields`로 일반화. `runner.py`(benchmark 실행)와 `commands.py`(leaderboard 재생성)가 bench yaml의 `control.extra_fields`를 읽어 전달하도록 배선. `PLAN-P4 §8.3`이 사전 지정한 등급 B 계약 확장(`model.params.score_thresh/nms_iou/max_det`을 P4 한정 통제 필드로 승격)을 구현. 태스크 이름 분기 없음 — `extra_fields`가 비어 있으면 기존 태스크는 동작 불변 | P4 task 에이전트 | master (2026-08-19) |
| 2026-08-19 | `src/bench/leaderboard.py` | B | `build_leaderboard()`의 `fieldnames`를 고정 목록에서 `core_fields`(기존 목록) + `extra_fields`(각 row에 실제로 존재하는, `core_fields`에 없는 키를 정렬해 자동 수집)로 일반화. Detection의 `map_50_95` 메트릭 하나가 `map_50_95`/`map_50`/`map_75` 3개 leaderboard 컬럼을 보고하기 때문에(`PLAN-P4 §6`) 기존 고정 스키마로는 `save_csv_rows`가 `ValueError`로 실패했음. P1-P3는 `compute_metrics()`가 등록된 메트릭 이름 외의 키를 반환하지 않으므로 `extra_fields`가 항상 빈 집합 — 기존 leaderboard 출력 불변 확인. 태스크 이름 분기 없음 | P4 task 에이전트 | master (2026-08-19) |
| 2026-08-19 | `src/core/adapter.py`, `src/tasks/detection/adapter.py`, `src/bench/profile.py`, `src/bench/runner.py` | B | `TaskAdapter`에 `dummy_forward_input(image_size, device)` 훅 추가. 기본 구현은 기존 `profile_model`이 인라인으로 만들던 배치 Tensor(`torch.zeros(1,3,H,W)`)를 그대로 반환해 P1-P3 프로파일링 출력이 불변임을 검증(classification 재실행으로 확인). `DetectionAdapter`는 `forward(images: list[Tensor(3,H,W)])` 계약(`PLAN-P4 §4.1`)에 맞춰 `[torch.zeros(3,H,W)]`를 반환하도록 override. `profile.py`의 `measure_flops`/`measure_fps`/`profile_model`이 인라인 Tensor 생성 대신 `adapter.dummy_forward_input(...)`을 호출하도록 변경, `runner.py`의 호출부도 `adapter`를 전달하도록 갱신. `collate_fn`/`batch_size`와 동일하게 태스크별 호출 규약을 어댑터로 라우팅하는 기존 패턴을 따름 — 공통 코드에 태스크 이름 분기 없음 | P4 task 에이전트 | master (2026-08-19) |
| 2026-08-19 | `src/core/adapter.py`, `src/tasks/detection/adapter.py`, `src/cli/commands.py` | B | P4.5 Codex 적대적 검증(Major)에서 `predict()` 경로가 `DetectionAdapter.__init__`의 기본값(`["background","cat","dog"]`)을 그대로 쓰는 문제가 지적됨 — `train`/`evaluate`는 `bind_class_names(adapter, dataset)`으로 `dataset.classes`를 바인딩하지만 `predict`는 Dataset을 만들지 않아 이 경로를 타지 않았음(`PLAN-P4 §2.1` 일반성 요구 위반). `TaskAdapter`에 `bind_class_names_from_config(data_config)` 훅(기본 no-op) 추가, `DetectionAdapter`가 override해 `data.params.class_names`로부터 `["background"] + class_names`를 구성. `commands.py::predict()`가 어댑터 생성 직후 이 훅을 호출하도록 배선. `train`/`evaluate` 경로는 기존 `bind_class_names()` 흐름을 그대로 유지하며 영향 없음. 공통 코드에 태스크 이름 분기 없음 | master (Codex A4 재검토 반영) | master (2026-08-19) |
| 2026-08-19 | `src/core/adapter.py`, `src/bench/control.py`, `src/bench/leaderboard.py` | A | `scripts/check_engine_purity.py`가 이 세 파일의 주석·docstring에 남아 있던 `classification`/`segmentation`/`detection` 문자열(실제 분기 로직 아님, 설명용 예시 언급)을 금지어로 검출해 실패. 계약·동작 변경 없이 문구만 태스크 이름을 언급하지 않도록 재작성. 재실행 결과 `check_engine_purity.py` PASSED | master | master (2026-08-19) |

---

*작성일: 2026-08-18 · 버전: v0.1 · 상위 문서: PLAN.md · 다음 단계: plans/PLAN-P2-classification.md*
