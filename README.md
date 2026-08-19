# cv_boilerplate

## 1. 프로젝트 개요와 목적

`cv_boilerplate`는 Classification / Segmentation / Object Detection / Anomaly Detection 4개
컴퓨터 비전 태스크를 하나의 task-agnostic 엔진 위에서 학습·평가·벤치마크하는 pure-PyTorch
boilerplate다. 각 태스크는 3개 모델(커스텀 구현 1종 + 공개 아키텍처 2종)을 동일한 데이터·해상도·
augmentation·optimizer·epoch·seed 조건에서 비교한다.

**비교는 태스크 내부에서만 의미가 있다.** 예를 들어 Classification의 3모델 비교는 유효하지만,
Classification의 정확도와 Segmentation의 mIoU를 직접 비교하는 용도가 아니다. 태스크 간 성능 비교,
프로덕션 배포, 대규모 학습은 이 프로젝트의 범위 밖이다.

## 2. 요구 환경

- OS: WSL2(Linux), 셸: bash
- conda 환경: `pytorch_env` (Python 실행 전 `conda activate pytorch_env`)
- PyTorch 2.5.1+cu121, torchvision, torchmetrics
- GPU: 단일 CUDA 디바이스(`cuda:0`). 다중 GPU는 비범위
- 참조 개발 환경: GTX 1080 Ti (Pascal 세대, AMP fp16 tensor core 없음 → `runtime.amp` 기본값 `false`)

## 3. 로컬 자산 준비

이 프로젝트는 데이터셋과 사전학습 가중치를 **자동으로 다운로드하지 않는다.** 아래 경로에 미리
배치해야 한다.

- 데이터셋: `/mnt/d/datasets/{oxford_pets, mvtec}/...`
- 백본 가중치: `/mnt/d/backbones/*.pth`, `/mnt/d/backbones/*.pt`

준비 상태는 다음 명령으로 확인한다.

```bash
python -m src check-assets
```

전 자산이 `OK`로 출력되어야 하며, 하나라도 누락되면 어떤 학습도 무작위 초기화로 대체되지 않고
`LocalAssetError`(또는 config 검증 단계의 `ConfigError`)로 즉시 실패한다. 모든 실행은 시작 시
네트워크 접근을 차단하는 오프라인 가드를 켠다(`runtime.allow_network: false`가 기본값이며, 이를
우회해도 가드가 소켓 연결·DNS 조회를 막는다).

## 4. 빠른 시작

### 4.1. toy 태스크로 파이프라인 확인 (실데이터 불필요)

```bash
python -m src train configs/toy/toy_cls.yaml
python -m src train configs/toy/toy_seg.yaml
python -m src train configs/toy/toy_det.yaml
python -m src train configs/toy/toy_anomaly.yaml
```

toy 4종은 합성 데이터로 CPU에서도 수 초 내에 완주하며, 4가지 target 규약(스칼라 라벨, dense 라벨,
가변 N 박스, 타깃 없는 학습)을 모두 검증한다.

### 4.2. 실태스크 학습 → 평가 → 추론

```bash
python -m src train configs/classification/resnet50.yaml
python -m src evaluate configs/classification/resnet50.yaml \
    --checkpoint outputs/runs/classification/<run_name>/checkpoints/best.pth --split test
python -m src predict configs/classification/resnet50.yaml \
    --checkpoint outputs/runs/classification/<run_name>/checkpoints/best.pth \
    --input /path/to/images --output outputs/runs/classification/<run_name>/predict
```

### 4.3. 벤치마크와 leaderboard

```bash
python -m src benchmark configs/benchmarks/classification_baseline.yaml --overwrite
```

태스크당 3모델을 순차 실행하고 통제 조건(seed·해상도·augmentation·optimizer 등)이 동일한지 학습
시작 전에 기계적으로 검사한다. 통제 위반 config는 어떤 split도 실행하지 않고 즉시 실패한다.
결과는 `outputs/benchmarks/<bench_name>/leaderboard.{csv,md}`에 생성된다.

## 5. CLI 레퍼런스

단일 진입점은 `python -m src <subcommand>`다.

| 서브커맨드 | 형식 | 동작 |
|---|---|---|
| `check-assets` | `check-assets [--assets configs/assets.yaml]` | 로컬 데이터셋·가중치 전수 확인 |
| `config` | `config <config.yaml> [--set ...]` | resolve + 검증 후 최종 config 출력 (미실행) |
| `train` | `train <config.yaml> [--set ...] [--resume <ckpt>]` | 학습. train/valid만 접근 |
| `evaluate` | `evaluate <config.yaml> --checkpoint <ckpt> [--split test]` | 평가. test split 접근 허용 |
| `predict` | `predict <config.yaml> --checkpoint <ckpt> --input <path> [--output <dir>]` | 임의 이미지/디렉토리 추론 |
| `benchmark` | `benchmark <bench.yaml> [--set ...] [--only a,b] [--overwrite]` | 통제 검사 후 모델별 split 순차 실행 |
| `leaderboard` | `leaderboard <bench_output_dir>` | 통제 재검사 후 CSV/Markdown 재생성 |

공통 옵션: `--set key.path=value`(반복 가능), `--log-level`. `test` split은 `evaluate`·`benchmark`
경로에서만 접근할 수 있으며, `train`·`predict`가 이를 시도하면 오류로 차단된다.

## 6. Config 구조

config는 단일 YAML이며 최상위 키 9개(`meta`, `runtime`, `data`, `model`, `loss`, `metrics`,
`adapter`, `optim`, `train`, `output`)로 고정된다. `_base` 키로 다른 config를 상속하며(최대 3단계,
순환 참조 금지), 병합은 dict에 대해 재귀적이고 list·스칼라는 자식 값이 부모를 통째로 대체한다.
병합 순서는 `_base` → 자기 자신 → benchmark split override → CLI `--set` 순이며 뒤에 적용되는 값이
이긴다.

```bash
python -m src train configs/classification/resnet50.yaml \
    --set train.epochs=5 \
    --set optim.optimizer.params.lr=0.0005 \
    --set data.image_size='[128, 128]'
```

`{name, params}` 쌍은 registry 조회 규격이며 `data`(dataset), `model`, `loss`, `metrics`, `adapter`
전 계층에서 동일한 형태를 쓴다.

## 7. 새 모델 추가 방법

1. `src/tasks/<task>/models/<name>.py`에 모델을 구현하고 `@MODELS.register("<name>")`로 등록한다.
2. `configs/<task>/<name>.yaml`을 새로 만들어 `model.name`/`model.params`를 지정한다.
3. 사전학습 가중치가 필요하면 `weights_path`에 `/mnt/d/backbones/`의 로컬 경로를 지정한다.
   `weights=None`으로 모델을 생성한 뒤 `load_state_dict`로 주입하는 패턴을 따른다.

공통 엔진·CLI·벤치마크 코드는 수정하지 않는다. registry 등록과 config 파일 1개만으로 새 모델이
기존 학습·평가·벤치마크 흐름에 편입된다.

## 8. 새 데이터셋 추가 방법

1. `src/tasks/<task>/dataset.py`(또는 신규 태스크면 `src/tasks/<new_task>/`)에 `Dataset`을 구현하고
   `@DATASETS.register("<name>")`로 등록한다. 반환하는 target 형태는 태스크별 공통 규약(스칼라 라벨/
   dense 라벨/`{"boxes","labels"}`/타깃 없음)을 따라야 하며, 모델·Loss·Metric은 이 규약에만 의존한다.
2. `configs/splits/<name>.json`에 `train`/`valid`/`test` ID 목록을 결정적으로 생성해 둔다. 세
   집합은 반드시 상호 배타적이어야 하며(`scripts/check_split_integrity.py`로 검사), 정본 split을
   재사용할 수 없는 사정이 있다면 `note` 필드에 사유를 남긴다.
3. `configs/assets.yaml`에 데이터 경로를 추가해 `check-assets`가 확인하도록 한다.

## 9. 결과 해석 시 유의사항

모든 벤치마크·스모크 결과는 **5 epoch, 축소 subset** 규격으로 산출된다(`classification` 370/74/74,
`segmentation`/`detection` 296/74/74, `anomaly` train good 209 / valid·test 분할). 이는 파이프라인
정합성 검증이 목적이며, 절대적인 모델 성능이나 실제 배포 가능 수준의 정확도를 의미하지 않는다.
leaderboard의 수치를 논문 재현 성능이나 프로덕션 기준으로 사용하지 않는다.

## 10. 한계와 다음 버전

공통 엔진이 4개 태스크를 거치며 실제로 흡수하지 못한 경계, 발생한 등급 B 계약 변경 목록,
알려진 채택 리스크는 [`docs/dev/v0.1/LIMITS.md`](docs/dev/v0.1/LIMITS.md)에 기록되어 있다. v0.2를
시작하기 전에 이 문서를 먼저 읽는다.
