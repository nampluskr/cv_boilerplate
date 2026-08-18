# PLAN-P6 — 통합 검증·마무리

## 1. 문서 목적

이 문서는 `PLAN.md` 3.7의 P6(통합 검증·마무리)를 실행 가능한 검증 절차로 상세화한다. 절 번호
(`PLAN-P6 §4.2` 형식)가 적대적 검증과 `backlog.json`이 참조하는 조항 ID다.

- 범위: 4태스크 전체 재실행, 재현성·오프라인·누수 검증, `AC-01`~`AC-10` 대조표, 공통 엔진 한계선 문서화, README
- 비범위: 새 기능 구현. P6에서는 결함 수정 외의 기능을 추가하지 않는다
- 상위 문서: `PLAN.md` 3.7, `PLAN-P1`~`PLAN-P5`, `PRD.md` 7장

master 단독 수행이며 `PLAN.md` 2.3의 태스크 Phase 템플릿을 따르지 않는다. 검증 항목의 나열이다.
적대적 검증 **필수** Phase다(`PLAN.md` 3.1).

관련 요구사항: `AC-01`~`AC-10`, `FR-09`, `NFR-01`, `NFR-07`, `NFR-09`.

## 2. 전제 조건

P6 착수 전에 다음이 모두 충족되어야 한다. 하나라도 미충족이면 P6를 시작하지 않는다.

- `backlog.json`의 P1~P5 `status`가 모두 완료다.
- P1·P4·P5의 적대적 검증에서 미해결 Critical이 없다(`PLAN.md` 3.1).
- 마지막 공통 코드 변경 이후 완료된 모든 태스크의 스모크가 재실행되었다(`PLAN.md` 5.4).

## 3. 검증 원칙

- P6의 모든 검증은 **재실행 결과로 판정한다.** 앞 Phase에서 통과했다는 기록을 근거로 삼지 않는다.
  공통 코드가 P2~P5 진행 중 변경되었을 수 있기 때문이다(`PLAN.md` 5.2).
- 규격은 `PLAN-P1 §13`의 스모크 규격과 5 epoch을 그대로 사용한다(`PLAN.md` 4). 전체 데이터 장시간 학습은
  v0.1의 완료 조건이 아니다.
- 모든 검증 산출물은 `outputs/`에 남기고, 판정 근거를 `docs/dev/v0.1/reviews/A6.md`에 경로와 함께 기록한다.

## 4. 하위 단계

### 4.1. P6.1 — 4태스크 전체 흐름 재실행

4개 벤치마크를 순차 실행한다.

```bash
python -m src benchmark configs/benchmarks/classification_baseline.yaml --overwrite
python -m src benchmark configs/benchmarks/segmentation_baseline.yaml  --overwrite
python -m src benchmark configs/benchmarks/detection_baseline.yaml     --overwrite
python -m src benchmark configs/benchmarks/anomaly_baseline.yaml       --overwrite
```

판정 기준:

- 12개 split(4태스크 x 3모델)이 모두 `status = ok`로 완주한다.
- 각 태스크의 `leaderboard.csv`와 `leaderboard.md`가 생성되고 metric·params·FLOPs·FPS 열이 채워진다.
- 각 split의 run 디렉토리에 `config.resolved.yaml`, `env.json`, `metrics_final.json`, `checkpoints/best.pth`,
  `predictions/`, `visualizations/`가 존재한다.
- `scripts/check_engine_purity.py`가 통과한다(`PLAN-P1 §7.4`). 엔진에 태스크 분기가 침투하지 않았음을
  최종 확인하는 지점이다.

### 4.2. P6.2 — 재현성 검증 (`NFR-01`, `AC-08`)

각 태스크에서 1개 split을 골라 동일 config·동일 seed로 2회 실행하고 `metrics_final.json`을 비교한다.
비교는 스크립트로 수행하며 육안 대조하지 않는다(`PLAN-P1 §5.2`).

| 대상 | 허용 오차 |
|---|---|
| 비율형 metric | 절대 오차 `<= 1e-3` |
| loss | 상대 오차 `<= 1e-2` (0 근처는 절대 `1e-4`) |
| params, FLOPs | 완전 일치 |
| FPS | 판정 제외 |

추가로 toy 4종을 `deterministic: strict` + `device: cpu`로 2회 실행해 **비트 단위 일치**를 확인한다.
GPU 경로의 편차가 비결정 커널에서 온 것인지 코드 결함에서 온 것인지 구분하는 대조군이다.

실측 편차가 허용 오차를 넘으면 기준을 완화하지 않고, 원인(비결정 연산 목록, `env.json`의
`nondeterministic_warnings`)과 함께 `PLAN-P1 §5.2`를 개정하고 `PLAN-P1 §16`에 기록한다.

### 4.3. P6.3 — 오프라인 완주 검증 (`NFR-07`, `AC-07`)

`PLAN-P1 §9.1`의 오프라인 가드는 모든 실행에서 상시 동작하므로 P6.1이 통과한 것만으로도 1차 근거가 된다.
여기서는 가드 자체가 무력화되지 않았음을 별도로 확인한다.

1. 전 config에서 `runtime.allow_network`가 `false`임을 확인한다.
2. 가드가 실제로 동작하는지 역검증한다. 외부 주소로 연결을 시도하는 한 줄짜리 스크립트를 가드 활성
   상태에서 실행해 `OfflineViolationError`가 발생하는지 확인한다. 가드가 조용히 통과하면 P6.1의
   완주는 근거가 되지 못한다.
3. `python -m src check-assets`가 통과한다(`PLAN-P1 §9.3`).
4. 가중치 파일 1개를 임시로 다른 이름으로 옮긴 뒤 해당 모델을 실행해 `LocalAssetError`로 실패하는지
   확인한다. 무작위 초기화 폴백이 없음을 실증하는 절차다. 확인 후 원상 복구한다.

### 4.4. P6.4 — split 누수 검사 (`AC-10`, `NFR-03`)

`scripts/check_split_integrity.py`를 작성해 다음을 일괄 검사한다.

- `configs/splits/*.json` 각각에서 `train`/`valid`/`test` ID 집합의 쌍별 교집합이 비어 있다.
- Detection 자체 분할이 `PLAN-P4 §3.2`의 사유와 함께 `note` 필드를 갖는다.
- 각 run의 `train.log`에서 test split DataLoader 생성이 `train` 경로에 나타나지 않는다.
- `allow_test_split` 플래그가 `evaluate` 서브커맨드 외에서 `True`로 설정되지 않는다(코드 검토).
- Anomaly의 threshold가 valid에서만 결정되었음을 `metrics_final.json`의 기록으로 확인한다
  (`PLAN-P5 §6`).

### 4.5. P6.5 — 수용 기준 대조표와 문서화

#### 4.5.1. `AC-01`~`AC-10` 대조표

`docs/dev/v0.1/reviews/A6.md`에 다음 표를 채운다. 각 행에 **판정 근거가 되는 산출물 경로 또는 명령**을
반드시 기입한다. "통과"만 적힌 행은 무효로 간주한다.

| AC | 검증 방법 | 근거 산출물 |
|---|---|---|
| AC-01 | 4태스크가 동일 CLI/Config 흐름으로 학습·평가 | P6.1의 12개 run 디렉토리 |
| AC-02 | 태스크별 3모델이 동일 조건 비교, 비교 축 확인 | 4개 `leaderboard.md`, `control_report.json` |
| AC-03 | metric + params/FLOPs/FPS 리포트 | `leaderboard.csv`의 해당 열 |
| AC-04 | split 순차 학습·평가·추론, 한 표 비교, 추론 산출물 | `leaderboard.md`, 각 split의 `predictions/` |
| AC-05 | `oxford_pets`가 Cls/Seg/Det에 재사용 | 3태스크 config의 `data.root` |
| AC-06 | Anomaly 3모델 비교, image·pixel AUROC 동시 산출 | `anomaly_baseline/leaderboard.md` |
| AC-07 | 네트워크 차단 상태 완주 | P6.3의 1~4 |
| AC-08 | 동일 seed 재실행이 허용 오차 내 일치 | P6.2 비교 스크립트 출력 |
| AC-09 | N=0·N=1·N>1·복수 클래스 혼합 배치 완주 | `PLAN-P4 §9.2` fixture 재실행 로그, toy-det 실행 로그 |
| AC-10 | split 상호 배타, test 미사용 | P6.4 검사 출력 |

미충족 AC가 있으면 v0.1을 완료로 판정하지 않고 사용자에게 보고한다.

#### 4.5.2. 공통 엔진 한계선 문서화 (`FR-09`, `NFR-04`)

`docs/dev/v0.1/LIMITS.md`를 작성해 공통 엔진이 흡수하지 못하는 경계를 기록한다. 최소한 다음을 포함한다.

- 표준 gradient 루프를 벗어나는 모델의 처리 방식과 한계 (`OUT-14` memory-bank류가 `on_fit_start` 훅과
  `epochs=0`으로 표현 가능한지, P1에서 확인한 결과)
- `eval_step`이 loss를 반환하지 못하는 태스크(Detection, Anomaly)와 그로 인해 valid loss 기반 모델 선택을
  쓸 수 없다는 제약 (`PLAN-P4 §7`, `PLAN-P5 §7.2`)
- 고정 서브모듈을 가진 모델이 `train()` 오버라이드를 요구한다는 계약상의 부담 (`PLAN-P5 §4.2`)
- 통제 필드가 태스크별로 확장되어야 했던 사례 (`PLAN-P4 §8.3`)
- Detection이 정본 split을 공유하지 못한 데이터상의 제약 (`PLAN-P4 §3.2`)
- `deterministic: strict`가 실데이터 태스크에서 불가능한 이유 (`PLAN-P1 §5.1`)
- P2~P5 진행 중 발생한 등급 B·C 계약 변경 목록 (`PLAN-P1 §16`)

이 문서는 v0.2의 출발점이 된다. 한계선을 감추면 다음 버전에서 같은 문제를 다시 만난다.

#### 4.5.3. README

저장소 루트 `README.md`를 작성한다. 목차는 다음으로 고정한다.

1. 프로젝트 개요와 목적 (벤치마크 boilerplate, 태스크 내 비교 한정)
2. 요구 환경 (conda `pytorch_env`, GPU, torch 2.5.1+cu121)
3. 로컬 자산 준비 (`/mnt/d/datasets`, `/mnt/d/backbones`, `check-assets`) — 자동 다운로드하지 않음을 명시
4. 빠른 시작 (toy 태스크 실행 → 실태스크 학습 → 벤치마크 → leaderboard)
5. CLI 레퍼런스 (`PLAN-P1 §11`)
6. Config 구조 (`PLAN-P1 §3`)
7. 새 모델 추가 방법 (registry 등록 + config 1개, `NFR-05`)
8. 새 데이터셋 추가 방법 (target 규약 준수 Dataset + registry, `NFR-06`)
9. 결과 해석 시 유의사항 (5 epoch·축소 subset 규격이며 절대 성능이 아님)
10. 한계와 다음 버전 (`LIMITS.md` 링크)

이모지를 사용하지 않는다(`CON-14`). 코드·명령어는 원문 표기를 유지한다.

### 4.6. P6.6 — 최종 적대적 검증과 마무리

- 검토자는 마지막 실질 구현자의 반대 벤더 CLI다. P6는 master(Claude Code)가 수행하므로 Codex CLI다
  (`AGENTS.md` Adversarial Review Rules).
- 공격 초점은 §5와 같다.
- 결과를 `docs/dev/v0.1/reviews/A6.md`에 기록한다. 실행 횟수는 Verification Attempt Limit(3회)를 따른다.
- Critical을 수정했으면 P6.1~P6.4 중 영향받는 검증을 재실행한다.
- 변경 내용·검증 결과·남은 위험을 사용자에게 보고하고 커밋·푸시 승인을 요청한다. 승인 전에는
  커밋·푸시하지 않는다.

## 5. 적대적 검증 초점

`backlog.json`의 P6 `adversarialFocus`가 이 표를 사용한다. P6는 개별 태스크가 아니라 **통합 상태**를
대상으로 하므로 공격 축이 앞 Phase와 다르다.

| 축 | 공격 내용 | 대응 조항 |
|---|---|---|
| 엔진 순수성 (최종) | P2~P5를 거치며 `core`·`bench`에 태스크 분기가 침투했는가. 순수성 검사가 우회 가능한가 | §4.1, `PLAN-P1 §7.4` |
| 검증의 실효성 | 각 AC가 재실행 산출물로 판정되었는가. 앞 Phase의 기록을 근거로 재사용한 항목이 있는가 | §3, §4.5.1 |
| 오프라인 역검증 | 가드가 실제로 차단하는지 확인했는가. 완주 사실만으로 통과 판정하지 않았는가 | §4.3 |
| 재현성 판정 | 허용 오차를 실측에 맞춰 사후 완화하지 않았는가. 개정이 기록되었는가 | §4.2 |
| 누수 (최종) | 4태스크 전체에서 test가 모델 선택·threshold에 개입하지 않았는가 | §4.4 |
| 한계선 정직성 | `LIMITS.md`가 실제 발생한 제약과 계약 변경을 빠짐없이 기록했는가 | §4.5.2 |

## 6. 조항 개정 이력

| 일자 | 조항 | 등급 | 변경 내용 | 요청자 | 승인 |
|---|---|---|---|---|---|
| 2026-08-18 | 전체 | — | 최초 작성 | master | — |

---

*작성일: 2026-08-18 · 버전: v0.1 · 상위 문서: PLAN.md · 다음 단계: backlog.json*
