# LIMITS — 공통 엔진 한계선 (v0.1)

이 문서는 `PLAN-P6 §4.5.2`에 따라 v0.1 공통 엔진(`src/core`, `src/bench`)이 4개 태스크(P2~P5)를
거치며 실제로 흡수하지 못한 경계와, 그 결과 발생한 계약 변경(등급 B·C)을 기록한다. v0.2 착수 시
이 문서를 출발점으로 삼는다.

## 1. 표준 gradient 루프를 벗어나는 모델

`Trainer.fit()`은 `on_fit_start` → (`train.epochs`회 학습) → `on_fit_end` 순서로 4개 훅만 제공한다.
`train.epochs=0`이면 `range(start_epoch, 0+1)`이 빈 시퀀스가 되어 학습 루프 없이 `on_fit_start`와
`on_fit_end`만 실행된다. `OUT-14`(PatchCore 등 memory-bank류)가 요구하는 "학습 없이 `on_fit_start`에서
뱅크를 채우는" 경로는 이 구조로 표현 가능함을 P1에서 코드 경로로 확인했다(`PLAN-P1 §6.5`). 다만
v0.1은 이 경로로 실제 모델을 구현하지 않았으므로(`OUT-14` 비범위), memory-bank류가 이 훅 4개만으로
실제 요구를 모두 충족하는지는 v0.2에서 검증이 필요하다.

## 2. `eval_step`이 loss를 반환하지 못하는 태스크

Detection(`PLAN-P4 §7`)과 Anomaly(`PLAN-P5 §7.2`)는 valid 단계에서 의미 있는 스칼라 loss를 만들지
못한다(Detection은 `model.eval()` 모드에서 loss를 계산하지 않는 torchvision/ultralytics 관례를 따르고,
Anomaly는 정상 이미지만으로 학습해 재구성/거리 기반 손실이 valid 성능과 직결되지 않는다). 두 태스크
모두 `train.monitor.metric`을 loss가 아닌 태스크 지표(`map_50_95`, `image_auroc`)로 설정해 모델 선택을
수행한다. 공통 엔진은 "loss 기반 모델 선택"을 강제하지 않고 `monitor.metric`을 config로 결정하게
설계되어 있어 이 제약을 흡수했지만, valid loss 자체가 `None`이거나 `0.0`으로 기록되는 것은(§4.4 벤치마크
로그에서 `loss=0.0000` 확인) 공통 엔진의 표현 한계로 남는다.

## 3. 고정 서브모듈을 가진 모델의 `train()` 오버라이드 부담

STFPM(`stfpm_anomaly`)과 EfficientAD(`efficientad_anomaly`)는 고정 teacher를 가지며, `model.train()`
호출 시 teacher가 BatchNorm 통계를 갱신하지 않도록 각 모델이 `train(self, mode=True)`를 오버라이드해
`self.teacher.eval()`을 강제한다(`src/tasks/anomaly/models/stfpm.py:97`,
`src/tasks/anomaly/models/efficientad.py:132`, `PLAN-P5 §4.2`). 공통 `Trainer._train_epoch()`은
`model.train()`을 한 번만 호출하므로 이 오버라이드에 의존한다 — 고정 서브모듈을 가진 새 모델을 추가할
때마다 이 패턴을 반복해야 하며, 엔진이 "학습 대상 파라미터"와 "고정 서브모듈"을 구조적으로 분리해주지
않는다.

## 4. 통제 필드가 태스크별로 확장되어야 했던 사례

Detection은 `score_thresh`/`nms_iou`/`max_det` 3개 후처리 파라미터가 모델 간 다르면 비교가
불공정해지므로, `PLAN-P4 §8.3`에서 등급 B 계약 확장으로 `control.extra_fields`를 도입해 벤치마크
yaml이 태스크별 통제 필드를 추가로 지정할 수 있게 했다(`PLAN-P1 §16` 2026-08-19 항목,
`src/bench/control.py`). 공통 `CONTROL_FIELDS` 목록만으로는 Detection의 공정 비교 요구를 충족하지
못했다는 뜻이며, 새 태스크가 추가될 때마다 유사한 확장이 필요할 수 있다.

## 5. Detection이 정본 split을 공유하지 못한 데이터 제약

`oxford_pets`의 `test.txt`에는 XML bounding-box 주석이 0건이어서, Classification/Segmentation이
쓰는 정본 split(`configs/splits/oxford_pets.json`)을 Detection이 재사용할 수 없다. Detection은
`trainval.txt ∩ annotations/xmls`를 모집단으로 자체 70/15/15 split(`configs/splits/oxford_pets_det.json`)을
만들었고, 그 결과 Detection의 test 이미지 일부가 Classification의 train 이미지와 겹친다. 이는
`AC-10`/`NFR-03`이 "한 태스크 내부의" 상호 배타성만 요구하므로(`OUT-01`, 태스크 간 비교는 비범위)
허용되지만, `oxford_pets` 하나로 4태스크를 완전히 통일된 split으로 비교할 수는 없다는 한계로 남는다
(`PLAN-P4 §3.2`).

## 6. `deterministic: strict`가 실데이터 태스크에서 불가능한 이유

Segmentation의 `interpolate` backward, Detection의 산포(scatter) 연산 등 일부 CUDA 커널은 결정적
구현이 없다. `strict`(`torch.use_deterministic_algorithms(True)`, 비결정 연산 시 예외 발생)로는 이
연산들이 예외를 던져 학습 자체가 불가능하므로, 실데이터 4태스크는 기본값 `warn`
(`use_deterministic_algorithms(True, warn_only=True)`)을 사용한다(`PLAN-P1 §5.1`). `strict` +
`device: cpu`는 toy 4종에서만 회귀 테스트 용도로 사용되며, P6.2에서 재검증한 결과 4종 모두 2회 실행이
`model_state`·`metrics_epoch.csv`(경과 시간 제외) 비트 단위로 일치했다.

## 7. 가중치 누락 시 실제로 발생하는 예외 타입

`PLAN-P5 §4.3`/`PLAN-P6 §4.3`은 teacher 가중치가 없을 때 `LocalAssetError`로 실패함을 검증 절차로
명시한다. P6.3 역검증 결과, `weights_path`를 갖는 모든 모델(classification의 `resnet50`, anomaly의
`stfpm` 포함)에서 실제로는 `validate_config()`가 학습 시작 전 `model.params.weights_path` 파일
존재 여부를 먼저 검사해 `ConfigError`로 실패하며, `load_local_weights()`가 던지는 `LocalAssetError`
코드 경로는 CLI 경로상 도달하지 않는다(config 검증이 더 이른 시점에 동일한 의도 — 무작위 초기화 폴백
없음 — 를 충족하기 때문). 두 예외 모두 "폴백 없이 즉시 실패"라는 요구(`NFR-07`)는 만족하지만, 문서가
명시한 예외 타입과 실제 예외 타입이 다르다는 점을 한계로 기록한다.

## 8. 등급 B·C 계약 변경 목록 (P2~P6)

전체 조항 개정 이력은 `PLAN-P1 §16`에 있다. 등급 B 이상만 요약한다.

| 일자 | 대상 | 등급 | 요약 |
|---|---|---|---|
| 2026-08-19 | `bench/control.py`, `bench/runner.py`, `cli/commands.py` | B | 통제 필드 목록에 `extra_fields` 도입 (Detection 후처리 파라미터 통제, §4) |
| 2026-08-19 | `bench/leaderboard.py` | B | leaderboard 컬럼을 `core_fields` + `extra_fields`로 일반화 (Detection의 다중 mAP 컬럼 대응) |
| 2026-08-19 | `core/adapter.py`, `tasks/detection/adapter.py`, `bench/profile.py`, `bench/runner.py` | B | `dummy_forward_input()` 훅 도입 (Detection의 list-of-Tensor 입력 계약 대응) |
| 2026-08-19 | `core/adapter.py`, `tasks/detection/adapter.py`, `cli/commands.py` | B | `bind_class_names_from_config()` 훅 도입 (`predict` 경로의 클래스 이름 바인딩 누락 수정) |
| 2026-08-19 | `core/adapter.py`, `cli/commands.py` | B | `extra_final_metrics()` 훅 도입 (Anomaly threshold를 `metrics_final.json`에 저장) |

등급 C(계약 변경) 사례는 v0.1 기간 중 발생하지 않았다.

등급 A(계약 무변경 버그 수정) 중 P6에서 발견된 것은 다음과 같다 (전체는 `PLAN-P1 §16` 참조).

- `core/engine.py`: `Trainer.fit()`의 best-checkpoint 재로드 순서가 `on_fit_end` 이후였던 버그 수정
  (P5 구현 중 발견, 이번 P6 재실행으로 재검증)
- `bench/runner.py`: 벤치마크 경로가 `adapter.visualize()`를 호출하지 않아 `visualizations/`가
  생성되지 않던 버그 수정 (P6.1 재실행 중 발견)
- `bench/runner.py`: 벤치마크 경로가 `adapter.extra_final_metrics()`를 호출하지 않아 Anomaly
  threshold가 벤치마크의 `metrics_final.json`에 기록되지 않던 버그 수정 (P6.1 재실행 중 발견)
- `core/builders.py::build_optimizer()`: 고정 서브모듈(teacher) 파라미터가 필터링 없이 optimizer에
  전달되어 `param_groups`에 포함되던 문제 수정 (P6.6 Codex 검토 Major #2)
- `core/offline.py`: 오프라인 가드가 UDP `sendto()`/`sendmsg()` 경로를 막지 않던 gap 수정
  (P6.6 Codex 검토 Major #3)
- `cli/commands.py::train()`, `bench/runner.py::execute_split()`: `metrics_final.json`의 `valid`가
  `on_fit_end`의 calibration 이전 시점 값을 기록하던 문제 수정 — `fit()` 직후 valid를 재평가
  (P6.6 Codex 검토 Major #1)

## 9. 통제 필드 확장이 opt-in이라는 구조적 한계 (P6.6 Codex 검토 Major #4)

`src/bench/control.py`의 `CONTROL_FIELDS`는 `model.params`를 포함하지 않는다 — 모델마다 아키텍처가
달라 `model.params`(예: `backbone_name`, `weights_path`)가 정당하게 다른 것이 일반적이기 때문이다.
Detection의 `score_thresh`/`nms_iou`/`max_det`처럼 "모델이 달라도 반드시 동일해야 하는" 후처리
파라미터는 벤치마크 yaml이 `control.extra_fields`에 명시적으로 선언해야만 검사된다
(`configs/benchmarks/detection_baseline.yaml`은 이를 올바르게 선언하고 있다 — §4 참조).

이 설계는 "새 태스크의 통제 요구를 공통 코드 수정 없이 config로 확장할 수 있다"는 장점이 있지만,
반대로 **새 detection 벤치마크 config가 `extra_fields` 선언을 빠뜨리면 `enforce_control()`이 이를
감지하지 못하고 통과시킨다**는 약점이 있다. `model.params`를 기본 통제 필드로 승격하면 이 위험은
없어지지만 서로 다른 모델의 정당한 `params` 차이(예: `custom_fcos`엔 없는 `weights_path`)가 전부
위반으로 잡혀 `control.exceptions`를 모델 수만큼 등록해야 하므로 채택하지 않았다. v0.1에서는
Detection 벤치마크 config 1개가 유일한 사용처이고 올바르게 선언되어 있어 실제 위반은 없으나, 새
태스크가 유사한 통제 요구를 가질 때 이 opt-in 구조를 그대로 반복할지 재검토가 필요하다.

## 10. 알려진 채택 리스크 (ISS-06)

`Trainer.fit()`을 새 `checkpoint_dir`로 `--resume`하되 그 디렉터리에 `best.pth`가 아직 없는 조합에서
best-reload/재저장이 스킵되어 `on_fit_end`가 non-best 모델을 대상으로 calibration하는 edge case가
P5 Codex 적대적 검증(A5, Major #2)에서 지적되었다. v0.1 CLI 사용 패턴에는 없는 드문 조합이라
리스크로 수용했으며, `Trainer.fit()`의 resume 분기 재설계는 v0.2 이후 별도 Grade B 변경 요청 대상이다.

---

*작성일: 2026-08-19 · 버전: v0.1 · 상위 문서: `PLAN-P6-integration.md` §4.5.2*
