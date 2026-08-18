# PyTorch CV Boilerplate Agent Instructions

Claude Code가 이 저장소에서 작업할 때 매 턴 지켜야 하는 규칙이다. 배경과 근거는 `docs/dev/v0.1/` 문서에 둔다. 전체 지침은 `AGENTS.md`를 참조한다.

## General Rules

- 이모지를 사용하지 않는다.
- 제품 코드는 Python으로 구현하고 학습·평가·추론 로직은 pure-PyTorch로 작성한다. Lightning 등 상위 학습 프레임워크를 도입하지 않는다.
- 코드 내 주석은 영어로 작성한다. Markdown 문서는 한국어로 작성한다. 코드, 명령어, 파일 경로, 제품·라이브러리 고유 이름은 원문 표기를 유지한다.
- 사용자의 명시적인 요청 없이 코드나 문서를 생성하지 않는다.
- 대상 환경은 WSL2(Linux)이며 셸은 bash를 사용한다. Python 실행·검증 전에 conda 환경 `pytorch_env`를 활성화한다.
- 경로 표기는 `os.path` 방식을 사용하며 `pathlib.Path`를 사용하지 않는다.
- PEP8 네이밍을 따르고 멤버 변수 접두사와 등호·콜론 세로 정렬을 금지한다. split은 `valid`, 백본은 `backbone_name`으로 표기한다.

## Project Rules

- 엔진(Trainer/Engine 루프)은 task-agnostic을 유지한다. 공통 루프에 태스크 이름으로 분기하는 조건문을 두지 않는다.
- Dataset이 반환하는 target 형태는 `BRIEF.md` 6.1의 공통 규약을 따른다. 모델·Loss·Metric은 규약에만 의존하고 특정 데이터셋 구조에 결합하지 않는다.
- Detection 파이프라인은 multi-class / multi-object를 일반적으로 지원한다. `oxford_pets`가 단일 객체라는 이유로 축약하지 않는다.
- Anomaly Detection은 외부 라이브러리 임포트를 최소화하고 PyTorch 수준에서 직접 작성한다. anomalib는 모델 정의만 차용한다.
- 데이터셋과 백본 가중치를 자동 다운로드하지 않는다. `/mnt/d/datasets`와 `/mnt/d/backbones`의 로컬 경로만 참조하고, pretrained는 `weights=None` + 로컬 `.pth`의 `load_state_dict`로 주입한다.
- 실험 관리 도구(tensorboard, wandb)를 도입하지 않는다. 결과는 config·metric 기록과 leaderboard 표(CSV/Markdown)로 관리한다.
- 새 의존성을 추가하기 전 기존 스택(torch, torchvision, torchmetrics)으로 구현 가능한지 확인하고 사용자 승인을 받는다.
- `BRIEF.md`에 확정된 범위, 데이터셋, 비교 모델, 기술 스택을 임의로 변경하지 않고 불필요한 추상화 계층을 추가하지 않는다.

## Common Contract and Agent Execution Rules

- P1(공통 기반)과 P6(통합 검증)은 메인 세션이 직접 구현한다. 태스크 Phase(P2~P5)는 task 에이전트에 위임하고, task 에이전트는 태스크 기반을 확정한 뒤 모델 3종을 모델 에이전트 3개에 동시 위임한다. 태스크 Phase는 순차 진행한다.
- 모델 에이전트는 자기 모델 파일과 해당 config 외의 파일을 수정하지 않는다.
- 공통 코드(엔진·벤치마크·CLI 계층) 수정 권한은 메인 세션만 갖는다. 하위 에이전트는 변경 요청만 반환한다. 변경 등급 C(계약 변경)는 사용자 승인 후에만 진행한다.
- 공통 루프에 태스크 이름으로 분기하는 수정은 허용하지 않는다. 공통 코드가 바뀌면 완료된 모든 태스크의 스모크를 재실행한다.
- 모든 Phase의 구현 검증은 toy 데이터 또는 축소 subset과 5 epoch로 수행한다.
- 상세는 `docs/dev/v0.1/PLAN.md` 2장과 5장을 따른다.

## Document Rules

- 개발 문서는 `docs/dev/v{major}.{minor}/`에 둔다. 문서 체인은 `BRIEF.md → PRD.md → PLAN.md → plans/PLAN-P{n}-*.md → backlog.json`이다. v0.1은 별도 `SPEC.md`를 두지 않고 각 Phase PLAN 문서가 사양을 담는다.
- Phase별 PLAN 문서는 해당 Phase 착수 직전에 작성한다. 각 문서의 절 번호(예: `PLAN-P1 §4.2`)가 적대적 검증과 `backlog.json`이 참조하는 조항 ID다.
- 사용자 요청으로 구현 또는 프로젝트 내용이 변경되면 `plans/PLAN-P{n}-*.md → PLAN.md → backlog.json → PRD.md` 순서로 갱신한다.
- 완료된 버전의 문서는 참조 전용으로 유지하며, 사용자의 명시적 요청 없이는 수정하지 않는다. 문서와 구현 작업은 현재 진행 중인 버전 폴더에만 반영한다. 현재 진행 중인 버전이 v0.2이면 `docs/dev/v0.2/`만 갱신하고 `docs/dev/v0.1/`은 참고용으로만 읽는다.
- Phase 완료 상태는 `backlog.json`의 각 Phase `status` 필드에서만 관리한다.
- 문서(`BRIEF.md`, `PRD.md`, `PLAN.md`, `plans/PLAN-P{n}-*.md`)도 사용자 요청이 있으면 반대 벤더 CLI로 적대적 검증을 받으며, 검증 횟수는 Verification Attempt Limit을 따른다.

## Phase Execution Workflow

1. 해당 Phase의 `scope`, `acceptanceCriteria`를 구현하고 검증한다. 검증에는 실제 학습·평가 스모크 실행을 포함한다.
2. 마지막 실질 구현자의 **반대 벤더 CLI**에 적대적 검증을 위임한다. Codex 구현은 Claude Sonnet headless CLI가, Claude Code 구현은 Codex CLI가 검토한다. 토큰 한도로 구현자가 Phase 중간에 바뀌면 마지막 실질 구현자를 기준으로 다시 정한다.
3. Critical 지적은 모두 수정하고 관련 검증을 재실행한다. Major와 Minor는 처리 여부와 근거를 기록한다. Critical을 수정했다면 같은 반대 벤더 검토를 한 번 더 실행해 해소를 확인한다. 실행 횟수는 아래 Verification Attempt Limit을 따른다.
4. `docs/dev/v{major}.{minor}/reviews/A{n}.md`에 구현자, 검토 모델, 대상 파일, 실행 일시, 심각도별 건수, 지적·재현 조건·관련 PLAN 조항·처리 상태를 기록한다. 유효하지 않은 지적의 반박 근거도 기록한다.
5. 변경 내용, 검증 결과, 검토 결과와 남은 위험을 사용자에게 보고하고 커밋·푸시 승인을 요청한다. 승인 전에는 커밋 또는 푸시하지 않는다.

적대적 검증 필수 통과 Phase는 `backlog.json`의 `phaseReviewProfiles[*].mandatory`로 지정한다. 미해결 Critical이 있으면 다음 Phase로 진행하지 않는다. 필요한 반대 벤더 CLI를 사용할 수 없으면 오류와 대체 검증안을 사용자에게 보고하고 승인 없이 생략하지 않는다.

## Verification Attempt Limit

적대적 검증 실행 횟수는 Phase 검증과 문서 검증에 동일하게 적용한다.

- 하나의 검증 대상(Phase 1개 또는 문서 1개)에 대한 검토 CLI 실행은 실패·오류·프롬프트 재작성·재검토를 모두 포함해 최대 3회로 제한한다.
- 지적 보완 후의 재검증도 이 3회에 포함된다. 보완 사이클을 별도로 세지 않는다.
- 3회를 소진하면 추가 실행 대신 마지막 유효 검토 결과, 반영한 보완 내용, 미해결 지적과 남은 위험을 기록하고 사용자에게 보고한다.
- 3회 소진 시점에 미해결 Critical이 남아 있으면 다음 Phase로 진행하지 않고 사용자 판단을 요청한다.
- 실행 횟수와 각 회차의 결과(모델, 실행 일시, 유효 여부, 사유)를 검토 기록 문서에 남긴다.

## Adversarial Review Rules

- 검토자는 제품 소스 파일, 해당 Phase의 `adversarialFocus`, `planRefs`만 사용한다. 구현 세션 대화, 구현 판단 근거, 문서, 설정, 실험 산출물, Git 상태·이력·원격 저장소, 셸 도구는 요청하거나 사용하지 않는다.
- 검토자는 파일을 수정하지 않는다. 지적은 `Critical / Major / Minor`, 정확한 재현 조건, 위반한 PLAN 조항을 포함해 심각도순으로 반환한다.
- 이 프로젝트의 우선 공격 축은 엔진의 task-agnostic 위반, target 규약 위반, Detection 일반성(가변 N·다중 클래스·`collate_fn`·N=0), 공정 비교 통제(해상도·augmentation·optimizer·seed 고정), 오프라인 위반, 학습/평가 누수(`model.eval()`·`no_grad()`·metric 리셋), 재현성이다.
- Claude Sonnet 검토는 `claude -p`에 `--model sonnet --safe-mode --allowedTools "Read,Glob,Grep" --disallowedTools "Edit,Write,Bash" --permission-mode dontAsk --max-turns 5 --output-format json --no-session-persistence`를 사용한다. 필요하면 `--max-budget-usd`로 호출별 비용 상한을 둔다.
- Codex 검토는 `codex exec --model gpt-5.6-sol --sandbox read-only --cd "/mnt/d/projects/nampluskr/00_review/260818_cv-boilerplate"`를 사용한다. 모델 접근이 거부되면 기본 모델로 조용히 폴백하지 않고 오류와 대체안을 보고한다.
- 문서 검증 시에는 검토자가 대상 문서와 상위 문서, `AGENTS.md`를 읽기 위해 읽기 전용 셸 명령(`cat`, `sed -n` 등)을 사용하는 것을 허용한다. Codex CLI에는 별도 파일 읽기 도구가 없어 셸을 전면 금지하면 검토가 불가능하다.
- 검토 CLI 실행의 시간 제한은 기본 10분으로 설정한다.

## Commit and Push Rules

- 원격 저장소는 `https://github.com/nampluskr/cv_boilerplate`로 확정되었다(2026-08-18 사용자 확인). `backlog.json` 생성 시 `remoteRepository`에 이 URL을 기록한다.
- 로컬 저장소 초기화(`git init`)와 원격 연결, 첫 푸시는 이후 초기 작업 Phase에서 사용자 승인 아래 수행한다. 그 전에는 커밋하지 않는다.
- 커밋은 하나의 완료된 Phase에 대응하며, Phase 번호와 핵심 변경 사항을 포함한다.
- 다른 작업의 변경 사항을 임의로 포함, 되돌리기, 삭제하지 않는다.
- 데이터셋, 백본 가중치, 체크포인트, 실험 산출물은 커밋하지 않는다.
