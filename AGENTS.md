# PyTorch CV Boilerplate Agent Instructions

이 저장소에서 에이전트(Claude Code, Codex CLI)가 작업할 때 따르는 전체 지침이다.
요구사항의 배경과 근거는 `docs/dev/v0.1/BRIEF.md`에 둔다.

## General Rules

- 이모지를 사용하지 않는다.
- 제품 코드는 Python으로 구현한다. 학습·평가·추론 로직은 pure-PyTorch로 작성하며 Lightning 등 상위 학습 프레임워크를 도입하지 않는다.
- 코드 내 주석은 영어로 작성한다. Markdown 문서는 한국어로 작성한다. 코드, 명령어, 파일 경로, 제품·라이브러리 고유 이름은 원문 표기를 유지한다.
- 사용자의 명시적인 요청 없이 코드나 문서를 생성하지 않는다.
- 대상 환경은 WSL2(Linux)이며 셸은 bash를 사용한다.
- Python 실행·검증은 conda 환경 `pytorch_env`에서 수행한다. 인터프리터 경로는 `/home/nampl/anaconda3/envs/pytorch_env/bin/python`이다.

  ```bash
  conda activate pytorch_env
  ```

## Code Style Rules

- 경로 표기는 `os.path` 방식을 사용하며 `pathlib.Path`를 사용하지 않는다.
- 네이밍은 PEP8을 따른다. 변수·함수는 `snake_case`, 클래스는 `PascalCase`, 상수는 `UPPER_SNAKE_CASE`를 사용한다.
- 멤버 변수에 접두사를 붙이지 않는다. private은 단일 언더스코어(`_`)만 허용한다.
- 등호와 콜론의 세로 정렬(vertical alignment)을 금지한다. dict 리터럴도 동일하다.
- 도메인 용어를 통일한다. split 명칭은 `valid`를 사용하고 `val`을 쓰지 않는다. 모델명과 백본명을 구분해 백본은 `backbone_name`으로 표기한다.

## Project Rules

- 엔진(Trainer/Engine 루프)은 task-agnostic을 유지한다. 엔진 코드에 태스크 이름으로 분기하는 조건문을 두지 않는다. 태스크별 차이는 Task 어댑터, Dataset, Loss, Metric, Postprocess에만 존재한다.
- Dataset이 반환하는 target 형태는 `BRIEF.md` 6.1의 태스크별 공통 규약을 따른다. 모델·Loss·Metric은 이 규약에만 의존하며 특정 데이터셋 구조에 결합하지 않는다.
- Detection 파이프라인은 multi-class / multi-object를 일반적으로 지원한다. `oxford_pets`가 이미지당 객체 1개라는 이유로 단일 객체 전용으로 축약하지 않는다.
- Anomaly Detection은 외부 라이브러리 임포트를 최소화하고 PyTorch 수준에서 직접 작성한다. anomalib는 모델 정의(`nn.Module`) 차용에 한정하며 anomalib의 학습 프레임워크는 사용하지 않는다.
- 데이터셋과 백본 가중치를 프로젝트가 자동 다운로드하지 않는다. 데이터셋은 `/mnt/d/datasets`, 백본 가중치는 `/mnt/d/backbones`의 로컬 경로만 참조한다.
- pretrained 모델은 `weights=None`으로 아키텍처를 만든 뒤 로컬 `.pth`를 `load_state_dict`로 주입한다. 네트워크 접근을 유발하는 `weights=` 인자나 `torch.hub` 다운로드를 사용하지 않는다.
- 실험 관리 도구(tensorboard, wandb 등)를 도입하지 않는다. 벤치마크 결과는 config·metric 기록과 leaderboard 표(CSV/Markdown)로 관리한다.
- 새 의존성을 추가하기 전 기존 스택(torch, torchvision, torchmetrics)으로 구현 가능한지 확인하고 사용자 승인을 받는다.
- `BRIEF.md`에 확정된 범위, 데이터셋 선택, 비교 모델 구성, 기술 스택을 임의로 변경하지 않고 불필요한 추상화 계층을 추가하지 않는다.
- 폴더구조·파일 구성·모듈 경계는 기존 프로젝트를 복사하지 않고 `BRIEF.md` 6장 지침대로 원점에서 재설계하여 `plans/PLAN-P1-foundation.md`에서 확정한다.

## Common Contract and Agent Execution Rules

Phase 구조와 실행 계층은 `docs/dev/v0.1/PLAN.md`에 정의한다. 에이전트는 다음을 지킨다.

- P1(공통 기반)과 P6(통합 검증)은 메인 세션이 직접 구현한다. 태스크 Phase(P2~P5)는 task 에이전트에 위임하고, task 에이전트는 태스크 기반을 먼저 확정한 뒤 모델 3종을 모델 에이전트 3개에 동시 위임한다.
- 태스크 Phase는 순차 진행한다. 앞 Phase가 완료·검토·승인된 뒤 다음 Phase를 시작한다.
- 모델 에이전트는 자기 모델 파일과 해당 config 외의 파일을 수정하지 않는다.
- 공통 코드(엔진·벤치마크·CLI 계층) 수정 권한은 메인 세션만 갖는다. task 에이전트와 모델 에이전트는 직접 수정하지 않고 문제, 재현 조건, 최소 수정안, 영향받는 PLAN 조항 번호를 담은 변경 요청을 반환한다.
- 공통 코드 변경은 등급 A(계약 무변경) / B(하위호환 확장) / C(계약 변경)로 구분한다. C는 사용자 승인 후에만 진행하며 완료된 태스크를 전부 재실행한다.
- 어떤 등급이든 공통 루프에 태스크 이름으로 분기하는 수정은 허용하지 않는다. 어댑터 또는 훅으로 흡수한다.
- 공통 코드가 바뀌면 그 시점까지 완료된 모든 태스크의 스모크를 재실행해 회귀를 확인한다.
- 모든 Phase의 구현 검증은 toy 데이터 또는 축소 subset과 5 epoch로 수행한다. 전체 데이터 장시간 학습은 v0.1의 완료 조건이 아니다.

## Document Rules

- 개발 문서는 `docs/dev/v{major}.{minor}/`에 둔다.
- 문서 체인은 `BRIEF.md`(요구사항) → `PRD.md`(제품 요구) → `PLAN.md`(Phase 전체 정의) → `plans/PLAN-P{n}-*.md`(Phase 상세 및 공통 계약 조항) → `backlog.json`(실행 단위) 순으로 파생된다. v0.1은 별도 `SPEC.md`를 두지 않으며, 사양은 각 Phase PLAN 문서가 담는다.
- Phase별 PLAN 문서는 해당 Phase 착수 직전에 작성한다. 각 문서의 절 번호(예: `PLAN-P1 §4.2`)가 적대적 검증과 `backlog.json`이 참조하는 조항 ID다.
- 사용자 요청으로 구현 또는 프로젝트 내용이 변경되면 `plans/PLAN-P{n}-*.md → PLAN.md → backlog.json → PRD.md` 순서로 갱신한다.
- 완료된 버전의 문서는 참조 전용으로 유지하며, 사용자의 명시적 요청 없이는 수정하지 않는다. 문서와 구현 작업은 현재 진행 중인 버전 폴더에만 반영한다. 현재 진행 중인 버전이 v0.2이면 `docs/dev/v0.2/`만 갱신하고 `docs/dev/v0.1/` 문서는 형식과 과거 결정의 참고 목적으로만 읽는다.
- Phase 완료 상태는 `backlog.json`의 각 Phase `status` 필드에서만 관리한다. `README.md`, `PLAN.md`, `plans/PLAN-P{n}-*.md`, `PRD.md`에는 Phase 상태를 기록하지 않는다.
- 문서(`BRIEF.md`, `PRD.md`, `PLAN.md`, `plans/PLAN-P{n}-*.md`)도 사용자 요청이 있으면 반대 벤더 CLI로 적대적 검증을 받는다. 검토자 선정 기준은 Phase 검증과 같고, 검토자는 대상 문서와 상위 문서, `AGENTS.md`만 읽는다.
- 문서 검증 역시 실행과 보완을 합쳐 최대 3회로 제한한다. 상세 규칙은 아래 Verification Attempt Limit을 따른다.

### backlog.json 구조

`backlog.json`은 최상위에 프로젝트 메타와 정책을, `phases` 배열에 실행 단위를 담는다.

| 키 | 내용 |
|---|---|
| `project`, `source` | 프로젝트 이름, 파생 원본 문서 경로 |
| `executionPolicy` | Phase 실행 순서 정책 (`sequential`) |
| `remoteRepository` | 원격 저장소 URL |
| `changeManagementPolicy` | 문서 갱신 순서 규칙 |
| `phaseStatusPolicy` | Phase 상태 기록 위치 규칙 |
| `adversarialReviewPolicy` | 검토자 선정, 허용 컨텍스트, 기록 위치, Critical 처리, 폴백 |
| `phaseReviewProfiles` | Phase별 `reviewId`, `mandatory`, `adversarialFocus`, `planRefs` |
| `commonContractChangePolicy` | 공통 코드 수정 권한, 변경 등급, 회귀 검증 규칙 (`PLAN.md` 5장) |
| `agentExecutionModel` | Phase별 실행 주체와 task/모델 에이전트 계층 (`PLAN.md` 2장) |
| `phaseCompletionWorkflow` | Phase 완료 절차 |
| `commitPolicy` | 커밋·푸시 규칙 |
| `phases` | `id`, `order`, `title`, `status`, `dependsOn`, `scope`, `acceptanceCriteria`, `implementationNotes` |

## Phase Execution Workflow

`docs/dev/v{major}.{minor}/backlog.json`과 `PLAN.md`의 Phase는 정의된 순서와 의존성을 지켜 진행한다.
각 Phase는 다음 순서로 완료한다.

1. 해당 Phase의 `scope`, `acceptanceCriteria`를 구현하고 검증한다. 검증에는 실제 학습·평가 스모크 실행을 포함한다.
2. 마지막 실질 구현자의 **반대 벤더 CLI**에 적대적 교차 검증을 위임한다. Codex 구현은 Claude Sonnet headless CLI가, Claude Code 구현은 Codex CLI가 검토한다. 토큰 한도로 구현자가 Phase 중간에 바뀌면 마지막 실질 구현자를 기준으로 검토자를 다시 정한다.
3. 유효한 Critical 지적은 모두 수정하고 관련 검증을 재실행한다. Critical을 수정했다면 같은 반대 벤더 검토를 한 번 더 실행해 해소를 확인한다. Major와 Minor는 처리 여부와 근거를 기록한다. 적대적 검증 실행 횟수는 아래 Verification Attempt Limit을 따른다.
4. `docs/dev/v{major}.{minor}/reviews/A{n}.md`에 구현자, 검토 모델, 대상 파일, 실행 일시, 심각도별 건수, 지적·재현 조건·관련 PLAN 조항·처리 상태를 기록한다. 유효하지 않은 지적의 반박 근거도 기록한다.
5. 변경 내용, 검증 결과, 교차 검토 결과와 보완 조치, 남은 위험을 사용자에게 보고하고 커밋·푸시 승인을 요청한다. 검토 결과는 지적사항별 심각도, 근거, 처리 상태를 포함한 Markdown 표로 제시한다.
6. 사용자의 명시적 승인을 받은 후에만 해당 Phase 변경을 커밋하고 원격 저장소에 푸시한다.

적대적 검증 필수 통과 Phase는 `backlog.json`의 `phaseReviewProfiles[*].mandatory`로 지정한다. 필수 Phase에 미해결 Critical이 있으면 다음 Phase로 진행하지 않는다.

필요한 반대 벤더 CLI를 사용할 수 없는 환경이면 그 사실과 사유를 사용자에게 알리고, 대체 검증 방안을 제시한 뒤 승인을 요청한다. 교차 검증을 생략하거나 사용자 승인 전에 커밋 또는 푸시하지 않는다.

## Verification Attempt Limit

적대적 검증 실행 횟수는 Phase 검증과 문서 검증에 동일하게 적용한다.

- 하나의 검증 대상(Phase 1개 또는 문서 1개)에 대한 검토 CLI 실행은 **실패·오류·프롬프트 재작성·재검토를 모두 포함해 최대 3회**로 제한한다.
- 지적 보완 후의 재검증도 이 3회에 포함된다. 보완 사이클을 별도로 세지 않는다.
- 3회를 소진하면 추가 실행 대신 마지막 유효 검토 결과, 반영한 보완 내용, 미해결 지적과 남은 위험을 기록하고 사용자에게 보고한다.
- 3회 소진 시점에 미해결 Critical이 남아 있으면 다음 Phase로 진행하지 않고 사용자 판단을 요청한다.
- 실행 횟수와 각 회차의 결과(모델, 실행 일시, 유효 여부, 사유)를 검토 기록 문서에 남긴다.

## Cross-vendor Adversarial Review Sub-agent

- 메인 에이전트는 Phase 구현을 끝낸 뒤, 마지막 실질 구현자와 반대 벤더의 검증만 담당하는 별도 서브 에이전트를 실행한다.
- 서브 에이전트는 현재 Phase, 지정 제품 소스 파일, 해당 Phase의 `adversarialFocus`, `planRefs`만 프롬프트에 포함한다. 구현 세션 대화나 판단 근거는 전달하지 않는다.
- 검토자는 파일을 수정하지 않는다. 문서, 설정, 실험 산출물, Git 상태·브랜치·원격 저장소·커밋 이력과 셸 도구를 요청하거나 사용하지 않는다.
- 각 지적은 `Critical / Major / Minor` 등급, 정확한 재현 조건, 위반한 PLAN 조항을 포함하며 심각도순으로 반환한다.
- 검토자의 지적은 메인 에이전트가 검토한다. 유효하지 않은 지적은 근거와 함께 기록한다.
- 필요한 CLI의 응답 지연, 인증 실패, 네트워크 오류 등으로 검증하지 못하면 서브 에이전트는 오류 내용과 대체 검증안을 메인 에이전트에 반환한다.
- 문서 검증 시에는 검토자가 대상 문서와 상위 문서, `AGENTS.md`를 읽기 위해 읽기 전용 셸 명령(`cat`, `sed -n` 등)을 사용하는 것을 허용한다. Codex CLI에는 별도 파일 읽기 도구가 없어 셸을 전면 금지하면 검토가 불가능하다.
- 검토 CLI를 실행하는 외부 명령의 시간 제한은 기본 10분으로 설정한다.

### CV 프로젝트 공격 초점

이 프로젝트의 적대적 검증은 다음 축을 우선 공격 대상으로 삼는다.

- 엔진의 task-agnostic 위반 — 공통 루프에 태스크별 분기가 침투했는가
- target 규약 위반 — Dataset이 반환하는 형태가 6.1 규약과 다르거나, Loss/Metric이 특정 데이터셋 구조에 결합했는가
- Detection 일반성 — 가변 개수 박스, 다중 클래스, `collate_fn`, 빈 박스(N=0) 이미지 처리
- 공정 비교 통제 — split 간 해상도·정규화·augmentation·optimizer·seed가 실제로 고정되는가
- 오프라인 위반 — 네트워크 다운로드를 유발하는 경로가 남아 있는가
- 학습/평가 누수 — train/valid/test 분할 누수, 평가 시 `model.eval()`·`torch.no_grad()` 누락, metric 리셋 누락
- 재현성 — seed 고정 범위, 결과 기록과 config 저장이 실제 재현을 보장하는가

### 검토 명령

`<phase>`, `<changed-files>`, `<adversarial-focus>`, `<plan-refs>`는 현재 작업 내용으로 대체한다.

Claude Code가 마지막 구현자일 때 Codex 검토:

```bash
codex exec --model gpt-5.6-sol --sandbox read-only --cd "/mnt/d/projects/nampluskr/00_review/260818_cv-boilerplate" "You are an adversarial reviewer for <phase>. Your job is to break this code, not to confirm it works. Review only these product source-code files: <changed-files>. Attack these specific points: <adversarial-focus>. Validate against these plan clauses: <plan-refs>. Do not inspect documentation, configuration, experiment artifacts, Git status, branches, remotes, or commit history. For each finding, report severity (Critical/Major/Minor), exact reproduction conditions, and the violated plan clause. Order findings by severity. Do not modify files."
```

Codex가 마지막 구현자일 때 Claude Sonnet 검토:

```bash
claude -p "You are an adversarial reviewer for <phase>. Your job is to break this code, not to confirm it works. Review only these product source-code files: <changed-files>. Attack these specific points: <adversarial-focus>. Validate against these plan clauses: <plan-refs>. Do not inspect documentation, configuration, experiment artifacts, Git status, branches, remotes, commit history, or use Bash or any shell tool. For each finding, report severity (Critical/Major/Minor), exact reproduction conditions, and the violated plan clause. Order findings by severity. Do not modify files." --model sonnet --safe-mode --allowedTools "Read,Glob,Grep" --disallowedTools "Edit,Write,Bash" --permission-mode dontAsk --max-turns 5 --output-format json --no-session-persistence
```

Claude 검토는 `Read`, `Glob`, `Grep`만 허용하며 파일 변경과 셸 도구를 금지한다. 모델 접근이 거부되면 기본 모델로 조용히 폴백하지 않고 오류와 대체안을 보고한다. 필요하면 `--max-budget-usd`로 호출별 비용 상한을 둔다.

## Commit and Push Rules

- 원격 저장소는 `https://github.com/nampluskr/cv_boilerplate`로 확정되었다(2026-08-18 사용자 확인). `backlog.json` 생성 시 `remoteRepository`에 이 URL을 기록한다.
- 로컬 저장소 초기화(`git init`)와 원격 연결, 첫 푸시는 이후 초기 작업 Phase에서 사용자 승인 아래 수행한다. 그 전에는 커밋하지 않는다.
- 커밋은 하나의 완료된 Phase에 대응하며, 커밋 메시지에 Phase 번호와 핵심 변경 사항을 포함한다.
- 커밋 전에는 해당 Phase의 관련 검증을 실행한다.
- Phase 완료, 반대 벤더 교차 검증, 지적사항 수정 및 재검증을 마친 뒤 사용자에게 커밋 및 푸시 승인을 요청한다.
- 사용자의 명시적 승인 이후에만 커밋을 원격에 푸시한다.
- 다른 작업의 변경 사항을 임의로 포함, 되돌리기, 삭제하지 않는다.
- 데이터셋, 백본 가중치, 체크포인트, 실험 산출물은 커밋하지 않는다.
