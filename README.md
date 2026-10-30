# Plan A

| 기능 | 내용 |
|---|---|
| (1) 조기경보 | 규칙 위험점수 R(t) + 정상 배치 ±3σ 밴드 → STABLE / WARNING / CRITICAL |
| (2) 품질예측 | 배치 전반부(0~115 h) 요약 피처 → XGBoost → 최종 총수율(kg) |
| (3) XAI | SHAP 원인 Top 3, 결정트리 규칙, 더미변수 검증 |
| (4) What-if | 슬라이더 재예측, 안전 범위로 되돌렸을 때 CRITICAL 감소 건수 |
| (5) 프론트 API | `main/service.py` (화면은 구현하지 않음) |

## 폴더 구조

```
back/                           ← 저장소 폴더. 모든 명령은 여기서 실행
├─ data/
│  ├─ 100_Batches_IndPenSim_V3.csv          원본 데이터 (저장소에 포함하지 않음)
│  ├─ 100_Batches_IndPenSim_Statistics.csv  원본 데이터
│  └─ cache/                                자동 생성 (정리된 시계열, 분할, 위험점수, 요약 표)
├─ .venv/                       가상환경 (각자 생성)
├─ config.yaml                  설정 전부 (열 정의, 안전 범위, 하이퍼파라미터, 시연 배치)
├─ requirements.txt
├─ main/
│  ├─ __main__.py               실행 진입점 (python -m main <단계>)
│  ├─ data.py                   로딩·수율 정답·오프라인 지연·분할·배치 요약 표
│  ├─ warning.py                (1) 밴드, R(t), 등급, 경보 규칙, 경보 지표
│  ├─ model.py                  (2) XGBoost·기준선·지표  (3) SHAP·규칙 트리·더미 검증
│  ├─ whatif.py                 (4) What-if
│  └─ service.py                (5) 프론트 API
├─ tests/test_main.py
└─ artifacts/                   자동 생성 (밴드, 경보 기준, 모델, 규칙)
```

## 설치

Python 3.11 기준입니다.

```powershell
git clone https://github.com/2026-CAVA/back.git
cd back
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

원본 CSV 두 개(`100_Batches_IndPenSim_V3.csv`, `100_Batches_IndPenSim_Statistics.csv`)를 `back/data/`에 넣습니다. 저장소에는 포함되어 있지 않습니다. 위치를 바꾸려면 `config.yaml`의 `paths`를 고칩니다.

## 실행

```powershell
cd back
.venv\Scripts\python -m main all
```

`all`은 표의 앞 4단계(`prepare` → `warning` → `quality` → `explain`)를 차례로 실행합니다(약 45초). `check`와 `test`는 따로 실행합니다. 결과는 터미널에 출력되고, 다음 단계가 쓸 파일은 `data/cache/`와 `artifacts/`에 저장됩니다.

| 단계 | 하는 일 |
|---|---|
| `prepare` | 원본 CSV 정리, 학습/검증/시험 분할, 컷오프 결정 | 
| `warning` | 밴드, R(t), 경보 경계값 결정·고정 |
| `quality` | 요약 표, XGBoost 학습(+fold 모델), 기준선 비교, 컷오프 민감도 |
| `explain` | SHAP 중요도, 결정트리 규칙, 더미변수 검증 |
| `check` | 데이터 점검 출력 (요청할 때만) |
| `test` | 시험 세트 최종 평가 (**아래 규칙 참고**) |

```powershell
.venv\Scripts\python -m main quality            # 한 단계만
.venv\Scripts\python -m main warning quality    # 여러 단계
.venv\Scripts\python -m main --help
```

`config.yaml`을 고쳤으면 그 설정을 쓰는 단계부터 다시 실행합니다. 예: 안전 범위를 바꿨으면 `warning quality explain`.

### 정상 실행 확인용 값

같은 버전으로 설치했다면 `all`에서 아래 값이 나와야 합니다.

```
R(t) cuts: WARNING >= 1, CRITICAL >= 14; band rule: >= 2 vars for 2 points (24 min)
xgboost                496    451   17.0  0.564
```

## 테스트

```powershell
.venv\Scripts\python -m pytest
```

`pytest` 단독이 아니라 `python -m pytest`로 실행해야 `main`을 찾습니다. 테스트는 `data/cache/`와 `artifacts/`를 읽으므로 `all`을 먼저 실행해야 합니다.

검사 내용: 세트 간 배치 중복 없음, 금지 열이 입력에 없음, 밴드·허용폭을 학습 배치로만 계산, 학습 배치는 자기를 뺀 fold 모델로 예측, 컷오프 이후 값과 아직 나오지 않은 오프라인 값 미사용, 시험 단계가 경보 기준을 고치지 않음, What-if 동작, 가상 배치로 시험 단계 전체 실행.

## 시험 세트 평가 규칙

시험 세트(정상 15 + 고장 10배치)는 **다 고친 뒤 실제 성능을 한 번 재는 용도**입니다.

코드 성능 평가를 위해서는 검증 세트(정상 15배치: 레시피·운전자·APC 그룹별 5배치씩, 고장 배치 없음)를 사용하게끔 되어 있습니다.

- 실행 전에 `config.yaml`의 `demo_batches`에 발표용 시험 배치를 넣어야 합니다. 비어 있으면 `test`가 실행되지 않습니다.
- 한 번 실행하면 `artifacts/test_evaluated.json`이 생기고 그 뒤로는 다시 실행되지 않습니다. 이 장치는 PC마다 따로 있으므로 팀에서 누가 언제 실행할지 정해 둡니다.

## 프론트 API

```python
from main.service import list_batches, batch_view, run_whatif

list_batches()                    # [{batch_id, group, is_fault, split, tag}, ...]
batch_view(92)                    # 배치 하나의 화면 데이터 (JSON 직렬화 가능한 dict)
run_whatif(92, {"Fs_tot": 0.05})  # 슬라이더: 전반부 설탕 공급 +5%
```

- 저장소 폴더(`back`)를 작업 위치로 두고 실행합니다. 학습은 하지 않고 `data/cache/`와 `artifacts/`만 읽습니다.
- `batch_view` 주요 키: `pred_source`(학습 배치는 `"oof"` = 그 배치를 빼고 학습한 모델, 그 외 `"final"`), `tag`, `light_final` / `light_worst`, `y_pred_kg` / `y_true_kg`, `t`, `series`(변수별 값·밴드·안전선), `risk`(R(t)·등급·경보 원인 Top 3), `shap_top3`, `rule`, `whatif`(슬라이더 범위·반응 곡선, 규칙 What-if).
- 화면 문구(꼬리표, 안내, 슬라이더 이름)는 영어입니다. 슬라이더 이름은 `config.yaml`의 `sliders`에서 바꿉니다.
- 설탕 공급(`Fs_tot`)을 늘리면 늘어난 양(L)만큼 전반부 끝 부피 `V_last`(L)와 무게 `Wt_last`(kg, 밀도 약 1 kg/L 가정)도 함께 늘린 뒤 예측합니다. `run_whatif` 결과의 `inputs`에 `V_last`·`Wt_last`가 `linked_to: "Fs_tot"`로 함께 나오고, 반응 곡선도 같은 방식으로 계산합니다. 비율은 `main/whatif.py`의 `COUPLED`에서 바꿉니다.

## Appendix

| 항목 | 결정 | 이유 |
|---|---|---|
| 수율 정답 | 시계열로 계산: `P_T * V_T + sum_t P_t * abs(Fremoved_t) * dt` (kg) | Statistics 파일 수율은 시계열과 맞지 않음(배치별 상관 0.33)·예측 불가(R² ≤ 0) |
| 원본 헤더 | 읽을 때 복구 (`data.read_header`) | 헤더 한 칸 이름에 쉼표가 있어 데이터보다 2칸 많음. Raman 스펙트럼 열은 읽지 않음 |
| 오프라인 측정값 | 채취 4 h 뒤부터 사용 | 실제 분석 지연 반영, 미래 정보 누수 방지 |
| 안전 범위 | IndPenSim 기준 범위 + 없는 변수는 학습 정상 배치 0.5~99.5% 분위수 | 기준 범위만으로는 고장을 거의 못 잡음 |
| PAA 가중치 | R(t)에서 0 (화면 표시는 유지) | 정상 배치에서도 자주 넘어 R(t)가 100점에 붙음 |
| 컷오프 | 모든 배치 115 h 고정 (+69 h·161 h 민감도) | 배치별 50%는 진행 중에 알 수 없는 총 길이가 섞임 |
| 기질 S | 표시 전용 | 실공정에서 온라인 측정 불가 |
| 경보 기준 | 검증 정상 배치 오경보율 목표로 결정, 고장 10배치는 전부 시험용 | - |