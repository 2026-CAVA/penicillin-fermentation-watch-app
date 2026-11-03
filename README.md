# CAVA

| 기능 | 내용 |
|---|---|
| (1) 조기경보 | 규칙 위험점수 R(t) + 정상 배치 ±3σ 밴드 → STABLE / WARNING / CRITICAL |
| (2) 품질예측 | 배치 진행 중 시점별(46~161 h) 요약 피처 → 범위를 자른 ridge(115 h 전) / XGBoost(115 h부터) → 최종 총수율(kg) |
| (3) XAI | SHAP 원인 Top 3, 결정트리 규칙, 더미변수 검증 |
| (4) What-if | 슬라이더 재예측, 안전 범위로 되돌렸을 때 CRITICAL 감소 건수 |
| (5) API | `main/api.py` (HTTP) · `main/service.py` (화면은 구현하지 않음) |

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
│  ├─ model.py                  (2) XGBoost·ridge·기준선·지표  (3) SHAP·규칙 트리·더미 검증
│  ├─ forecast.py               8 h 앞 페니실린 농도 예측
│  ├─ whatif.py                 (4) What-if, 최적화 추천
│  ├─ cv.py                     반복 교차검증 (python -m main cv)
│  ├─ predict.py                새 운전 데이터 예측 (python -m main predict)
│  ├─ service.py                (5) API가 쓰는 함수 (화면 데이터 가공)
│  └─ api.py                    (5) HTTP API (python -m main serve)
├─ tests/test_main.py
├─ results/                     예전 고정 분할 시험 결과 기록 (2026-11-02)
└─ artifacts/                   자동 생성 (밴드, 경보 기준, 모델, 규칙, 반복 CV 결과)
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

`all`은 표의 앞 7단계(`prepare` → `warning` → `quality` → `cutoffs` → `forecast` → `explain` → `cv`)를 차례로 실행합니다(약 6분, 대부분 `cv`). `check`는 따로 실행합니다. 결과는 터미널에 출력되고, 다음 단계가 쓸 파일은 `data/cache/`와 `artifacts/`에 저장됩니다.

| 단계 | 하는 일 |
|---|---|
| `prepare` | 원본 CSV 정리, 정상 90배치를 학습용으로 5-fold 분할(고장 10배치는 평가 전용), 컷오프 결정 |
| `warning` | 밴드, R(t), 경보 경계값 결정·고정 |
| `quality` | 요약 표, XGBoost 학습(+fold 모델) |
| `cutoffs` | 컷오프(46 ~ 161 h)마다 수율 모델 학습(46 ~ 92 h는 범위를 자른 ridge, 115 h부터 XGBoost), 반복 CV로 제공 여부 결정 → 시점별 예측 추이 |
| `forecast` | 8 h 뒤까지 페니실린 농도 예측 모델, 지속·선형 연장 기준선 비교, 80 % 범위 |
| `explain` | SHAP 중요도, 결정트리 규칙, 더미변수 검증, SHAP·Permutation 교차 검증 (fold 모델을 자기 fold에서) |
| `cv` | **공식 성능.** 정상 90배치 반복 교차검증: 파이프라인 전체를 4/5로 다시 만들어 나머지 1/5과 고장 배치로 평가, 5회 반복 (결과 `artifacts/cv.json`) |
| `check` | 데이터 점검 출력 (요청할 때만) |
| `test` | (종료) 2026-11-02 고정 분할에서 1회 실행, 기록은 `results/`. 지금은 실행되지 않음 |
| `serve` | HTTP API (`http://127.0.0.1:8000`, 문서는 `/docs`) |
| `predict` | 데이터에 없는 운전 한 건(CSV)을 학습 없이 예측 (아래 "새 데이터로 예측") |

```powershell
.venv\Scripts\python -m main quality            # 한 단계만
.venv\Scripts\python -m main warning quality    # 여러 단계
.venv\Scripts\python -m main --help
```

`config.yaml`을 고쳤으면 그 설정을 쓰는 단계부터 다시 실행합니다. 예: 안전 범위를 바꿨으면 `warning quality cutoffs explain`.

### 결과 예시

같은 버전으로 설치했다면 `all`에서 아래 값이 나와야 합니다.

```
R(t) cuts: WARNING >= 28, CRITICAL >= 68, held 5 points (60 min); band rule: off (no setting meets the target); tuned on train normal batches
  115 h       560    19   0.498   0.476      742  yes       xgboost
alarm warning+  normal batches alarmed 10% (7%-13%), faults detected 78% (74%-80%), delay after onset 4.5 (3.3-7.2) h, cuts [23, 84]
```

### 새 데이터로 예측 (`predict`)

데이터에 없는 운전 한 건을 CSV로 넣으면, 학습 없이 저장된 모델로 지금 시각 기준 결과를 냅니다.

```powershell
.venv\Scripts\python -m main predict run.csv --control apc --t 100 --out result.json
```

- **입력:** 
  - t = 0부터 지금까지 한 운전의 시계열 CSV
  - 열 이름: `config.yaml`의 `columns` 이름(`t`, `Fg`, `Fs`, `DO`, `pH`, `T`, …, 오프라인 `P_off`·`PAA_off`·`NH3_off`·`X_off`·`Visc_off`)을 씀
  - 오프라인 열은 측정하지 않은 행을 비워 둠, 행 간격: 0.2 h, 그 밖의 열(P, fault_ref 등)은 읽지 않음
- **옵션:**
  - `--control`은 제어 방식(`recipe` 기본, `operator`, `apc`)
  - `--t`를 주면 그 시각까지만 씀
  - `--out`을 주면 전체 결과를 JSON으로도 저장함
- **출력:**
  - 지금 경보 등급과 R(t), 그 원인 변수
  - 지금까지 가장 나빴던 등급과 첫 경보 시각
  - 도달한 마지막 컷오프의 예상 수율, 80 % 범위, 기여도 상위 5개(46 h 전에는 없음)
  - 8 h 앞 페니실린 농도 경로
- **예시 입력 만들기:** 
  - 있는 배치를 잘라 쓰거나 값을 바꿔 가상 상황을 만들 수 있음
  ```powershell
  .venv\Scripts\python -c "from main.data import load_ts; d = load_ts(); d[(d.batch == 92) & (d.t <= 100)].to_csv('run.csv', index=False)"
  ```
- **주의:**
  - 정상 배치 1~90은 최종 모델이 학습에 쓴 배치라, 이 명령으로 넣으면 실제보다 잘 맞음
  - 그 배치의 공정한 예측은 API(`pred_source: "oof"`)에서 확인

## 테스트

```powershell
.venv\Scripts\python -m pytest
```

`pytest` 단독이 아니라 `python -m pytest`로 실행해야 `main`을 찾음

테스트는 `data/cache/`와 `artifacts/`를 읽으므로 `all`을 먼저 실행해야 함

검사 내용: 세트 간 배치 중복 없음, 금지 열이 입력에 없음, 밴드·허용폭을 학습 배치로만 계산, 학습 배치는 자기를 뺀 fold 모델로 예측, 컷오프 이후 값과 아직 나오지 않은 오프라인 값 미사용, 115 h 이전 ridge 기여도의 합 = 예측값, 기여도 라벨에 계산 구간 표시, 새 운전 CSV 예측(`predict`)이 API와 같은 값, 시험 단계가 경보 기준을 고치지 않음, What-if 동작, 가상 배치로 시험 단계 전체 실행.

## API

### 서버 실행

```powershell
.venv\Scripts\python -m main serve                                   # 이 PC에서만: http://127.0.0.1:8000
.venv\Scripts\python -m uvicorn main.api:app --host 0.0.0.0 --port 8000  # 같은 네트워크의 다른 PC에서도 접속
```

- 서버는 학습하지 않고 `data/cache/`와 `artifacts/`만 읽음. `all`을 한 번 돌린 폴더에서 실행
- 경로별 입력·출력은 서버를 띄운 뒤 `/docs`에서 직접 눌러 볼 수 있음
- CORS는 모든 출처를 허용함. 외부에 공개할 서버라면 `main/api.py`의 `allow_origins`를 접속할 주소로 좁힘
- 외부 접속 주소(누가, 어디에 띄울지)는 아직 정하지 않음

### 경로

모든 경로에서 `t`(시간, h)를 주면 그 시각까지의 데이터만 씁니다(진행 중 배치 재생). 미래 값이 응답에 섞이지 않는 것은 테스트로 확인합니다.

| 경로 | 내용 |
|---|---|
| `GET /batches?t=` | 배치 100개 요약 |
| `GET /batches/{id}?t=` | 요약 + 시점별 시계열(`series`), 수율 기여도 기준값(`shapBase`), 경보 설명 한 문장 |
| `GET /batches/{id}/shap?t=&k=` | 수율 기여도 상위 k개(기본 5, `k=0`이면 전부). 첫 예측 시점(46 h) 전에는 빈 목록 |
| `GET /batches/{id}/contrib?t=` | 시각 t의 위험점수 R(t)를 변수별로 나눈 비율 (시점마다 바뀌는 원인) |
| `GET /batches/{id}/events?all=&t=` | 1 h 이상 이어진 이탈 목록 (기본은 경보와 겹친 것만) |
| `GET /batches/{id}/forecast?t=` | t부터 8 h 뒤까지 페니실린 농도(g/L) 경로, 0.5 h 간격, 80 % 범위 |
| `GET /compare?ids=6,92&vars=DO,pH` | 여러 배치 시계열 나란히 (하나만 주면 같은 그룹 정상 배치를 기준으로 붙임) |
| `POST /simulate` | `{"batch_id": 6, "changes": {"Fpaa_tot": 0.05}}` → 슬라이더를 바꿨을 때 예상 수율 변화 |
| `POST /batches/{id}/optimize` | `{"max_change": 0.2}`(생략 가능) → 슬라이더 추천값 (참고용) |
| `GET /model` | 모델 제목, 시점별 모델(`cutoffs`), 반복 CV 성능(`cv`), 경보 기준, 시연 배치 |

### 값 읽는 법

- **배치 번호** -> `id` 숫자(1~100)[91~100이 고장 배치]
- **예측 수율**(kg): 
  - `predictedFinal` -> 그 시각까지 도달한 가장 늦은 컷오프(46·69·92·115·138·161 h)의 예측
  - `predictedRange` -> 80 % 범위(과거 오차의 80 % 분위수, 반복 CV에서 실제 포함률 79~85 %).
- **115 h 이전 예측(46·69·92 h)은 초기 추정**(ridge, R² 약 0.4)
  - 컷오프별 모델은 `GET /model`의 `cutoffs`에서 확인 (`model`이 `"ridge_clip"`이면 초기 추정)
- **수율 기여도**: `shapBase + 기여도 합 = 그 시점 예측`
  - 115 h부터는 XGBoost TreeSHAP, 그 전에는 ridge 계수 × 값
  - 라벨에 계산 구간이 들어 있음(예: "0-46 h mean Oxygen uptake rate")
- **위험점수**
  - `score`: R(t) [0~100] 
  - `scoreCuts`의 `[WARNING, CRITICAL]`(지금 28, 68) 이상이 1 h 이어지면 `caution` / `anomaly`
- **단위**: 온도 K, DO mg/L, 페니실린 농도 g/L, 유량 L/h, 총량 L.
- **정상 배치의 예측은 모두 그 배치를 빼고 학습한 모델의 값**(`pred_source: "oof"`)
  - 고장 배치는 최종 모델(`"final"`)

## Appendix

| 항목 | 결정 | 이유 |
|---|---|---|
| 수율 정답 | 시계열로 계산: `P_T * V_T + sum_t P_t * abs(Fremoved_t) * dt` (kg) | Statistics 파일 수율은 시계열과 맞지 않음(배치별 상관 0.33)·예측 불가(R² ≤ 0) |
| 원본 헤더 | 읽을 때 복구 (`data.read_header`) | 헤더 한 칸 이름에 쉼표가 있어 데이터보다 2칸 많음. Raman 스펙트럼 열은 읽지 않음 |
| 오프라인 측정값 | 채취 4 h 뒤부터 사용 | 실제 분석 지연 반영, 미래 정보 누수 방지 |
| 안전 범위 | IndPenSim 기준 범위 + 없는 변수는 학습 정상 배치 0.5~99.5% 분위수 | 기준 범위만으로는 고장을 거의 못 잡음 |
| PAA 가중치 | R(t)에서 0 (화면 표시는 유지) | 정상 배치에서도 자주 넘어 R(t)가 100점에 붙음 |
| 컷오프 | 모든 배치 같은 절대 시각 (설명·What-if는 115 h, 예측 추이는 46~161 h) | 배치별 50%는 진행 중에 알 수 없는 총 길이가 섞임 |
| 기질 S | 표시 전용 | 실공정에서 온라인 측정 불가 |
| 경보 기준 | 정상 90배치 오경보율 목표로 결정, 고장 10배치는 평가 전용 | 반복 CV에서 정상 배치 경보 비율이 목표 근처 |
| 시점별 수율 모델 | 46~92 h 범위를 자른 ridge, 115 h부터 XGBoost | 반복 CV: 이른 시점은 약한 선형 신호의 합이라 ridge가 5회 모두 우세, 늦은 시점은 PAA 문턱형 관계라 트리가 동률 이상. 피처를 학습 1~99 % 분위수로 잘라 이상 배치(67번)에서 선형 외삽이 터지는 것을 막음. spline ridge도 비교했으나 138 h에서만 근소 우세라 채택 안 함 |
