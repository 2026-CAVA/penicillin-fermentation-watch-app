# Penicillin Fermentation Watch
Early anomaly warning and final yield prediction for penicillin fermentation (IndPenSim)

*For the Catholic Univ. of Korea 2026 1st BioHealth X AI Convergence Competition*

---
### 1. 개발 환경

- Windows 11
- Python 3.11.9
- Node.js 24.19.0 (웹 화면을 띄울 때만 필요)

---
### 2. 사용 패키지

- numpy: 2.4.6
- pandas: 3.1.0
- pyarrow: 26.0.0
- scikit-learn: 1.9.1
- xgboost: 3.2.0
- PyYAML: 6.0.3
- fastapi: 0.162.0
- uvicorn: 0.54.0
- httpx: 0.28.1
- pytest: 9.1.1

웹 화면(`web/`)은 React, Vite, Recharts를 사용한다. 버전은 `web/package.json`에 있다.

---
### 3. 설치 방법

1. 파이썬 설치

    본인의 개발 환경에 맞추어 3.11 버전을 설치하면 된다.

    https://www.python.org/downloads/

2. 비주얼 스튜디오 코드 설치

    본인의 개발 환경에 맞추어 설치하면 된다.

    https://code.visualstudio.com/download

3. 패키지 설치

    터미널을 열고 다음을 순서대로 입력한다.

    ```
    git clone https://github.com/2026-CAVA/penicillin-fermentation-watch-app.git
    ```
    ```
    cd penicillin-fermentation-watch-app
    ```
    ```
    python -m venv .venv
    ```
    ```
    .venv\Scripts\python -m pip install -r requirements.txt
    ```

    오류가 나지 않았다면 가상 환경 구축이 완료되었다.

    원본 데이터는 저장소에 포함되어 있지 않다. IndPenSim 공개 데이터의 `100_Batches_IndPenSim_V3.csv`와 `100_Batches_IndPenSim_Statistics.csv` 두 파일을 `data/` 폴더에 넣는다. 위치를 바꾸려면 `config.yaml`의 `paths`를 고치면 된다.

    비주얼 스튜디오 코드에서 해당 폴더를 연 뒤, F1을 눌러 파이썬 인터프리터 설정에서 `.venv` 가상 환경을 선택한다.

4. 실행

    모든 명령은 저장소 폴더에서 실행한다.

    1. 모델 학습

        비주얼 스튜디오 코드에서 <u>Ctrl + `</u> 를 눌러 터미널을 열고 다음을 입력한다.
        ```
        .venv\Scripts\python -m main all
        ```
        데이터 정리, 경보 기준, 수율 모델, 8시간 농도 예측, SHAP, 반복 교차검증을 차례로 실행한다(약 6분). 결과는 `data/cache/`와 `artifacts/`에 저장된다. 한 단계만 다시 돌리려면 `all` 대신 단계 이름(`warning`, `quality` 등)을 쓰면 된다. 단계 목록은 `--help`로 볼 수 있다.

        PCA 다변량 이상 탐지(T² / SPE)는 따로 실행한다.
        ```
        .venv\Scripts\python -m main pca
        ```

    2. API 서버 실행

        ```
        .venv\Scripts\python -m main serve
        ```
        http://127.0.0.1:8000 에 서버가 열리고, API 문서는 http://127.0.0.1:8000/docs 에서 볼 수 있다. `start_cava_server.bat`을 실행해도 된다.

    3. 웹 화면 실행

        서버를 켜 둔 채로 새 터미널을 열어 다음을 입력한다.
        ```
        npm --prefix web install
        ```
        ```
        npm --prefix web run dev
        ```
        터미널에 뜨는 주소(http://localhost:5173)에 접속하면 된다.

    4. 새 운전 데이터 예측

        학습에 쓰지 않은 운전 기록(CSV) 한 건을 예측할 수 있다. 시험용 파일은 `demo/`에 있다.
        ```
        .venv\Scripts\python -m main predict demo/run.csv
        ```

    5. 테스트

        ```
        .venv\Scripts\python -m pytest
        ```
