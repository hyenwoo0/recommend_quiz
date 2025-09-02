# 딥러닝 기반 개인 맞춤형 퀴즈 추천 시스템

## 1. 프로젝트 개요 (Overview)

사용자의 퀴즈 풀이 이력(정답 여부, 풀이 시간)을 PyTorch 딥러닝 모델로 학습하여, 개인의 실력에 맞는 **가장 도전적인 퀴즈를 추천하는 API 서버**입니다. 이 시스템은 사용자가 자신의 취약점을 효율적으로 학습할 수 있도록 돕는 것을 목표로 합니다.

## 2. 주요 기능 (Features)

-   **개인화된 퀴즈 추천**: 사용자의 과거 학습 데이터를 기반으로 '틀릴 확률'이 가장 높은 퀴즈를 추천합니다.
-   **딥러닝 모델 기반**: 사용자와 퀴즈의 잠재적 특성을 임베딩하여 복잡한 상호작용을 학습합니다.
-   **모듈화된 구조**: 역할과 책임에 따라 명확하게 분리된 폴더 구조로 유지보수 및 확장이 용이합니다.
-   **FastAPI 기반**: FastAPI를 통해 빠르고 효율적인 API 서버를 구축하고, 자동 생성되는 문서를 통해 쉽게 테스트할 수 있습니다.

## 3. 프로젝트 구조 및 파일 설명


recommend_quiz/
├── 📁 app/               # API 서버 애플리케이션
│   ├── 📁 api/
│   │   └── 📜 recommend.py
│   └── 📜 main.py
├── 📁 core/              # DB 연결 등 핵심 공통 기능
│   └── 📜 db.py
├── 📁 models/            # AI 모델 관련 파일
│   ├── 📁 trained_models/
│   └── 📜 quiz_rec_model.py
├── 📁 scripts/           # 모델 학습 및 데이터 관리 스크립트
│   ├── 📜 insert_data.py
│   └── 📜 train_model.py
├── 📜 .env
└── 📜 README.md


### 코드 기능 설명

-   **`app/main.py`**:
    -   FastAPI 애플리케이션을 실행하는 **메인 파일**입니다.
    -   서버 시작 시 학습된 모델을 로드하고, `/recommend` API 엔드포인트를 통해 사용자의 요청을 받아 처리합니다.

-   **`app/api/recommend.py`**:
    -   실제 **추천 로직**을 수행하는 함수가 들어있습니다.
    -   사용자가 아직 풀지 않은 퀴즈에 대해 모델의 예측을 수행하고, '틀릴 확률' 순으로 정렬하여 최종 결과를 반환합니다.

-   **`core/db.py`**:
    -   MySQL 데이터베이스와의 **연결 및 데이터 조회**를 담당합니다.
    -   `load_data_from_db` 함수를 통해 모델 학습이나 추천에 필요한 데이터를 불러옵니다.

-   **`models/quiz_rec_model.py`**:
    -   PyTorch로 구현된 **딥러닝 모델(`QuizRecModel`)의 구조**가 정의된 파일입니다.
    -   사용자, 퀴즈 임베딩 레이어와 최종 정답 확률을 계산하는 신경망으로 구성되어 있습니다.

-   **`scripts/train_model.py`**:
    -   `QuizRecModel`을 **학습시키는 스크립트**입니다.
    -   DB에서 전체 데이터를 불러와 모델을 훈련시킨 후, 학습된 모델 파일(`.pt`)과 데이터 스케일러(`.pkl`)를 `models/trained_models/` 폴더에 저장합니다.

-   **`scripts/insert_data.py`**:
    -   모델 학습 및 테스트에 필요한 **샘플 데이터를 DB에 삽입**하는 유틸리티 스크립트입니다.

## 4. 주요 기술 (Tech Stack)

-   **Backend**: FastAPI, Uvicorn
-   **Machine Learning**: PyTorch, Scikit-learn
-   **Database**: MySQL
-   **Environment**: Conda

## 5. 실행 방법 (How to Run)

1.  **가상 환경 생성 및 활성화**
    ```bash
    conda create -n recommend_env python=3.9
    conda activate recommend_env
    ```

2.  **필요 라이브러리 설치**
    ```bash
    pip install fastapi uvicorn torch mysql-connector-python python-dotenv scikit-learn
    ```

3.  **.env 파일 설정**
    -   프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 아래와 같이 DB 접속 정보를 입력합니다.
    ```
    DB_HOST=127.0.0.1
    DB_USER=your_db_user
    DB_PASSWORD=your_db_password
    DB_DATABASE=your_db_name
    ```

4.  **(최초 1회) 모델 학습 실행**
    -   프로젝트 루트 디렉토리에서 아래 명령어를 실행하여 AI 모델을 학습시키고 결과 파일을 생성합니다.
    ```bash
    python scripts/train_model.py
    ```

5.  **API 서버 실행**
    -   프로젝트 루트 디렉토리에서 아래 명령어를 실행합니다.
    ```bash
    uvicorn app.main:app --reload
    ```

6.  **API 테스트**
    -   웹 브라우저에서 `http://127.0.0.1:8000/docs` 로 접속하여 API 문서를 확인하고 직접 테스트할 수 있습니다.
