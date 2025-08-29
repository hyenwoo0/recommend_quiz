import torch
import torch.nn as nn
from fastapi import FastAPI, Query, HTTPException
from typing import List
import uvicorn
import mysql.connector
from mysql.connector import Error
import numpy as np
import joblib
import os
from dotenv import load_dotenv # .env 파일을 위해 추가

# .env 파일에서 환경변수를 로드합니다.
load_dotenv()

# --- DB 연동 및 데이터 로드 ---

def load_data_from_db():
    """DB에서 사용자, 퀴즈, 기록 데이터를 모두 불러옵니다."""
    conn = None
    cursor = None
    try:
        # .env 파일에서 DB 접속 정보를 안전하게 불러옵니다.
        db_host = os.getenv("DB_HOST", "127.0.0.1")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_name = os.getenv("DB_DATABASE")

        # 필수 정보가 있는지 확인
        if not all([db_user, db_password, db_name]):
            print("오류: .env 파일에 DB 접속 정보(DB_USER, DB_PASSWORD, DB_DATABASE)가 없습니다.")
            return [], [], []

        conn = mysql.connector.connect(
            host=db_host,
            port=3306,
            user=db_user,
            password=db_password,
            database=db_name
        )
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, quiz_id, is_correct, time_taken FROM quiz_log")
        history = cursor.fetchall()

        if not history:
            return [], [], []

        users = sorted(list(set(row[0] for row in history)))
        quizzes = sorted(list(set(row[1] for row in history)))

        return users, quizzes, history
    except Error as e:
        print(f"DB 연결/쿼리 오류: {e}")
        return [], [], []
    finally:
        # (수정) cursor와 conn을 더 안전하게 닫습니다.
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()


# API 서버 시작 시 DB에서 데이터를 로드합니다.
users, quizzes, history = load_data_from_db()

if not history:
    print("경고: DB에서 데이터를 불러오지 못했습니다. API가 정상 작동하지 않을 수 있습니다.")

# 사용자 및 퀴즈 ID → 인덱스 매핑
user_idx = {u: i for i, u in enumerate(users)}
quiz_idx = {q: i for i, q in enumerate(quizzes)}


# --- 모델 정의 및 로드 ---

class QuizRecModel(torch.nn.Module):
    """
    사용자, 퀴즈, 그리고 풀이 시간 정보를 바탕으로
    사용자의 퀴즈 정답 확률을 예측하는 추천 모델
    """
    def __init__(self, n_users, n_quizzes, emb_dim=16):
        """
        모델의 구조를 정의하고 필요한 레이어들을 초기화합니다.

        Args:
            n_users (int): 전체 사용자의 수. 임베딩 레이어의 크기를 결정합니다.
            n_quizzes (int): 전체 퀴즈의 수. 임베딩 레이어의 크기를 결정합니다.
            emb_dim (int, optional): 각 사용자/퀴즈를 표현할 벡터의 차원(크기).
                                     기본값은 16입니다.
        """
        # 부모 클래스인 nn.Module의 생성자를 호출하여 PyTorch 모델로 초기화합니다.
        super().__init__()

        # --- 1. 임베딩 레이어 정의 ---
        # 각 사용자를 고유한 'emb_dim' 차원의 벡터로 변환하기 위한 조회 테이블(lookup table)입니다.
        # 이 벡터는 사용자의 잠재적인 특성(예: 지식 수준, 강점)을 학습하게 됩니다.
        self.user_emb = torch.nn.Embedding(n_users, emb_dim)

        # 각 퀴즈를 고유한 'emb_dim' 차원의 벡터로 변환하기 위한 조회 테이블입니다.
        # 이 벡터는 퀴즈의 잠재적인 특성(예: 난이도, 카테고리)을 학습하게 됩니다.
        self.quiz_emb = torch.nn.Embedding(n_quizzes, emb_dim)

        # --- 2. 완전 연결(Fully Connected) 레이어 정의 ---
        # 임베딩된 정보와 추가 정보를 종합하여 최종 예측을 수행하는 신경망 부분입니다.
        self.fc = torch.nn.Sequential(
            # 입력: 사용자 벡터(16) + 퀴즈 벡터(16) + 풀이 시간(1) = 33차원
            # 출력: 32차원으로 정보를 요약합니다.
            torch.nn.Linear(emb_dim * 2 + 1, 32),

            # 활성화 함수(ReLU): 모델에 비선형성을 추가하여 더 복잡한 관계를 학습하게 합니다.
            torch.nn.ReLU(),

            # 입력: 32차원
            # 출력: 최종 예측을 위해 1차원으로 정보를 압축합니다.
            torch.nn.Linear(32, 1),

            # 활성화 함수(Sigmoid): 최종 출력값을 0과 1 사이의 '확률' 값으로 변환합니다.
            # 이 값이 곧 모델이 예측하는 '정답 확률'이 됩니다.
            torch.nn.Sigmoid()
        )

    def forward(self, user, quiz, time):
        """
        실제 데이터가 입력되었을 때 모델의 계산 과정을 정의합니다. (순전파)

        Args:
            user (torch.Tensor): 사용자 ID(인덱스) 텐서.
            quiz (torch.Tensor): 퀴즈 ID(인덱스) 텐서.
            time (torch.Tensor): 정규화된 풀이 시간 텐서.

        Returns:
            torch.Tensor: 각 사용자-퀴즈 쌍에 대한 예측된 정답 확률 텐서.
        """
        # --- 1. 임베딩 벡터 조회 ---
        # 입력된 사용자 ID에 해당하는 임베딩 벡터를 조회합니다.
        u = self.user_emb(user)
        # 입력된 퀴즈 ID에 해당하는 임베딩 벡터를 조회합니다.
        q = self.quiz_emb(quiz)

        # --- 2. 특징 벡터 결합 ---
        # 사용자 벡터, 퀴즈 벡터, 풀이 시간 정보를 하나의 긴 벡터로 이어 붙입니다.
        # dim=1은 각 샘플(행)별로 특징들을 옆으로 나란히 붙이라는 의미입니다.
        x = torch.cat([u, q, time], dim=1)

        # --- 3. 최종 확률 예측 ---
        # 결합된 벡터를 fc 레이어에 통과시켜 최종 정답 확률을 계산합니다.
        # .squeeze(1)은 결과 텐서의 불필요한 차원(크기가 1인 차원)을 제거하여
        # [batch_size, 1] 형태를 [batch_size] 형태로 만듭니다.
        return self.fc(x).squeeze(1)


# 장치 설정, 모델 및 스케일러 로드
device = torch.device("cpu")
# gpu 사용 가능하면 cuda로 변경
# n_users와 n_quizzes가 0일 경우를 대비한 예외 처리
if not users or not quizzes:
    print("오류: 사용자 또는 퀴즈 데이터가 없어 모델을 로드할 수 없습니다.")
    model = None
    scaler = None
    # model, scaler 변수를 None으로 설정해 비정상적인 종료를 막는다.
else:   # 데이터가 정상적으로 있을 떄
    model = QuizRecModel(len(users), len(quizzes)).to(device)
    # 데이터를 넣어 실체를 생성, .to(device) 생성된 모델을 지정된 장치로 이동
    try:
        model.load_state_dict(torch.load("quizrec_with_time_db.pt", map_location=device))
        model.eval()
    except FileNotFoundError:
        print("오류: quizrec_with_time_db.pt 파일을 찾을 수 없습니다.")
        model = None

    try:
        scaler = joblib.load('time_scaler.pkl')
    except FileNotFoundError:
        print("경고: time_scaler.pkl 파일을 찾을 수 없습니다. 스케일링 없이 진행됩니다.")
        scaler = None

# --- FastAPI 앱 정의 ---

app = FastAPI()


def recommend_quizzes_logic(
        user_id_str: str,
        candidate_quiz_ids: List[str],
        avg_time: float,
        top_n: int = 3
):
    if not model:
        raise HTTPException(status_code=503, detail="Model is not available.")
    if user_id_str not in user_idx:
        raise HTTPException(status_code=404, detail=f"User '{user_id_str}' not found in the data.")

    if scaler:
        time_scaled = scaler.transform(np.array([[avg_time]]))
        solved_times_scaled = [time_scaled[0][0]] * len(candidate_quiz_ids)
    else:
        solved_times_scaled = [avg_time] * len(candidate_quiz_ids)

    ui = torch.LongTensor([user_idx[user_id_str]] * len(candidate_quiz_ids)).to(device)
    qi = torch.LongTensor([quiz_idx[q] for q in candidate_quiz_ids]).to(device)
    ti = torch.FloatTensor(solved_times_scaled).unsqueeze(1).to(device)

    with torch.no_grad():
        prob_correct = model(ui, qi, ti).cpu().numpy()

    prob_wrong = 1 - prob_correct
    ranked = sorted(zip(prob_wrong, candidate_quiz_ids), reverse=True)
    return [qid for _, qid in ranked[:top_n]]


@app.get("/recommend")
def recommend_endpoint(
        user_id: str = Query(..., description="추천을 받을 사용자 ID"),
        top_n: int = Query(3, description="추천할 퀴즈 개수")
):
    attempted = {q for u, q, _, _ in history if u == user_id}
    candidates = [q for q in quizzes if q not in attempted]

    if not candidates:
        return {"user_id": user_id, "recommended_quizzes": []}

    user_times = [t for u, _, _, t in history if u == user_id]
    avg_time = sum(user_times) / len(user_times) if user_times else 30.0

    recommendations = recommend_quizzes_logic(user_id, candidates, avg_time, top_n)
    return {"user_id": user_id, "recommended_quizzes": recommendations}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
