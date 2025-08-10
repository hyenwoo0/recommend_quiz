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
    def __init__(self, n_users, n_quizzes, emb_dim=16):
        super().__init__()
        self.user_emb = torch.nn.Embedding(n_users, emb_dim)
        self.quiz_emb = torch.nn.Embedding(n_quizzes, emb_dim)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 2 + 1, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, user, quiz, time):
        u = self.user_emb(user)
        q = self.quiz_emb(quiz)
        x = torch.cat([u, q, time], dim=1)
        return self.fc(x).squeeze(1)


# 장치 설정, 모델 및 스케일러 로드
device = torch.device("cpu")
# (수정) n_users와 n_quizzes가 0일 경우를 대비한 예외 처리
if not users or not quizzes:
    print("오류: 사용자 또는 퀴즈 데이터가 없어 모델을 로드할 수 없습니다.")
    model = None
    scaler = None
else:
    model = QuizRecModel(len(users), len(quizzes)).to(device)
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
