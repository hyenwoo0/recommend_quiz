# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

def load_data_from_db():
    """DB에서 사용자, 퀴즈, 기록 데이터를 모두 불러옵니다."""
    conn = None
    cursor = None
    try:
        db_host = os.getenv("DB_HOST", "127.0.0.1")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_name = os.getenv("DB_DATABASE")

        if not all([db_user, db_password, db_name]):
            print("오류: .env 파일에 DB 접속 정보가 없습니다.")
            return [], [], []

        conn = mysql.connector.connect(
            host=db_host, port=3306, user=db_user,
            password=db_password, database=db_name
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
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()


# 1) DB에서 모든 데이터 로드
users, quizzes, history = load_data_from_db()

if not history:
    print("오류: DB에서 데이터를 불러오지 못했거나 데이터가 없습니다. 스크립트를 종료합니다.")
    exit()

print(f"총 {len(users)}명의 사용자와 {len(quizzes)}개의 퀴즈 데이터를 로드했습니다.")

# 2) 사용자 및 퀴즈 ID → 인덱스 매핑
user_to_idx = {u: i for i, u in enumerate(users)}
quiz_to_idx = {q: i for i, q in enumerate(quizzes)}

# 3) 데이터셋 준비
X_user, X_quiz, X_time, y = [], [], [], []
for u, q, c, t in history:
    if u in user_to_idx and q in quiz_to_idx:
        X_user.append(user_to_idx[u])
        X_quiz.append(quiz_to_idx[q])
        X_time.append(t)
        y.append(c)

# 4) 시간 데이터 스케일링 및 스케일러 저장
time_scaler = MinMaxScaler()
X_time_scaled = time_scaler.fit_transform(np.array(X_time).reshape(-1, 1))
joblib.dump(time_scaler, 'time_scaler.pkl')
print("스케일러 저장 완료: time_scaler.pkl")


# 5) 텐서 변환
X_user = torch.LongTensor(X_user)
X_quiz = torch.LongTensor(X_quiz)
X_time_tensor = torch.FloatTensor(X_time_scaled)
y = torch.FloatTensor(y)


# 6) 모델 정의 (기존과 동일)
class QuizRecModel(nn.Module):
    def __init__(self, n_users, n_quizzes, emb_dim=16):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.quiz_emb = nn.Embedding(n_quizzes, emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, user, quiz, time):
        u = self.user_emb(user)
        q = self.quiz_emb(quiz)
        x = torch.cat([u, q, time], dim=1)
        return self.fc(x).squeeze(1)


# 7) 학습
model = QuizRecModel(len(users), len(quizzes))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

print("\n모델 학습을 시작합니다...")
for epoch in range(1001):
    model.train()
    optimizer.zero_grad()
    pred = model(X_user, X_quiz, X_time_tensor)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, loss: {loss.item():.4f}")

# 8) 모델 저장
torch.save(model.state_dict(), "quizrec_with_time_db.pt")
print("모델 저장 완료: quizrec_with_time_db.pt")
