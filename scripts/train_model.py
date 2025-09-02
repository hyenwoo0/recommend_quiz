# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import os

# --- 모듈 임포트 ---
# 중복 정의 대신, 이미 만들어진 모듈을 가져와서 사용합니다.
from core.db import load_data_from_db
from models.quiz_rec_model import QuizRecModel

# 1) DB에서 모든 데이터 로드
users, quizzes, history = load_data_from_db()

if not history:
    print("오류: DB 데이터가 없습니다. 스크립트를 종료합니다.")
    exit()

print(f"총 {len(users)}명의 사용자와 {len(quizzes)}개의 퀴즈 데이터를 로드했습니다.")

# 2) ID → 인덱스 매핑
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

# 4) 시간 데이터 스케일링
time_scaler = MinMaxScaler()
X_time_scaled = time_scaler.fit_transform(np.array(X_time).reshape(-1, 1))

# 5) 텐서 변환
X_user_tensor = torch.LongTensor(X_user)
X_quiz_tensor = torch.LongTensor(X_quiz)
X_time_tensor = torch.FloatTensor(X_time_scaled)
y_tensor = torch.FloatTensor(y)

# 6) 모델 정의 및 학습
model = QuizRecModel(len(users), len(quizzes))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

print("\n모델 학습을 시작합니다...")
for epoch in range(1001):
    model.train()
    optimizer.zero_grad()
    pred = model(X_user_tensor, X_quiz_tensor, X_time_tensor)
    loss = criterion(pred, y_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, loss: {loss.item():.4f}")

# 7) 모델 및 스케일러 저장
SAVE_DIR = os.path.join("models", "trained_models")
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(SAVE_DIR, "quizrec_with_time_db.pt")
SCALER_PATH = os.path.join(SAVE_DIR, "time_scaler.pkl")

torch.save(model.state_dict(), MODEL_PATH)
print(f"모델 저장 완료: {MODEL_PATH}")

joblib.dump(time_scaler, SCALER_PATH)
print(f"스케일러 저장 완료: {SCALER_PATH}")
