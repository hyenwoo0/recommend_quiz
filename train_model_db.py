import torch
import torch.nn as nn
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 1) 사용자·퀴즈 리스트 정의 (변경 없음)
users = ["user1", "user2", "user3", "user4", "user5"]
quizzes = [
    "Q01", "Q02", "Q03", "Q04", "Q05", "Q06", "Q07", "Q08", "Q09", "Q10",
    "Q11", "Q12", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18", "Q19", "Q20"
]
user_idx = {u: i for i, u in enumerate(users)}
quiz_idx = {q: i for i, q in enumerate(quizzes)}


# 2) DB에서 history 로드 (수정됨)
def load_history_from_db():
    conn = None
    cursor = None
    try:
        # (수정) .env 파일에서 DB 접속 정보를 안전하게 불러옵니다.
        db_host = os.getenv("DB_HOST", "127.0.0.1")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_name = os.getenv("DB_DATABASE")

        # (수정) 필수 정보가 있는지 확인
        if not all([db_user, db_password, db_name]):
            print("오류: .env 파일에 DB 접속 정보(DB_USER, DB_PASSWORD, DB_DATABASE)가 없습니다.")
            return []

        conn = mysql.connector.connect(
            host=db_host,
            port=3306,
            user=db_user,
            password=db_password,
            database=db_name
        )
        cursor = conn.cursor()

        cursor.execute("""
                       SELECT user_id, quiz_id, is_correct, time_taken
                       FROM quiz_log
                       """)
        return cursor.fetchall()
    except Error as e:
        print(f"DB 연결/쿼리 오류: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()


# 3) history 데이터 가져오기
history = load_history_from_db()

if not history:
    print("오류: DB에서 데이터를 불러오지 못했거나 데이터가 없습니다. 스크립트를 종료합니다.")
    exit()

# 4) 데이터셋 준비
X_user, X_quiz, X_time, y = [], [], [], []
for u, q, c, t in history:
    X_user.append(user_idx[u])
    X_quiz.append(quiz_idx[q])
    X_time.append(t)
    y.append(c)

X_user = torch.LongTensor(X_user)
X_quiz = torch.LongTensor(X_quiz)
X_time = torch.FloatTensor(X_time).unsqueeze(1)
y = torch.FloatTensor(y)


# 5) 모델 정의 (변경 없음)
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


# 6) 학습 세팅 (변경 없음)
model = QuizRecModel(len(users), len(quizzes))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# 7) 학습 루프 (변경 없음)
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    pred = model(X_user, X_quiz, X_time)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, loss: {loss.item():.4f}")

# 8) 모델 저장 (변경 없음)
torch.save(model.state_dict(), "quizrec_with_time_db.pt")
print("모델 저장 완료: quizrec_with_time_db.pt")
