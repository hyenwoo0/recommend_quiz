# train_model.py
import torch
import torch.nn as nn
from sample_data import users, quizzes, history

user_idx = {u: i for i, u in enumerate(users)}
quiz_idx = {q: i for i, q in enumerate(quizzes)}

# 데이터셋 준비 (solved_time 추가)
X_user, X_quiz, X_time, y = [], [], [], []
for log_id, u, q, c, t in history:
    X_user.append(user_idx[u])
    X_quiz.append(quiz_idx[q])
    X_time.append(t)      # 풀이 시간
    y.append(c)

X_user = torch.LongTensor(X_user)
X_quiz = torch.LongTensor(X_quiz)
X_time = torch.FloatTensor(X_time).unsqueeze(1)  # (N, 1)
y      = torch.FloatTensor(y)

class QuizRecModel(nn.Module):
    def __init__(self, n_users, n_quizzes, emb_dim=16):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.quiz_emb = nn.Embedding(n_quizzes, emb_dim)
        # 추가된 t (1차원) 까지 합쳐서 emb_dim*2 + 1
        self.fc = nn.Sequential(
            nn.Linear(emb_dim*2 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, user, quiz, time):
        u = self.user_emb(user)
        q = self.quiz_emb(quiz)
        # time 을 바로 이어 붙이기
        x = torch.cat([u, q, time], dim=1)
        return self.fc(x).squeeze(1)

model     = QuizRecModel(len(users), len(quizzes))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# 학습 루프
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    pred = model(X_user, X_quiz, X_time)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, loss: {loss.item():.4f}")

torch.save(model.state_dict(), "quizrec_with_time.pt")
print("모델 저장 완료: quizrec_with_time.pt")
