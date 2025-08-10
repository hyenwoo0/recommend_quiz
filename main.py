# main.py
import torch
from fastapi import FastAPI, Query, HTTPException
from typing import List
import uvicorn
from sample_data import users, quizzes, history  # history: (log_id, user_id, quiz_id, is_correct, solved_time)

# 사용자 및 퀴즈 ID → 인덱스 매핑 (변수명 변경)
user_idx = {u: i for i, u in enumerate(users)}
quiz_idx = {q: i for i, q in enumerate(quizzes)}
idx_quiz = {i: q for q, i in quiz_idx.items()}

# 모델 정의 (solved_time 입력 반영)
class QuizRecModel(torch.nn.Module):
    def __init__(self, n_users, n_quizzes, emb_dim=16):
        super().__init__()
        self.user_emb = torch.nn.Embedding(n_users, emb_dim)
        self.quiz_emb = torch.nn.Embedding(n_quizzes, emb_dim)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(emb_dim*2 + 1, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, user, quiz, time):
        u = self.user_emb(user)
        q = self.quiz_emb(quiz)
        x = torch.cat([u, q, time], dim=1)
        return self.fc(x).squeeze(1)

# 장치 설정 및 모델 로드
device = torch.device("cpu")
model = QuizRecModel(len(users), len(quizzes)).to(device)
model.load_state_dict(torch.load("quizrec_with_time.pt", map_location=device))
model.eval()

app = FastAPI()

def recommend_quizzes(
    model,
    user_id: str,
    candidate_quiz_ids: List[str],
    solved_times: List[float],
    top_n: int = 3
):
    if user_id not in user_idx:
        raise HTTPException(404, f"Unknown user_id: {user_id}")
    if len(candidate_quiz_ids) != len(solved_times):
        raise HTTPException(400, "quiz_ids와 solved_times의 길이가 달라요")

    ui = torch.LongTensor([user_idx[user_id]] * len(candidate_quiz_ids)).to(device)
    qi = torch.LongTensor([quiz_idx[q] for q in candidate_quiz_ids]).to(device)
    ti = torch.FloatTensor(solved_times).unsqueeze(1).to(device)

    with torch.no_grad():
        prob = model(ui, qi, ti).cpu().numpy()

    wrong_prob = 1 - prob
    ranked = sorted(zip(wrong_prob, candidate_quiz_ids), reverse=True)
    return [qid for _, qid in ranked[:top_n]]

@app.get("/recommend_user")
def recommend_user(
    user_id: str = Query(..., description="추천할 사용자 ID"),
    top_n: int = Query(3, description="추천할 퀴즈 개수")
):
    attempted = {q for _, u, q, _, _ in history if u == user_id}
    candidates = [q for q in quizzes if q not in attempted]
    if not candidates:
        return {"user_id": user_id, "recommended_quizzes": []}

    times = [t for _, u, _, _, t in history if u == user_id]
    avg_time = sum(times) / len(times) if times else 30.0
    solved_times = [avg_time] * len(candidates)

    recs = recommend_quizzes(model, user_id, candidates, solved_times, top_n=top_n)
    return {"user_id": user_id, "recommended_quizzes": recs}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
