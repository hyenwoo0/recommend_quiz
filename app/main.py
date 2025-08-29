# -*- coding: utf-8 -*-
import torch
import joblib
import uvicorn
from fastapi import FastAPI, Query, HTTPException

# --- 모듈 임포트 ---
from database import load_data_from_db
from model import QuizRecModel
from recommend import get_recommendations

# --- 전역 변수 및 설정 ---
# 연산 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"연산 장치로 '{device}'를 사용합니다.")

# 데이터 로드
users, quizzes, history = load_data_from_db()
if not history:
    print("경고: DB에서 데이터를 불러오지 못했습니다. API가 정상적으로 작동하지 않을 수 있습니다.")

# ID to Index 매핑
user_to_idx = {user_id: i for i, user_id in enumerate(users)}
quiz_to_idx = {quiz_id: i for i, quiz_id in enumerate(quizzes)}

# 모델 및 스케일러 로드
model = None
scaler = None
if users and quizzes:
    try:
        model = QuizRecModel(len(users), len(quizzes)).to(device)
        model.load_state_dict(torch.load("quizrec_with_time_db.pt", map_location=device))
        model.eval()
        print("모델('quizrec_with_time_db.pt') 로드에 성공했습니다.")
    except FileNotFoundError:
        print("오류: 모델 파일('quizrec_with_time_db.pt')을 찾을 수 없습니다.")
    except Exception as e:
        print(f"모델 로드 중 오류가 발생했습니다: {e}")

    try:
        scaler = joblib.load('time_scaler.pkl')
        print("스케일러('time_scaler.pkl') 로드에 성공했습니다.")
    except FileNotFoundError:
        print("경고: 스케일러 파일('time_scaler.pkl')을 찾을 수 없습니다. 스케일링 없이 진행됩니다.")
    except Exception as e:
        print(f"스케일러 로드 중 오류가 발생했습니다: {e}")
else:
    print("오류: 사용자 또는 퀴즈 데이터가 없어 모델을 초기화할 수 없습니다.")


# --- FastAPI 앱 정의 ---
app = FastAPI(
    title="퀴즈 추천 API",
    description="사용자에게 맞춤형 퀴즈를 추천하는 시스템입니다.",
    version="1.0.0"
)

@app.get("/recommend", summary="사용자 맞춤 퀴즈 추천")
def recommend_endpoint(
        user_id: str = Query(..., description="추천을 받을 사용자 ID", example="user_1"),
        top_n: int = Query(3, description="추천할 퀴즈 개수", ge=1, le=20)
):
    """
    특정 사용자에게 아직 풀지 않은 퀴즈 중에서 도전적인 문제를 `top_n`개 추천합니다.
    """
    attempted_quizzes = {q_id for u_id, q_id, _, _ in history if u_id == user_id}
    candidate_quizzes = [q for q in quizzes if q not in attempted_quizzes]

    if not candidate_quizzes:
        return {"user_id": user_id, "recommended_quizzes": []}

    user_times = [time for u_id, _, _, time in history if u_id == user_id]
    avg_time = sum(user_times) / len(user_times) if user_times else 30.0

    try:
        recommendations = get_recommendations(
            user_id_str=user_id,
            candidate_quiz_ids=candidate_quizzes,
            avg_time=avg_time,
            top_n=top_n,
            model=model,
            scaler=scaler,
            user_to_idx=user_to_idx,
            quiz_to_idx=quiz_to_idx,
            device=device
        )
        return {"user_id": user_id, "recommended_quizzes": recommendations}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"추천 생성 중 예기치 않은 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="추천을 생성하는 중 서버에 오류가 발생했습니다.")

# --- 서버 실행 ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)