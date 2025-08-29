# -*- coding: utf-8 -*-
import torch
import numpy as np
from fastapi import HTTPException
from typing import List, Dict


def get_recommendations(
        user_id_str: str,
        candidate_quiz_ids: List[str],
        avg_time: float,
        top_n: int,
        model: torch.nn.Module,
        scaler,
        user_to_idx: Dict[str, int],
        quiz_to_idx: Dict[str, int],
        device: torch.device
) -> List[str]:
    """
    추천 로직을 수행하는 내부 함수.
    모델을 사용하여 후보 퀴즈들에 대한 정답 확률을 예측하고, '틀릴 확률'이 높은 순으로 정렬하여 반환합니다.
    """
    if not model:
        raise HTTPException(status_code=503, detail="서버 모델을 사용할 수 없습니다. (Not Available)")

    if user_id_str not in user_to_idx:
        raise HTTPException(status_code=404, detail=f"사용자 '{user_id_str}'를 찾을 수 없습니다.")

    if scaler:
        time_scaled_value = scaler.transform(np.array([[avg_time]]))[0, 0]
    else:
        time_scaled_value = avg_time

    user_indices = torch.LongTensor([user_to_idx[user_id_str]] * len(candidate_quiz_ids)).to(device)
    quiz_indices = torch.LongTensor([quiz_to_idx[q] for q in candidate_quiz_ids]).to(device)
    time_values = torch.FloatTensor([time_scaled_value] * len(candidate_quiz_ids)).unsqueeze(1).to(device)

    with torch.no_grad():
        prob_correct = model(user_indices, quiz_indices, time_values).cpu().numpy()

    prob_wrong = 1 - prob_correct

    ranked_quizzes = sorted(zip(prob_wrong, candidate_quiz_ids), reverse=True)

    return [qid for _, qid in ranked_quizzes[:top_n]]