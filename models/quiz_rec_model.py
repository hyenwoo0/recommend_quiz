# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class QuizRecModel(nn.Module):
    """
    사용자와 퀴즈의 상호작용(풀이 시간 포함)을 기반으로 정답 확률을 예측하는 딥러닝 모델.
    """

    # __init__ 함수는 반드시 n_users와 n_quizzes를 파라미터로 받아야 합니다.
    def __init__(self, n_users, n_quizzes, emb_dim=16):
        """
        모델의 레이어를 초기화합니다.
        """
        super().__init__()
        # 부품(레이어)의 이름은 user_emb, quiz_emb 여야 합니다.
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.quiz_emb = nn.Embedding(n_quizzes, emb_dim)

        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, user, quiz, time):
        """
        모델의 순전파 로직을 정의합니다.
        """
        u_emb = self.user_emb(user)
        q_emb = self.quiz_emb(quiz)

        x = torch.cat([u_emb, q_emb, time], dim=1)

        return self.fc(x).squeeze(1)
