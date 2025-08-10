import random

# 1. 유저 5명
users = [f"user{i}" for i in range(1, 6)]  # user1 ~ user5

# 2. 퀴즈 20개
quizzes = [f"Q{str(i).zfill(2)}" for i in range(1, 21)]  # Q01 ~ Q20

# 3. 히스토리 100개 이상 (랜덤 생성)
history = []
random.seed(42)  # 결과 고정 (재현성)

for _ in range(110):  # 110개 생성
    user = random.choice(users)
    quiz = random.choice(quizzes)
    correct = random.randint(0, 1)
    history.append((user, quiz, correct))

# (옵션) 중복 제거 (user, quiz, correct) 조합이 동일한 것만 제거
# 완전한 중복 제거가 필요하다면 아래 주석 해제
# history = list(set(history))

# 데이터 예시 출력
if __name__ == "__main__":
    print("users =", users)
    print("quizzes =", quizzes)
    print("history sample =", history[:10])
    print("history length =", len(history))
