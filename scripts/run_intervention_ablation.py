import os
import json
import random

# =========================
# CONFIG
# =========================

RESULT_PATH = "/home/ji/Desktop/pyvision/results"
LATE_THRESHOLD = 20
TIMESTEPS = 30
os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
# 기존 collapse rate (없으면 기본값)
if os.path.exists(RESULT_PATH):
    with open(RESULT_PATH) as f:
        results = json.load(f)
else:
    results = {}

# =========================
# MOCK SIMULATION FUNCTION
# 👉 여기만 너 코드에 맞게 바꾸면 됨
# =========================

def simulate_run(intervention_policy):
    """
    intervention_policy: function(timestep) -> bool
    return: collapse (True/False)
    """

    collapse = False
    stability = 1.0

    for t in range(TIMESTEPS):

        # intervention 적용
        if intervention_policy(t):
            stability += 0.05  # 안정화 효과

        # 자연 붕괴 진행
        stability -= random.uniform(0.01, 0.08)

        if stability < 0.2:
            collapse = True
            break

    return collapse

# =========================
# POLICIES
# =========================

def always_policy(t):
    return True

def late_policy(t):
    return t >= LATE_THRESHOLD

def random_policy(t):
    return random.random() < 0.3

# 기존 CI 정책은 결과 파일에서 가져온다고 가정
# (없으면 skip)
ci_rate = results.get("CI-based", None)

# =========================
# RUN EXPERIMENTS
# =========================

N = 100  # runs

def evaluate(policy):
    collapses = 0
    for _ in range(N):
        if simulate_run(policy):
            collapses += 1
    return collapses / N

print("Running intervention ablations...")

always_rate = evaluate(always_policy)
late_rate = evaluate(late_policy)
random_rate = evaluate(random_policy)

# =========================
# UPDATE RESULTS
# =========================

results.update({
    "Always": always_rate,
    "Late": late_rate,
    "Random_recheck": random_rate  # 기존 random과 비교용
})

with open(RESULT_PATH, "w") as f:
    json.dump(results, f, indent=2)
import os

RESULT_PATH = "results/intervention.json"

# 🔥 이 줄 추가
os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
print("Updated results:")
print(json.dumps(results, indent=2))