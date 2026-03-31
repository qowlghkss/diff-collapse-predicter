import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ----------------------
# PATH 설정
# ----------------------
RESULT_DIR = "/home/ji/Desktop/pyvision/results"
FIG_DIR = "/home/ji/Desktop/pyvision/figures/publication"

os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------
# 안전 로드 함수
# ----------------------
def load_json(path):
    if not os.path.exists(path):
        print(f"[WARNING] Missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)

# ----------------------
# 1. CI TRAJECTORY
# ----------------------
ci_values = load_json(os.path.join(RESULT_DIR, "ci_values.json"))

if ci_values:
    plt.figure()
    plt.plot(ci_values)
    
    # collapse timestep 자동 추정 (peak)
    collapse_t = ci_values.index(max(ci_values))
    plt.axvline(collapse_t)

    plt.xlabel("Timestep")
    plt.ylabel("CI")
    plt.title("CI Trajectory")
    plt.savefig(os.path.join(FIG_DIR, "fig1_ci.png"))
    plt.close()

# ----------------------
# 2. ROC CURVE
# ----------------------
y_true = load_json(os.path.join(RESULT_DIR, "labels.json"))
y_score = load_json(os.path.join(RESULT_DIR, "scores.json"))

if y_true and y_score:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(os.path.join(FIG_DIR, "fig2_roc.png"))
    plt.close()

# ----------------------
# 3. INTERVENTION BAR
# ----------------------
results = load_json(os.path.join(RESULT_DIR, "intervention.json"))

if results:
    labels = list(results.keys())
    values = list(results.values())

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Collapse Rate")
    plt.title("Intervention Comparison")
    plt.savefig(os.path.join(FIG_DIR, "fig3_intervention.png"))
    plt.close()

# ----------------------
# 4. MULTI-MODEL (optional)
# ----------------------
multi_model = load_json(os.path.join(RESULT_DIR, "multi_model.json"))

if multi_model:
    labels = list(multi_model.keys())
    values = list(multi_model.values())

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Collapse Rate")
    plt.title("Multi-Model Comparison")
    plt.savefig(os.path.join(FIG_DIR, "fig4_multimodel.png"))
    plt.close()

print(f"✅ Figures saved in {FIG_DIR}")