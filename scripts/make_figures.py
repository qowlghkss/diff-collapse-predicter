#!/usr/bin/env python3
import json
import os

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

RESULT_DIR = "results"
FIG_DIR = "figures/publication"
os.makedirs(FIG_DIR, exist_ok=True)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Figure 2: ROC with single AUC definition (roc_auc_score)
y_true = load_json(os.path.join(RESULT_DIR, "labels.json"))
y_score = load_json(os.path.join(RESULT_DIR, "scores.json"))
auc_payload = load_json(os.path.join(RESULT_DIR, "auc.json"))

if y_true is not None and y_score is not None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = float(auc_payload["auc"]) if (auc_payload and "auc" in auc_payload) else roc_auc_score(y_true, y_score)

    plt.figure(figsize=(5.8, 5.0))
    plt.plot(fpr, tpr, lw=2.3, label=f"AUC={auc_val:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="#888888", lw=1.2)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_roc.png"), dpi=220)
    plt.close()

# Figure 3: intervention comparison in fixed order
results = load_json(os.path.join(RESULT_DIR, "intervention.json"))
if results:
    ordered_labels = ["Always", "CI-timing", "Random_budget_matched", "Late"]
    labels = [k for k in ordered_labels if k in results]
    values = [results[k] for k in labels]

    plt.figure(figsize=(7.2, 4.8))
    plt.bar(labels, values, color=["#4CAF50", "#2196F3", "#FF9800", "#E53935"][: len(labels)])
    plt.ylabel("Collapse Rate")
    plt.title("Intervention Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_intervention.png"), dpi=220)
    plt.close()

print(f"Figures saved in {FIG_DIR}")
