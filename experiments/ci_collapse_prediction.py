"""
CI Collapse Prediction Experiment
==================================
Validates whether Collapse Indicator (CI) computed at early timesteps
(t = 0..15) can predict diffusion collapse early.

Data
----
experiments/data/multiview/
  <model>_control_<seed>_ci.npy     — CI trajectory, shape (50,)
  <model>_control_<seed>_thin.npy   — thin_pixel_count trajectory, shape (50,)

There are 12 model variants × 10 seeds = 120 samples in the control condition.

Pipeline
--------
1. Each (model, seed) pair is one independent generation sample.
2. CI feature  : nanmean of CI values at t = 0..15.
3. Collapse label (MinSim proxy):
     final_thin = thin_traj[-1]          (final thin-pixel count)
     MinSim     = final_thin             (higher = more detail preserved)
     collapsed  = 1  if final_thin < bottom-25th-percentile
                  0  otherwise
   Low thin-pixel count at the end ≡ thin structures collapsed / disappeared.
4. Logistic Regression (CI → collapse probability), 5-fold stratified CV.
5. Metrics: ROC-AUC, Accuracy, Precision, Recall.
6. Plots: ROC curve, CI distribution by collapse label.

Output
------
experiments/figures/ci_roc_curve.png
experiments/figures/ci_distribution.png
experiments/metrics/ci_collapse_prediction.json
"""

import os
import re
import glob
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    roc_curve, confusion_matrix, classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ─────────────────────────────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, "data", "multiview")
FIGURES_DIR  = os.path.join(SCRIPT_DIR, "figures")
METRICS_DIR  = os.path.join(SCRIPT_DIR, "metrics")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
EARLY_T_END    = 16        # use t = 0 .. 15 for CI feature
COLLAPSE_PCTILE = 25       # bottom-25th-percentile of final thin-pixel → collapsed
CONDITION      = "control"
N_FOLDS        = 5
RANDOM_STATE   = 42

SEP = "=" * 70

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Load data
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  STEP 1 — Load data from {DATA_DIR}\n{SEP}")

ci_files = sorted(glob.glob(os.path.join(DATA_DIR, f"*_{CONDITION}_*_ci.npy")))
print(f"  Found {len(ci_files)} CI trajectory files for condition='{CONDITION}'")

if len(ci_files) == 0:
    raise FileNotFoundError(
        f"No files matching '*_{CONDITION}_*_ci.npy' in {DATA_DIR}.\n"
        "Run the CI runner to generate data first."
    )

def parse_model_seed(path: str, condition: str):
    """
    Parse <model_prefix> and <seed> from filename like:
        mvdream_baseline_control_42_ci.npy
    Returns (model_prefix, seed) or raises ValueError.
    """
    base = os.path.basename(path)
    m = re.search(rf"^(.+)_{condition}_(\d+)_ci\.npy$", base)
    if not m:
        raise ValueError(f"Cannot parse from: {base}")
    return m.group(1), int(m.group(2))


records = []
for cif in ci_files:
    model_prefix, seed = parse_model_seed(cif, CONDITION)
    thin_path = cif.replace("_ci.npy", "_thin.npy")
    if not os.path.exists(thin_path):
        print(f"  [SKIP] Missing thin file for {model_prefix} seed={seed}")
        continue
    ci_traj   = np.load(cif).astype(np.float64)
    thin_traj = np.load(thin_path).astype(np.float64)
    records.append({
        "model": model_prefix,
        "seed":  seed,
        "ci":    ci_traj,
        "thin":  thin_traj,
    })

records.sort(key=lambda r: (r["model"], r["seed"]))
N = len(records)

models_unique = sorted(set(r["model"] for r in records))
seeds_unique  = sorted(set(r["seed"]  for r in records))
print(f"  Loaded {N} samples")
print(f"  Models ({len(models_unique)}): {models_unique}")
print(f"  Seeds  ({len(seeds_unique)}): {seeds_unique}")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — CI feature from early timesteps  (t = 0 .. EARLY_T_END-1)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  STEP 2 — Compute CI feature (t=0..{EARLY_T_END-1})\n{SEP}")

def ci_feature(ci_traj: np.ndarray, t_end: int = EARLY_T_END) -> float:
    """
    nanmean of valid CI values in t=0..t_end-1.
    CI is NaN for the first ~9 steps (window warm-up), then real values.
    Returns 0.0 if all values are NaN (e.g., very short window).
    """
    early  = ci_traj[:t_end]
    valid  = early[~np.isnan(early)]
    return float(np.mean(valid)) if len(valid) > 0 else 0.0


ci_scores   = np.array([ci_feature(r["ci"]) for r in records])   # (N,)
final_thins = np.array([r["thin"][-1]       for r in records])    # (N,)

print(f"  CI feature   : mean={ci_scores.mean():.4f}, std={ci_scores.std():.4f}")
print(f"  CI feature   : min={ci_scores.min():.4f},  max={ci_scores.max():.4f}")
print(f"  Valid CI entries (non-zero): {np.sum(ci_scores != 0)}/{N}")
print(f"  Final thin-px: mean={final_thins.mean():.0f}, std={final_thins.std():.0f}")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — MinSim collapse label (final thin-pixel count)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  STEP 3 — Compute MinSim collapse label\n{SEP}")

# MinSim interpretation: high final thin-pixel count = diverse / preserved
# structure; low value = collapsed (thin branches disappeared).
# We label samples whose final thin-pixel count is below the bottom-25th
# percentile as collapsed (y=1).
#
# Note: this is equivalent to using final_thin as a proxy for MinSim
# because "min pairwise similarity" in the structural sense maps to
# "how many thin pixels survived to the end".

threshold = np.percentile(final_thins, COLLAPSE_PCTILE)
labels    = (final_thins < threshold).astype(int)   # 1=collapsed, 0=stable

n_collapsed = int(labels.sum())
n_stable    = N - n_collapsed
print(f"  MinSim threshold ({COLLAPSE_PCTILE}th pctile of final thin-px): {threshold:.0f}")
print(f"  Collapsed:  {n_collapsed}  ({n_collapsed/N*100:.1f}%)")
print(f"  Stable:     {n_stable}  ({n_stable/N*100:.1f}%)")

X = ci_scores.reshape(-1, 1)   # (N, 1)
y = labels                      # (N,)

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — Logistic Regression  (5-fold stratified CV)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  STEP 4 — Train Logistic Regression ({N_FOLDS}-fold CV)\n{SEP}")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                   class_weight="balanced")),
])

cv     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

auc       = roc_auc_score(y, y_prob)
accuracy  = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division=0)
recall    = recall_score(y, y_pred, zero_division=0)
cm        = confusion_matrix(y, y_pred)

print(f"\n  ── Evaluation Metrics ──────────────────────────────────────────")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
print(f"              Pred-0  Pred-1")
print(f"  Actual-0    {cm[0,0]:5d}   {cm[0,1]:5d}")
print(f"  Actual-1    {cm[1,0]:5d}   {cm[1,1]:5d}")
print(f"\n{classification_report(y, y_pred, target_names=['Stable','Collapsed'])}")

# Fit once on full data
model.fit(X, y)
coef      = model.named_steps["clf"].coef_[0][0]
intercept = model.named_steps["clf"].intercept_[0]
print(f"  Full-data LR coefficient  (CI → logit): {coef:+.4f}")
print(f"  Full-data LR intercept                : {intercept:+.4f}")

# AUC interpretation
verdict = "STRONG" if auc >= 0.8 else ("MODERATE" if auc >= 0.65 else "WEAK")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — Save metrics JSON
# ─────────────────────────────────────────────────────────────────────────────
metrics_dict = {
    "condition":              CONDITION,
    "n_samples":              N,
    "n_models":               len(models_unique),
    "n_seeds":                len(seeds_unique),
    "early_t_end":            EARLY_T_END,
    "collapse_percentile":    COLLAPSE_PCTILE,
    "minsim_threshold_thin":  float(threshold),
    "n_collapsed":            n_collapsed,
    "n_stable":               n_stable,
    "roc_auc":                float(auc),
    "auc_verdict":            verdict,
    "accuracy":               float(accuracy),
    "precision":              float(precision),
    "recall":                 float(recall),
    "confusion_matrix":       cm.tolist(),
    "lr_coefficient_ci":      float(coef),
    "lr_intercept":           float(intercept),
}
metrics_path = os.path.join(METRICS_DIR, "ci_collapse_prediction.json")
with open(metrics_path, "w") as f:
    json.dump(metrics_dict, f, indent=2)
print(f"\n  Saved metrics → {metrics_path}")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — Visualization
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}\n  STEP 6 — Visualization\n{SEP}")

C_COLLAPSE = "#E05252"
C_STABLE   = "#3B82F6"
C_ROC      = "#7C3AED"
color_v    = "#16A34A" if auc >= 0.8 else ("#D97706" if auc >= 0.65 else "#DC2626")

# ── Figure 1: ROC Curve ───────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y, y_prob)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, color=C_ROC, lw=2.5, label=f"CI predictor  (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], color="gray", lw=1.5, ls="--", label="Random (AUC = 0.50)")
ax.fill_between(fpr, tpr, alpha=0.12, color=C_ROC)

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_title(
    f"ROC Curve — CI Early Prediction of Collapse\n"
    f"(t = 0..{EARLY_T_END-1}, {N} samples, {N_FOLDS}-fold CV)",
    fontsize=13, fontweight="bold",
)
ax.legend(fontsize=11, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.text(0.60, 0.12, f"AUC = {auc:.3f}\n({verdict})",
        transform=ax.transAxes, fontsize=12, fontweight="bold",
        color=color_v, bbox=dict(boxstyle="round,pad=0.4", fc="white",
                                  ec=color_v, alpha=0.85))
fig.tight_layout()
roc_path = os.path.join(FIGURES_DIR, "ci_roc_curve.png")
fig.savefig(roc_path, dpi=150)
plt.close(fig)
print(f"  Saved → {roc_path}")

# ── Figure 2: CI distribution (collapse vs stable) ────────────────────────────
ci_collapse = ci_scores[y == 1]
ci_stable   = ci_scores[y == 0]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: overlapping histograms
ax = axes[0]
all_min = min(ci_scores.min() - 0.01, -0.5)
all_max = max(ci_scores.max() + 0.01,  0.5)
bins = np.linspace(all_min, all_max, 30)
ax.hist(ci_stable,   bins=bins, color=C_STABLE,   alpha=0.65,
        label=f"Stable (n={n_stable})",       edgecolor="white", lw=0.5)
ax.hist(ci_collapse, bins=bins, color=C_COLLAPSE, alpha=0.65,
        label=f"Collapsed (n={n_collapsed})", edgecolor="white", lw=0.5)

if len(ci_stable) > 0:
    ax.axvline(ci_stable.mean(), color=C_STABLE, lw=2, ls="--",
               label=f"Stable μ = {ci_stable.mean():.3f}")
if len(ci_collapse) > 0:
    ax.axvline(ci_collapse.mean(), color=C_COLLAPSE, lw=2, ls="--",
               label=f"Collapse μ = {ci_collapse.mean():.3f}")

ax.set_xlabel(f"Early CI Score  (t = 0..{EARLY_T_END-1})", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("CI Distribution by Collapse Label", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Right: violin + scatter
ax2 = axes[1]
data_vio = [ci_stable, ci_collapse] if len(ci_stable) > 1 and len(ci_collapse) > 1 else None

if data_vio is not None:
    parts = ax2.violinplot(data_vio, positions=[0, 1],
                            showmeans=True, showmedians=True, widths=0.55)
    for pc, color in zip(parts["bodies"], [C_STABLE, C_COLLAPSE]):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    for key in ["cmeans", "cmedians", "cbars", "cmaxes", "cmins"]:
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.5)

ax2.scatter(np.zeros(len(ci_stable)),   ci_stable,   color=C_STABLE,   alpha=0.5,
            s=18, zorder=3, label="Stable")
ax2.scatter(np.ones(len(ci_collapse)),  ci_collapse, color=C_COLLAPSE, alpha=0.5,
            s=18, zorder=3, label="Collapsed")
ax2.set_xticks([0, 1])
ax2.set_xticklabels(["Stable", "Collapsed"], fontsize=12)
ax2.set_ylabel(f"Early CI Score (t = 0..{EARLY_T_END-1})", fontsize=12)
ax2.set_title("CI Score by Group (Violin + Scatter)", fontsize=13, fontweight="bold")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.suptitle(
    f"CI Early-Warning Prediction  |  AUC = {auc:.3f}  |  "
    f"Collapse threshold = {COLLAPSE_PCTILE}th pctile of final thin-px",
    fontsize=12, fontweight="bold", y=1.01,
)
fig.tight_layout()
dist_path = os.path.join(FIGURES_DIR, "ci_distribution.png")
fig.savefig(dist_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {dist_path}")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 — Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  CI COLLAPSE PREDICTION — SUMMARY")
print(f"  Samples          : {N}  ({len(models_unique)} models × {len(seeds_unique)} seeds)")
print(f"  Condition        : {CONDITION}")
print(f"  Collapse label   : final thin_px < {threshold:.0f}  (bottom {COLLAPSE_PCTILE}th pctile)")
print(f"  Collapsed / Total: {n_collapsed} / {N}  ({n_collapsed/N*100:.1f}%)")
print(f"  ── Metrics ─────────────────────────────────────────────────────")
print(f"  ROC-AUC   : {auc:.4f}  ← {verdict} predictive signal")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  ── Outputs ─────────────────────────────────────────────────────")
print(f"  ROC curve    → {roc_path}")
print(f"  CI dist.     → {dist_path}")
print(f"  Metrics JSON → {metrics_path}")
print(f"{SEP}\n")
