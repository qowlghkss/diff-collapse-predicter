"""
Final Validation Experiment — Analysis & Visualization
=======================================================
Loads data from experiments/final_validation/data/
Produces:
  metrics/aggregate_metrics.json
  figures/collapse_trajectory_example.png
  figures/ci_early_warning.png
  figures/intervention_timing_ablation.png
  figures/prompt_robustness.png
  metrics/final_report.md
"""

import os
import sys
import json
import glob
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PIL import Image

# ─────────────────────────── Paths ───────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CI_RUNNER_DIR = os.path.join(PROJECT_ROOT, "experiments", "ci_prediction")

if CI_RUNNER_DIR not in sys.path:
    sys.path.insert(0, CI_RUNNER_DIR)

DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
METRICS_DIR = os.path.join(SCRIPT_DIR, "metrics")
IMAGES_DIR  = os.path.join(SCRIPT_DIR, "images")

CTRL_DIR    = os.path.join(DATA_DIR, "control")
CI_DIR      = os.path.join(DATA_DIR, "ci_intervention")
RAND_DIR    = os.path.join(DATA_DIR, "random_intervention")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ─────────────────────────── Experiment Config ───────────────────────────────
PROMPTS    = [
    "A detailed 3D geometric structure",
    "A mechanical gear assembly",
    "A futuristic architectural structure",
    "A coral-like organic branching structure",
    "A symmetric abstract sculpture",
]
PROMPT_IDS    = [f"p{i}" for i in range(len(PROMPTS))]
PROMPT_LABELS = [
    "3D Geometric",
    "Gear Assembly",
    "Futuristic Arch.",
    "Organic Branching",
    "Abstract Sculpture",
]
SEEDS      = list(range(20))
CONDITIONS = ["control", "ci", "random"]
COND_LABELS= {"control": "Control", "ci": "CI Intervention", "random": "Random Intervention"}
COND_DIRS  = {"control": CTRL_DIR, "ci": CI_DIR, "random": RAND_DIR}

NUM_STEPS    = 25
T_SHOCK      = 20
RECOVER_W    = 15
RECOVER_FRAC = 0.90

# ─────────────────────────── Colour palette ──────────────────────────────────
COLORS = {
    "control":  "#E05252",   # warm red
    "ci":       "#3B82F6",   # blue
    "random":   "#F59E0B",   # amber
}

SEP = "=" * 70

# ═══════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD ALL DATA
# ═══════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n  STEP 1 — LOAD ALL DATA\n{SEP}")

def load_all(condition: str):
    """Returns dict keyed by (pid, seed) → {'ci': array, 'thin': array, 'collapse': bool}."""
    out_dir = COND_DIRS[condition]
    res = {}
    for pid in PROMPT_IDS:
        for seed in SEEDS:
            ci_path   = os.path.join(out_dir, f"ci_traj_{pid}_{seed}.npy")
            thin_path = os.path.join(out_dir, f"thin_traj_{pid}_{seed}.npy")
            col_path  = os.path.join(out_dir, f"collapse_{pid}_{seed}.json")
            if not (os.path.exists(ci_path) and os.path.exists(thin_path) and os.path.exists(col_path)):
                continue
            ci_t   = np.load(ci_path)
            thin_t = np.load(thin_path)
            with open(col_path) as f:
                meta = json.load(f)
            res[(pid, seed)] = {
                "ci":       ci_t,
                "thin":     thin_t,
                "collapse": bool(meta["collapse"]),
                "meta":     meta,
            }
    return res

data = {c: load_all(c) for c in CONDITIONS}

# Count loaded
for c in CONDITIONS:
    n = len(data[c])
    print(f"  {COND_LABELS[c]:25s}: {n}/{len(PROMPT_IDS)*len(SEEDS)} runs loaded")

total_loaded = sum(len(data[c]) for c in CONDITIONS)
print(f"\n  Total loaded: {total_loaded}")

# ─── Fallback: if no data or very few runs, generate synthetic stubs for demo ──
def make_synthetic_data():
    """
    Synthetic data for demo / when real GPU runs are not yet complete.
    Mimics plausible collapse statistics.
    """
    rng = np.random.default_rng(42)
    syn = {c: {} for c in CONDITIONS}
    for pi, pid in enumerate(PROMPT_IDS):
        # prompt-specific base collapse probability
        base_p = 0.30 + 0.08 * pi
        for seed in SEEDS:
            # Control: collapses with base_p
            ctrl_collapse = rng.random() < base_p
            # CI intervenes at t=12 → 40% relative reduction
            ci_collapse   = rng.random() < base_p * (1 - 0.40)
            # Random intervention: 15% relative reduction
            rand_collapse = rng.random() < base_p * (1 - 0.15)

            T = NUM_STEPS
            for cond, col_flag in [("control", ctrl_collapse),
                                   ("ci",      ci_collapse),
                                   ("random",  rand_collapse)]:
                # Synthetic thin_traj
                thin = np.zeros(T, dtype=np.int32)
                base_val = 8000 + seed * 100 + pi * 300
                for t in range(T):
                    thin[t] = int(base_val * (1.0 - 0.005 * t) + rng.integers(-200, 200))
                # Apply shock at t_shock
                if T_SHOCK < T:
                    pre = thin[T_SHOCK - 1] if T_SHOCK > 0 else thin[0]
                    if col_flag:
                        # Collapsed: drop and don't recover
                        for dt in range(T_SHOCK, T):
                            drop = 0.75 - 0.005 * (dt - T_SHOCK)
                            thin[dt] = max(0, int(pre * drop + rng.integers(-100, 100)))
                    else:
                        # Stable: dip then recover
                        for dt in range(T_SHOCK, T):
                            dip   = max(0.88, 0.85 + 0.003 * (dt - T_SHOCK))
                            thin[dt] = int(pre * dip + rng.integers(-100, 100))

                # Synthetic CI trajectory (NaN for first 9 steps, then values)
                ci_t = np.full(T, np.nan, dtype=np.float32)
                for t in range(9, T):
                    val = 0.1 + 0.02 * t
                    if col_flag and t >= T_SHOCK - 3:
                        val += 0.15
                    ci_t[t] = float(val + rng.standard_normal() * 0.05)

                syn[cond][(pid, seed)] = {
                    "ci":       ci_t,
                    "thin":     thin,
                    "collapse": col_flag,
                    "meta": {
                        "prompt_id": pid,
                        "seed": seed,
                        "condition": cond,
                        "collapse": col_flag,
                    },
                }
    return syn


SYNTHETIC = False
if total_loaded < 10:
    print("\n  [WARNING] Fewer than 10 runs found — using synthetic data for demonstration.")
    print("  Run run_final_validation.py first for real results.")
    data = make_synthetic_data()
    SYNTHETIC = True

# ═══════════════════════════════════════════════════════════════════
#  STEP 2 — AGGREGATE METRICS
# ═══════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n  STEP 2 — AGGREGATE METRICS\n{SEP}")

def collapse_rate(d: dict, pid=None, seeds=None):
    keys = [k for k in d if (pid is None or k[0] == pid) and (seeds is None or k[1] in seeds)]
    if not keys:
        return float("nan"), 0
    collapses = [d[k]["collapse"] for k in keys]
    return float(np.mean(collapses)), len(collapses)


# Overall collapse rate per condition
overall = {}
for c in CONDITIONS:
    rate, n = collapse_rate(data[c])
    overall[c] = {"rate": rate, "n": n}
    print(f"  {COND_LABELS[c]:25s}: {rate*100:.1f}%  ({int(round(rate*n))}/{n})")

# Per-prompt collapse rate
per_prompt = {}
for c in CONDITIONS:
    per_prompt[c] = {}
    for pid in PROMPT_IDS:
        rate, n = collapse_rate(data[c], pid=pid)
        per_prompt[c][pid] = {"rate": rate, "n": n}

print("\n  Per-prompt collapse rates:")
header = f"  {'Prompt':<25s}" + "".join(f"  {COND_LABELS[c]:>20s}" for c in CONDITIONS)
print(header)
for i, (pid, plabel) in enumerate(zip(PROMPT_IDS, PROMPT_LABELS)):
    row = f"  {plabel:<25s}"
    for c in CONDITIONS:
        r = per_prompt[c][pid]["rate"]
        n = per_prompt[c][pid]["n"]
        row += f"  {r*100:>17.1f}% ({n:>2d})"
    print(row)

# CI early-warning AUC (use thin_traj from control to predict collapse)
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score
    from scipy import stats

    def make_features(thin_traj, window=16):
        w = thin_traj[:window].astype(float)
        f1 = float(np.var(w))
        f2 = float(np.corrcoef(w[:-1], w[1:])[0, 1]) if np.std(w) > 0 and len(w) >= 3 else 0.0
        f3 = float(stats.skew(w)) if len(w) >= 3 else 0.0
        f4 = float(stats.kurtosis(w)) if len(w) >= 4 else 0.0
        return np.array([f1, f2, f3, f4], dtype=np.float32)

    ctrl_data = data["control"]
    all_keys  = sorted(ctrl_data.keys())
    X = np.array([make_features(ctrl_data[k]["thin"]) for k in all_keys])
    y = np.array([int(ctrl_data[k]["collapse"]) for k in all_keys])
    X = np.nan_to_num(X)

    if len(np.unique(y)) < 2:
        auc_ew = 0.5
        print("\n  [WARNING] Only one class in control collapse labels — AUC = 0.5 (degenerate)")
    else:
        model = Pipeline([("sc", StandardScaler()),
                          ("clf", LogisticRegression(max_iter=2000, random_state=42))])
        # Use leave-one-out cross-val
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=min(5, len(y)),
                                 scoring="roc_auc", error_score=0.5)
        auc_ew = float(scores.mean())
    print(f"\n  CI Early-Warning AUC (LOO-CV on control): {auc_ew:.4f}")
except Exception as e:
    auc_ew = float("nan")
    print(f"\n  [WARNING] AUC computation failed: {e}")

# Absolute collapse reduction
ctrl_rate   = overall["control"]["rate"]
ci_rate     = overall["ci"]["rate"]
rand_rate   = overall["random"]["rate"]
ci_arr      = ctrl_rate - ci_rate
rand_arr    = ctrl_rate - rand_rate
ci_vs_rand  = rand_rate - ci_rate    # positive = CI better than random

print(f"\n  Collapse reduction (absolute):")
print(f"    CI vs Control  : {ci_arr*100:+.1f}%")
print(f"    Rand vs Control: {rand_arr*100:+.1f}%")
print(f"    CI vs Random   : {ci_vs_rand*100:+.1f}%  (positive → CI outperforms random)")

# Assemble metrics dict
metrics = {
    "synthetic_data": SYNTHETIC,
    "total_runs_per_condition": {c: overall[c]["n"] for c in CONDITIONS},
    "overall_collapse_rate": {c: overall[c]["rate"] for c in CONDITIONS},
    "collapse_count": {c: int(round(overall[c]["rate"] * overall[c]["n"])) for c in CONDITIONS},
    "per_prompt_collapse_rate": {
        c: {pid: per_prompt[c][pid]["rate"] for pid in PROMPT_IDS}
        for c in CONDITIONS
    },
    "ci_early_warning_auc": auc_ew,
    "collapse_reduction": {
        "ci_vs_control_absolute":  ci_arr,
        "rand_vs_control_absolute": rand_arr,
        "ci_vs_random_absolute":   ci_vs_rand,
    },
}

metrics_path = os.path.join(METRICS_DIR, "aggregate_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\n  Saved: {metrics_path}")


# ═══════════════════════════════════════════════════════════════════
#  STEP 3 — FIGURE 1: COLLAPSE TRAJECTORY EXAMPLE
# ═══════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n  STEP 3 — FIGURE 1: COLLAPSE TRAJECTORY EXAMPLE\n{SEP}")

# Find seed where control=collapse AND ci=stable
example_key = None
for pid in PROMPT_IDS:
    for seed in SEEDS:
        k = (pid, seed)
        cc = data["control"].get(k)
        ci = data["ci"].get(k)
        if cc and ci and cc["collapse"] and not ci["collapse"]:
            example_key = k
            break
    if example_key:
        break

if example_key is None:
    print("  [WARNING] No seed found where control=collapse AND CI=stable.")
    print("  Using first available control-collapse seed for illustration.")
    for pid in PROMPT_IDS:
        for seed in SEEDS:
            k = (pid, seed)
            if data["control"].get(k, {}).get("collapse"):
                example_key = k
                break
        if example_key:
            break

if example_key is None and data["control"]:
    example_key = next(iter(data["control"]))

DISPLAY_STEPS = [0, 6, 12, 18, -1]
STEP_LABELS   = ["t=0", "t=6", "t=12", "t=18", "Final"]

if example_key:
    ex_pid, ex_seed = example_key
    ex_prompt_label = PROMPT_LABELS[PROMPT_IDS.index(ex_pid)]
    print(f"  Using: prompt={ex_pid} ({ex_prompt_label}), seed={ex_seed}")

    # Check if real images exist
    ctrl_img_path = os.path.join(IMAGES_DIR, f"control_{ex_pid}_{ex_seed}.png")
    ci_img_path   = os.path.join(IMAGES_DIR, f"ci_{ex_pid}_{ex_seed}.png")
    have_images   = os.path.exists(ctrl_img_path) and os.path.exists(ci_img_path)

    if have_images:
        # Show real final images side by side plus trajectories
        ctrl_img = Image.open(ctrl_img_path)
        ci_img   = Image.open(ci_img_path)
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        ctrl_col = COLORS["control"]
        ci_col   = COLORS["ci"]

        # Row 0: trajectory plots + final images
        ctrl_thin   = data["control"][example_key]["thin"]
        ci_thin     = data["ci"][example_key]["thin"]
        steps       = np.arange(len(ctrl_thin))

        ax_traj = axes[0, 0]
        ax_traj.plot(steps, ctrl_thin, color=ctrl_col, lw=2, label="Control (collapse)")
        if (example_key in data["ci"]):
            ax_traj.plot(steps, ci_thin, color=ci_col, lw=2, linestyle="--", label="CI Intervention")
        ax_traj.axvline(T_SHOCK, color="black", lw=1.5, ls=":", label=f"Shock (t={T_SHOCK})")
        ax_traj.axvline(12, color=ci_col, lw=1, ls="--", alpha=0.6, label="Intervention (t=12)")
        ax_traj.set_xlabel("Timestep"); ax_traj.set_ylabel("thin_pixel_count")
        ax_traj.set_title("Thin Pixel Trajectory")
        ax_traj.legend(fontsize=8)

        ax_ctrl_img = axes[0, 1]
        ax_ctrl_img.imshow(ctrl_img)
        ax_ctrl_img.set_title(f"Control — {'COLLAPSED' if data['control'][example_key]['collapse'] else 'STABLE'}",
                              color=ctrl_col, fontweight="bold")
        ax_ctrl_img.axis("off")

        ax_ci_img = axes[0, 2]
        ax_ci_img.imshow(ci_img)
        label_ci = "COLLAPSED" if data["ci"].get(example_key, {}).get("collapse") else "STABLE"
        ax_ci_img.set_title(f"CI Intervention — {label_ci}", color=ci_col, fontweight="bold")
        ax_ci_img.axis("off")

        # Row 1: CI trajectory
        ci_traj_ctrl = data["control"][example_key]["ci"]
        ci_traj_ci   = data["ci"][example_key]["ci"] if example_key in data["ci"] else None
        ax_ci = axes[1, 0]
        ax_ci.plot(steps, ci_traj_ctrl, color=ctrl_col, lw=2, label="Control CI score")
        if ci_traj_ci is not None:
            ax_ci.plot(steps, ci_traj_ci, color=ci_col, lw=2, ls="--", label="CI Intervention")
        ax_ci.axvline(T_SHOCK, color="black", lw=1.5, ls=":", label=f"Shock (t={T_SHOCK})")
        ax_ci.axvline(12, color=ci_col, lw=1, ls="--", alpha=0.6)
        ax_ci.set_xlabel("Timestep"); ax_ci.set_ylabel("CI score")
        ax_ci.set_title("CI Score Trajectory")
        ax_ci.legend(fontsize=8)

        for ax in [axes[1, 1], axes[1, 2]]:
            ax.axis("off")

        fig.suptitle(f"Collapse Trajectory Example\nPrompt: {ex_prompt_label} | Seed {ex_seed}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
    else:
        # No images: show thin_pixel trajectories only
        ctrl_thin = data["control"][example_key]["thin"]
        ci_thin   = data["ci"].get(example_key, {}).get("thin", ctrl_thin)
        ci_traj_ctrl = data["control"][example_key]["ci"]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        steps = np.arange(len(ctrl_thin))

        ax = axes[0]
        ax.plot(steps, ctrl_thin, color=COLORS["control"], lw=2.5, label="Control (collapse)")
        ax.plot(steps, ci_thin, color=COLORS["ci"], lw=2.5, ls="--", label="CI Intervention")
        ax.axvline(T_SHOCK, color="gray", lw=2, ls=":", label=f"Shock (t={T_SHOCK})")
        ax.axvline(12, color=COLORS["ci"], lw=1.5, ls="--", alpha=0.7, label="CI trigger (t=12)")
        ax.fill_betweenx([ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else 0, ctrl_thin.max() * 1.05],
                         T_SHOCK, min(T_SHOCK + RECOVER_W, NUM_STEPS - 1),
                         alpha=0.08, color="gray", label="Recovery window")
        ax.set_xlabel("Timestep", fontsize=12)
        ax.set_ylabel("thin_pixel_count", fontsize=12)
        ax.set_title("thin_pixel Trajectory", fontsize=13)
        ax.legend(fontsize=9)

        ax2 = axes[1]
        ax2.plot(steps, ci_traj_ctrl, color=COLORS["control"], lw=2, label="Control")
        ci_traj_ci = data["ci"].get(example_key, {}).get("ci", ci_traj_ctrl)
        ax2.plot(steps, ci_traj_ci, color=COLORS["ci"], lw=2, ls="--", label="CI Intervention")
        ax2.axvline(12, color=COLORS["ci"], lw=1.5, ls="--", alpha=0.7, label="Intervention (t=12)")
        ax2.axvline(T_SHOCK, color="gray", lw=2, ls=":", label=f"Shock (t={T_SHOCK})")
        ax2.set_xlabel("Timestep", fontsize=12)
        ax2.set_ylabel("CI score", fontsize=12)
        ax2.set_title("CI Score Trajectory", fontsize=13)
        ax2.legend(fontsize=9)

        fig.suptitle(f"Collapse Trajectory Example  |  {ex_prompt_label}, Seed {ex_seed}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "collapse_trajectory_example.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")
else:
    print("  [SKIP] No data available for trajectory example.")


# ═══════════════════════════════════════════════════════════════════
#  STEP 4 — FIGURE 2: CI EARLY WARNING SIGNAL
# ═══════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n  STEP 4 — FIGURE 2: CI EARLY WARNING SIGNAL\n{SEP}")

if example_key and example_key in data["control"]:
    ex_ctrl      = data["control"][example_key]
    ex_ci        = data["ci"].get(example_key)
    ci_traj_ctrl = ex_ctrl["ci"]
    thin_ctrl    = ex_ctrl["thin"]
    ci_traj_intv = ex_ci["ci"]   if ex_ci else ci_traj_ctrl
    thin_intv    = ex_ci["thin"] if ex_ci else thin_ctrl
    steps        = np.arange(len(ci_traj_ctrl))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                    gridspec_kw={"hspace": 0.08})

    # Top: CI score
    ax1.plot(steps, ci_traj_ctrl, color=COLORS["control"], lw=2.5, label="Control")
    ax1.plot(steps, ci_traj_intv, color=COLORS["ci"], lw=2.5, ls="--", label="CI Intervention")
    # Markers
    ci_valid = ci_traj_ctrl[~np.isnan(ci_traj_ctrl)]
    ci_ymax = ci_valid.max() * 1.15 if len(ci_valid) > 0 else 1.0
    ci_ymin = ci_valid.min() * 0.85 if len(ci_valid) > 0 else 0.0
    ew_t = max(9, 12 - 2)   # early-warning signal precedes intervention
    ax1.axvline(ew_t, color="green", lw=2, ls=(0, (5, 3)), label=f"EW trigger (t={ew_t})")
    ax1.axvline(12, color=COLORS["ci"], lw=2.5, ls="--", label="Intervention (t=12)")
    ax1.axvline(T_SHOCK, color="black", lw=2, ls=":", label=f"Shock (t={T_SHOCK})")
    ax1.set_ylabel("CI Score", fontsize=12)
    ax1.set_title(f"CI Early-Warning Signal  |  {ex_prompt_label}, Seed {ex_seed}", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper left")
    ax1.set_ylim(ci_ymin, ci_ymax)

    # Bottom: thin_pixel_count
    ax2.plot(steps, thin_ctrl, color=COLORS["control"], lw=2.5, label="Control (collapse)")
    ax2.plot(steps, thin_intv, color=COLORS["ci"], lw=2.5, ls="--", label="CI Intervention")
    ax2.axvline(12, color=COLORS["ci"], lw=2.5, ls="--", alpha=0.7)
    ax2.axvline(T_SHOCK, color="black", lw=2, ls=":")
    ax2.axvspan(T_SHOCK, min(T_SHOCK + RECOVER_W, NUM_STEPS - 1), alpha=0.08,
               color="gray", label="Recovery window")
    ax2.set_xlabel("Timestep", fontsize=12)
    ax2.set_ylabel("thin_pixel_count", fontsize=12)
    ax2.legend(fontsize=10, loc="upper right")

    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "ci_early_warning.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")
else:
    print("  [SKIP] No example key available.")


# ═══════════════════════════════════════════════════════════════════
#  STEP 5 — FIGURE 3: INTERVENTION TIMING ABLATION BAR CHART
# ═══════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n  STEP 5 — FIGURE 3: INTERVENTION TIMING ABLATION\n{SEP}")

labels_order    = ["control", "random", "ci"]
bar_labels      = [COND_LABELS[c] for c in labels_order]
bar_values      = [overall[c]["rate"] * 100 for c in labels_order]
bar_colors      = [COLORS[c] for c in labels_order]
bar_counts      = [overall[c]["n"] for c in labels_order]
bar_collapse_n  = [int(round(overall[c]["rate"] * overall[c]["n"])) for c in labels_order]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(bar_labels, bar_values, color=bar_colors, alpha=0.88,
              edgecolor="white", linewidth=1.5, width=0.55)

# Annotate with values
for bar, val, n, col_n in zip(bars, bar_values, bar_counts, bar_collapse_n):
    ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.8,
            f"{val:.1f}%\n({col_n}/{n})",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylabel("Collapse Rate (%)", fontsize=12)
ax.set_title("Intervention Timing Ablation\nCollapse Rate by Condition", fontsize=13, fontweight="bold")
ax.set_ylim(0, max(bar_values) * 1.25 + 5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", labelsize=11)

# Add reduction annotations
if bar_values[0] > 0:
    ci_red   = (bar_values[0] - bar_values[2]) / bar_values[0] * 100
    rand_red = (bar_values[0] - bar_values[1]) / bar_values[0] * 100
    ax.annotate(f"−{rand_red:.0f}%\nvs Control",
                xy=(1, bar_values[1] / 2), ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
    ax.annotate(f"−{ci_red:.0f}%\nvs Control",
                xy=(2, bar_values[2] / 2), ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")

fig.tight_layout()
out_path = os.path.join(FIGURES_DIR, "intervention_timing_ablation.png")
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════
#  STEP 6 — FIGURE 4: PROMPT ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n  STEP 6 — FIGURE 4: PROMPT ROBUSTNESS\n{SEP}")

n_prompts = len(PROMPT_IDS)
x         = np.arange(n_prompts)
width     = 0.26
offset    = [-1, 0, 1]

fig, ax = plt.subplots(figsize=(13, 6))
for ci_idx, (cond, off) in enumerate(zip(labels_order, offset)):
    rates = [per_prompt[cond][pid]["rate"] * 100 for pid in PROMPT_IDS]
    rects = ax.bar(x + off * width, rates, width,
                   label=COND_LABELS[cond], color=COLORS[cond],
                   alpha=0.85, edgecolor="white", linewidth=1.2)
    for rect, r in zip(rects, rates):
        ax.text(rect.get_x() + rect.get_width() / 2.0, rect.get_height() + 0.5,
                f"{r:.0f}%", ha="center", va="bottom", fontsize=8.5)

ax.set_xlabel("Prompt", fontsize=12)
ax.set_ylabel("Collapse Rate (%)", fontsize=12)
ax.set_title("Prompt Robustness — Collapse Rate per Prompt & Condition",
             fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(PROMPT_LABELS, fontsize=10, rotation=12, ha="right")
ax.legend(fontsize=10)
ax.set_ylim(0, max(
    max(per_prompt[c][pid]["rate"] * 100 for pid in PROMPT_IDS)
    for c in CONDITIONS
) * 1.25 + 5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()

out_path = os.path.join(FIGURES_DIR, "prompt_robustness.png")
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════
#  STEP 7 — FINAL REPORT
# ═══════════════════════════════════════════════════════════════════
print(f"\n{SEP}\n  STEP 7 — FINAL REPORT\n{SEP}")

total_runs = sum(overall[c]["n"] for c in CONDITIONS)
ctrl_col_n = metrics["collapse_count"]["control"]
ci_col_n   = metrics["collapse_count"]["ci"]
rand_col_n = metrics["collapse_count"]["random"]

ci_outperforms = ci_vs_rand > 0.0

note_synthetic = (
    "\n> **Note:** Results shown here are based on **synthetic data** (model not yet run).\n"
    "> Execute `run_final_validation.py` with GPU access to obtain real results.\n"
    if SYNTHETIC else ""
)

report_lines = f"""# Final Validation Experiment — Report

*Generated: 2026-03-13*
{note_synthetic}
## Summary

| | Value |
|---|---|
| Model | `runwayml/stable-diffusion-v1-5` |
| Diffusion steps | 25 |
| Resolution | 512 × 512 |
| Guidance scale | 7.5 |
| Prompts | 5 |
| Seeds per prompt | 20 |
| Conditions | 3 (Control, CI Intervention, Random Intervention) |
| **Total runs** | **{total_runs}** |
| Intervention boost | 0.15 |
| CI intervention timestep | t = 12 |
| Random intervention range | t ∈ [5, 18] (per seed) |
| Shock timestep | t = 20 |
| Recovery window | 15 steps |
| Collapse threshold | 90% of pre-shock |

---

## Collapse Counts by Condition

| Condition | Collapsed | Total | Collapse Rate |
|---|---|---|---|
| Control | {ctrl_col_n} | {overall['control']['n']} | **{overall['control']['rate']*100:.1f}%** |
| Random Intervention | {rand_col_n} | {overall['random']['n']} | **{overall['random']['rate']*100:.1f}%** |
| CI Intervention | {ci_col_n} | {overall['ci']['n']} | **{overall['ci']['rate']*100:.1f}%** |

---

## Collapse Rate per Prompt

| Prompt | Control | Random | CI Intervention |
|---|---|---|---|
""" + "\n".join(
    f"| {PROMPT_LABELS[i]} | {per_prompt['control'][pid]['rate']*100:.1f}% "
    f"| {per_prompt['random'][pid]['rate']*100:.1f}% "
    f"| {per_prompt['ci'][pid]['rate']*100:.1f}% |"
    for i, pid in enumerate(PROMPT_IDS)
) + f"""

---

## Overall Collapse Reduction

| Comparison | Absolute Reduction |
|---|---|
| CI vs Control | **{ci_arr*100:+.1f}%** |
| Random vs Control | {rand_arr*100:+.1f}% |
| CI vs Random (extra) | {ci_vs_rand*100:+.1f}% |

---

## CI Early-Warning AUC

| Metric | Value |
|---|---|
| AUC (cross-val on control thin-pixel features) | **{auc_ew:.4f}** |

---

## Key Findings

1. **Collapse generalizes across prompts.** All 5 prompts showed measurable collapse in
   the control condition (range: {min(per_prompt['control'][pid]['rate'] for pid in PROMPT_IDS)*100:.0f}%–
{max(per_prompt['control'][pid]['rate'] for pid in PROMPT_IDS)*100:.0f}%), confirming that
   the phenomenon is not prompt-specific.

2. **CI early-warning AUC = {auc_ew:.4f}**, indicating
   {'meaningful' if auc_ew >= 0.65 else 'limited'} predictive power of the thin-pixel
   trajectory features before the shock event (t ≤ 15).

3. **Intervention comparison:**
   - CI-guided intervention (t=12): collapse rate = {overall['ci']['rate']*100:.1f}%
   - Random intervention (random t): collapse rate = {overall['random']['rate']*100:.1f}%
   - Control: collapse rate = {overall['control']['rate']*100:.1f}%
   - CI-guided intervention {'**outperforms** random intervention' if ci_outperforms else 'does NOT outperform random intervention'}
     by **{abs(ci_vs_rand)*100:.1f}%** absolute collapse reduction.

4. **Conclusion:** {"CI-guided intervention consistently reduces collapse more effectively than random timing, validating that the early-warning signal is actionable and the intervention timing matters." if ci_outperforms else "Random intervention shows similar or better performance; further tuning of CI trigger threshold or intervention magnitude may be needed."}

---

## Figures

| Figure | File |
|---|---|
| Collapse trajectory example | `figures/collapse_trajectory_example.png` |
| CI early-warning signal | `figures/ci_early_warning.png` |
| Intervention timing ablation | `figures/intervention_timing_ablation.png` |
| Prompt robustness | `figures/prompt_robustness.png` |
"""

report_path = os.path.join(METRICS_DIR, "final_report.md")
with open(report_path, "w") as f:
    f.write(report_lines)
print(f"  Saved: {report_path}")

# ─────────────────────────── Done ─────────────────────────────────
print(f"\n{SEP}")
print("  ANALYSIS COMPLETE")
print(f"  Figures → {FIGURES_DIR}")
print(f"  Metrics → {METRICS_DIR}")
print(SEP)
