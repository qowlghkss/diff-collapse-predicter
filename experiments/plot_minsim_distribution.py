#!/usr/bin/env python3
"""
Plot MinSim Distribution: Control vs Intervention
==================================================
Overlaid KDE + histogram showing the shift in MinSim
between control and intervention conditions.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data", "multiview")
OUT_DIR    = os.path.join(SCRIPT_DIR, "figures")
CSV_PATH   = os.path.join(DATA_DIR, "master_results.csv")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    ctrl_ms, intv_ms = [], []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            ms = float(row["MinSim"])
            if row["method"] == "control":
                ctrl_ms.append(ms)
            elif row["method"] == "intervention":
                intv_ms.append(ms)

    ctrl_ms = np.array(ctrl_ms)
    intv_ms = np.array(intv_ms)

    print(f"Control     n={len(ctrl_ms):3d}  mean={ctrl_ms.mean():.4f}  std={ctrl_ms.std():.4f}")
    print(f"Intervention n={len(intv_ms):3d}  mean={intv_ms.mean():.4f}  std={intv_ms.std():.4f}")

    # ── Style ────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # ── Color palette ────────────────────────────────────────────────────
    ctrl_color = "#E53935"   # red
    intv_color = "#1E88E5"   # blue
    ctrl_fill  = "#E5393540"
    intv_fill  = "#1E88E540"

    # ── KDE ──────────────────────────────────────────────────────────────
    x_min = min(ctrl_ms.min(), intv_ms.min()) - 0.05
    x_max = max(ctrl_ms.max(), intv_ms.max()) + 0.05
    x_grid = np.linspace(x_min, x_max, 300)

    kde_ctrl = gaussian_kde(ctrl_ms, bw_method=0.25)
    kde_intv = gaussian_kde(intv_ms, bw_method=0.25)

    # ── Figure ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Histograms (translucent)
    bins = np.linspace(x_min, x_max, 20)
    ax.hist(ctrl_ms, bins=bins, density=True, alpha=0.25, color=ctrl_color,
            edgecolor="none", label=None)
    ax.hist(intv_ms, bins=bins, density=True, alpha=0.25, color=intv_color,
            edgecolor="none", label=None)

    # KDE curves
    ax.plot(x_grid, kde_ctrl(x_grid), color=ctrl_color, lw=2.5,
            label=f"Control  (μ={ctrl_ms.mean():.3f})")
    ax.fill_between(x_grid, kde_ctrl(x_grid), alpha=0.12, color=ctrl_color)

    ax.plot(x_grid, kde_intv(x_grid), color=intv_color, lw=2.5,
            label=f"Intervention  (μ={intv_ms.mean():.3f})")
    ax.fill_between(x_grid, kde_intv(x_grid), alpha=0.12, color=intv_color)

    # Rug plot (individual data points)
    rug_y = -0.08 * ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else -0.1
    # re-evaluate after first draw
    ax.plot(ctrl_ms, np.full_like(ctrl_ms, -0.05), "|", color=ctrl_color,
            ms=12, mew=1.2, alpha=0.7)
    ax.plot(intv_ms, np.full_like(intv_ms, -0.15), "|", color=intv_color,
            ms=12, mew=1.2, alpha=0.7)

    # Mean lines
    ax.axvline(ctrl_ms.mean(), color=ctrl_color, ls="--", lw=1.5, alpha=0.7)
    ax.axvline(intv_ms.mean(), color=intv_color, ls="--", lw=1.5, alpha=0.7)

    # Labels
    ax.set_xlabel("MinSim (Minimum Pairwise DINOv2 Cosine Similarity)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("Distribution of MinSim: Control vs Intervention",
                 fontsize=15, fontweight="bold", pad=12)
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True,
              loc="upper left")

    # Clean up
    ax.set_ylim(bottom=-0.25)
    ax.set_xlim(x_min, x_max)
    ax.tick_params(axis="both", labelsize=11)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "minsim_distribution.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
