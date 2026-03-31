#!/usr/bin/env python3
"""
Select Representative Paired Samples for Visualization
=======================================================
1. Find samples where control MinSim is in the bottom 10%
   AND intervention MinSim improves significantly.
2. Select top 3~5 seeds ranked by improvement (delta).
3. Output: list of seed IDs + side-by-side comparison images
   (control vs intervention).
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data", "multiview")
FIG_DIR    = os.path.join(SCRIPT_DIR, "figures", "multiview")
OUT_DIR    = os.path.join(SCRIPT_DIR, "figures", "representative_pairs")

CSV_PATH   = os.path.join(DATA_DIR, "master_results.csv")

# ── Settings ─────────────────────────────────────────────────────────────────
BOTTOM_PERCENTILE = 10   # control MinSim bottom 10%
TOP_N             = 5    # select top N seeds

# ── Model → prompt_key mapping (for image file lookup) ──────────────────────
# master_results.csv uses (seed, setting, method) but image files use
# {model}_{prompt_key}_{method}_{seed}.png
# The build_master_results.py maps:
#   main       → main
#   explicit   → mvdream_explicit
#   stress     → stress_test
# Model used in images: mvdream (primary), sd15 (secondary)
SETTING_TO_PROMPT = {
    "main":     "main",
    "explicit": "mvdream_explicit",
    "stress":   "stress_test",
}
# Only mvdream has real images (SDXL are 3129-byte placeholders)
MODELS = ["mvdream", "sd15"]

SEP = "=" * 70


def find_image(setting, method, seed):
    """Find first valid (non-placeholder) image for this sample."""
    prompt_key = SETTING_TO_PROMPT.get(setting, setting)
    for model in MODELS:
        fname = f"{model}_{prompt_key}_{method}_{seed}.png"
        fpath = os.path.join(FIG_DIR, fname)
        if os.path.exists(fpath):
            sz = os.path.getsize(fpath)
            if sz > 5000:  # skip SDXL 3129-byte placeholders
                return fpath
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Step 1: Load CSV and pair control/intervention ───────────────────
    ctrl = {}   # (seed, setting) → MinSim
    intv = {}

    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            key = (int(row["seed"]), row["setting"])
            ms = float(row["MinSim"])
            if row["method"] == "control":
                ctrl[key] = ms
            elif row["method"] == "intervention":
                intv[key] = ms

    # Build paired list
    paired = []
    for key in sorted(ctrl):
        if key not in intv:
            continue
        paired.append({
            "seed":       key[0],
            "setting":    key[1],
            "ctrl_ms":    ctrl[key],
            "intv_ms":    intv[key],
            "delta":      intv[key] - ctrl[key],
        })

    print(f"\n{SEP}")
    print("  STEP 1 — All paired samples")
    print(f"{SEP}\n")
    print(f"  {'seed':>4s}  {'setting':>10s}  {'ctrl_MinSim':>12s}  {'intv_MinSim':>12s}  {'delta':>8s}")
    print("  " + "-" * 52)
    for p in paired:
        print(f"  {p['seed']:4d}  {p['setting']:>10s}  {p['ctrl_ms']:12.4f}  {p['intv_ms']:12.4f}  {p['delta']:+8.4f}")

    # ── Step 2: Bottom 10% control MinSim ────────────────────────────────
    all_ctrl_ms = [p["ctrl_ms"] for p in paired]
    threshold = np.percentile(all_ctrl_ms, BOTTOM_PERCENTILE)

    print(f"\n{SEP}")
    print(f"  STEP 2 — Bottom {BOTTOM_PERCENTILE}% control MinSim threshold = {threshold:.4f}")
    print(f"{SEP}\n")

    bottom = [p for p in paired if p["ctrl_ms"] <= threshold]
    # Also include samples very close to the threshold if we have too few
    if len(bottom) < 3:
        threshold_relaxed = np.percentile(all_ctrl_ms, 20)
        bottom = [p for p in paired if p["ctrl_ms"] <= threshold_relaxed]
        print(f"  Relaxed threshold to 20th percentile = {threshold_relaxed:.4f}")

    # Among bottom, sort by delta (biggest improvement first)
    bottom.sort(key=lambda x: x["delta"], reverse=True)

    print(f"  Candidates (ctrl_MinSim <= {threshold:.4f}, sorted by improvement):")
    for p in bottom:
        print(f"    seed={p['seed']:2d}  setting={p['setting']:>10s}  "
              f"ctrl={p['ctrl_ms']:.4f}  intv={p['intv_ms']:.4f}  Δ={p['delta']:+.4f}")

    # ── Step 3: Select top N ─────────────────────────────────────────────
    selected = bottom[:TOP_N]

    print(f"\n{SEP}")
    print(f"  STEP 3 — Selected Top {len(selected)} Representative Seeds")
    print(f"{SEP}\n")

    for i, s in enumerate(selected, 1):
        print(f"  [{i}] seed={s['seed']}, setting={s['setting']}")
        print(f"      Control  MinSim = {s['ctrl_ms']:.4f}")
        print(f"      Interv.  MinSim = {s['intv_ms']:.4f}")
        print(f"      Improvement (Δ) = {s['delta']:+.4f}")
        print()

    # ── Step 4: Generate side-by-side comparison images ──────────────────
    print(f"{SEP}")
    print("  STEP 4 — Generating comparison images")
    print(f"{SEP}\n")

    for s in selected:
        ctrl_img_path = find_image(s["setting"], "control", s["seed"])
        intv_img_path = find_image(s["setting"], "intervention", s["seed"])

        if ctrl_img_path is None or intv_img_path is None:
            print(f"  ✗ Skipping seed={s['seed']} setting={s['setting']} — image not found")
            continue

        ctrl_img = Image.open(ctrl_img_path)
        intv_img = Image.open(intv_img_path)

        fig, axes = plt.subplots(2, 1, figsize=(16, 9))
        fig.suptitle(
            f"Paired Comparison: seed={s['seed']}, setting={s['setting']}\n"
            f"Control MinSim = {s['ctrl_ms']:.4f}  →  Intervention MinSim = {s['intv_ms']:.4f}  "
            f"(Δ = {s['delta']:+.4f})",
            fontsize=14, fontweight="bold"
        )

        # Control row
        axes[0].imshow(ctrl_img)
        axes[0].set_title(
            f"Control (MinSim = {s['ctrl_ms']:.4f})",
            fontsize=13, color="#d32f2f", fontweight="bold"
        )
        axes[0].axis("off")

        # Intervention row
        axes[1].imshow(intv_img)
        axes[1].set_title(
            f"Intervention (MinSim = {s['intv_ms']:.4f})",
            fontsize=13, color="#2e7d32", fontweight="bold"
        )
        axes[1].axis("off")

        plt.tight_layout()
        out_name = f"pair_seed{s['seed']}_{s['setting']}.png"
        out_path = os.path.join(OUT_DIR, out_name)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved: {out_path}")

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SUMMARY — Selected Seed IDs")
    print(f"{SEP}\n")
    print(f"  {'#':>2s}  {'seed':>4s}  {'setting':>10s}  {'ctrl_MinSim':>12s}  {'intv_MinSim':>12s}  {'Δ':>8s}")
    print("  " + "-" * 55)
    for i, s in enumerate(selected, 1):
        print(f"  {i:2d}  {s['seed']:4d}  {s['setting']:>10s}  {s['ctrl_ms']:12.4f}  {s['intv_ms']:12.4f}  {s['delta']:+8.4f}")

    print(f"\n  Output directory: {OUT_DIR}")
    print(f"  Total comparison images generated: {len(selected)}")
    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
