"""
Build Master Results CSV
========================
Creates a unified dataset for all analysis (tables and figures).

For each sample (seed × setting × method):
  1. Identify: seed, setting (main/explicit/stress), method (control/intervention)
  2. Compute: MinSim (min pairwise DINOv2 similarity across 4 views),
             CI score (early timestep average t=0~15)
  3. Define collapse label:
       collapse = 1 if MinSim < threshold (bottom 25th percentile within same setting)
       collapse = 0 otherwise
  4. Store as row: (seed, setting, method, MinSim, CI, collapse)

Output:
  experiments/data/multiview/master_results.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import itertools

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, "data", "multiview")
FIG_DIR      = os.path.join(SCRIPT_DIR, "figures", "multiview")

# ── Constants ────────────────────────────────────────────────────────────────
EARLY_T_END     = 16          # CI feature from t=0..15
COLLAPSE_PCTILE = 25          # bottom 25th percentile → collapse

PROMPT_TO_SETTING = {
    "main":              "main",
    "mvdream_explicit":  "explicit",
    "stress_test":       "stress",
}

VALID_CONDITIONS = ["control", "intervention"]
MODELS = ["sd15", "sdxl", "mvdream"]
SEEDS  = list(range(42, 52))

SEP = "=" * 70


def ci_feature(ci_traj: np.ndarray, t_end: int = EARLY_T_END) -> float:
    """nanmean of CI values in t=0..t_end-1 (skipping NaN warm-up)."""
    early = ci_traj[:t_end]
    valid = early[~np.isnan(early)]
    return float(np.mean(valid)) if len(valid) > 0 else 0.0


def split_views(combined_img: Image.Image) -> list:
    """Splits a horizontally concatenated 4-view image into separate views."""
    w, h = combined_img.size
    view_w = w // 4
    views = []
    for i in range(4):
        box = (i * view_w, 0, (i + 1) * view_w, h)
        views.append(combined_img.crop(box))
    return views


def compute_minsim(model, transform, device, img: Image.Image) -> float:
    """Compute minimum pairwise DINOv2 cosine similarity across 4 views."""
    views = split_views(img)
    tensors = [transform(v.convert("RGB")).unsqueeze(0).to(device) for v in views]
    batch = torch.cat(tensors, dim=0)  # [4, 3, 224, 224]

    with torch.no_grad():
        features = model(batch)  # [4, dim]
        features = F.normalize(features, p=2, dim=-1)

    pairwise_sims = []
    for i, j in itertools.combinations(range(4), 2):
        sim = (features[i] * features[j]).sum().item()
        pairwise_sims.append(sim)

    return float(np.min(pairwise_sims))


def build_dataset():
    print(f"\n{SEP}")
    print("  BUILD MASTER RESULTS CSV")
    print(f"{SEP}\n")

    # ── Load DINOv2 ──────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading DINOv2 on {device}...")
    sys.stdout.flush()

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = model.to(device)
    model.eval()
    print("  ✓ DINOv2 loaded\n")
    sys.stdout.flush()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ── Iterate all samples ──────────────────────────────────────────────
    rows = []
    skipped = []
    total = len(MODELS) * len(PROMPT_TO_SETTING) * len(VALID_CONDITIONS) * len(SEEDS)
    processed = 0

    for model_key in MODELS:
        for prompt_key, setting in PROMPT_TO_SETTING.items():
            for cond in VALID_CONDITIONS:
                for seed in SEEDS:
                    tag = f"{model_key}_{prompt_key}_{cond}_{seed}"
                    processed += 1

                    # ── CI trajectory ────────────────────────────────
                    ci_path = os.path.join(DATA_DIR, f"{tag}_ci.npy")
                    if not os.path.exists(ci_path):
                        skipped.append((tag, "ci.npy missing"))
                        continue
                    ci_traj = np.load(ci_path).astype(np.float64)
                    ci_score = ci_feature(ci_traj)

                    # ── Image for MinSim ─────────────────────────────
                    img_path = os.path.join(FIG_DIR, f"{tag}.png")
                    if not os.path.exists(img_path):
                        skipped.append((tag, "image missing"))
                        continue

                    img = Image.open(img_path)
                    if img.width != 4 * img.height:
                        # SDXL images are tiny placeholders (3129 bytes)
                        # Still compute if the aspect ratio is right
                        skipped.append((tag, f"bad aspect {img.width}x{img.height}"))
                        continue

                    try:
                        min_sim = compute_minsim(model, transform, device, img)
                    except Exception as e:
                        skipped.append((tag, f"MinSim error: {e}"))
                        continue

                    rows.append({
                        "seed":    seed,
                        "setting": setting,
                        "method":  cond,
                        "MinSim":  round(min_sim, 6),
                        "CI":      round(ci_score, 6),
                    })
                    print(f"  [{processed}/{total}] {tag}  MinSim={min_sim:.4f}  CI={ci_score:.4f}")
                    sys.stdout.flush()

    print(f"\n  Processed: {len(rows)} samples")
    if skipped:
        print(f"  Skipped:   {len(skipped)} samples")
        for tag, reason in skipped[:15]:
            print(f"    - {tag}: {reason}")
        if len(skipped) > 15:
            print(f"    ... and {len(skipped)-15} more")

    if len(rows) == 0:
        print("\n  ERROR: No valid samples found. Cannot build dataset.")
        return

    # ── Step 3: Collapse label ───────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  STEP 3 — Collapse labels (bottom {COLLAPSE_PCTILE}th %-ile per setting)")
    print(f"{SEP}\n")

    df = pd.DataFrame(rows)
    df["collapse"] = 0

    for setting in sorted(df["setting"].unique()):
        mask = df["setting"] == setting
        threshold = np.percentile(df.loc[mask, "MinSim"], COLLAPSE_PCTILE)
        n_collapsed = (df.loc[mask, "MinSim"] < threshold).sum()
        df.loc[mask & (df["MinSim"] < threshold), "collapse"] = 1
        print(f"  '{setting}': threshold={threshold:.4f}, "
              f"collapsed={n_collapsed}/{mask.sum()}")

    # ── Step 4: Save ─────────────────────────────────────────────────────
    df = df[["seed", "setting", "method", "MinSim", "CI", "collapse"]]
    out_path = os.path.join(DATA_DIR, "master_results.csv")
    df.to_csv(out_path, index=False)

    print(f"\n{SEP}")
    print(f"  SAVED: {out_path}")
    print(f"  Rows: {len(df)}")
    print(f"{SEP}")
    print(f"\n  Collapse distribution:")
    print(df.groupby(["setting", "method"])["collapse"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "n_collapsed", "count": "n_total"})
            .to_string())
    print(f"\n  MinSim summary by setting × method:")
    print(df.groupby(["setting", "method"])["MinSim"]
            .agg(["mean", "std", "min", "max"])
            .round(4)
            .to_string())
    print(f"\n  CI summary by setting × method:")
    print(df.groupby(["setting", "method"])["CI"]
            .agg(["mean", "std", "min", "max"])
            .round(4)
            .to_string())
    print(f"\n{SEP}\n")


if __name__ == "__main__":
    build_dataset()
