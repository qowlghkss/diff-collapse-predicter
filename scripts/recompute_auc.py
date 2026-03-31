#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DATA_DIR = Path("data/multiview")
MASTER_CSV = DATA_DIR / "master_results.csv"
OUT_DIR = Path("results")

SETTING_TO_PROMPT = {
    "main": "main",
    "explicit": "mvdream_explicit",
    "stress": "stress_test",
}


def interpolate_nan(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(float).copy()
    idx = np.arange(len(x))
    good = ~np.isnan(x)
    if good.sum() == 0:
        return np.zeros_like(x, dtype=float)
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return np.nan_to_num(x, nan=0.0)


def load_clean_dataset() -> pd.DataFrame:
    if not MASTER_CSV.exists():
        raise FileNotFoundError(f"Missing {MASTER_CSV}")

    df = pd.read_csv(MASTER_CSV)
    ctrl = df[df["method"] == "control"].copy()

    rows: List[Dict] = []
    nan_before = 0
    nan_after = 0

    for r in ctrl.itertuples(index=False):
        prompt = SETTING_TO_PROMPT.get(str(r.setting), str(r.setting))
        ci_path = DATA_DIR / f"mvdream_{prompt}_control_{int(r.seed)}_ci.npy"
        if not ci_path.exists():
            continue

        ci = np.load(ci_path)
        nan_before += int(np.isnan(ci).sum())
        ci_clean = interpolate_nan(ci)
        nan_after += int(np.isnan(ci_clean).sum())

        early = ci_clean[:21]
        t_peak = int(np.argmax(early))
        ci_peak = float(np.max(early))

        rows.append(
            {
                "setting": str(r.setting),
                "seed": int(r.seed),
                "collapse": int(r.collapse),
                "ci_peak": ci_peak,
                "t_peak": t_peak,
                "perceptual_score": float(r.MinSim) if "MinSim" in ctrl.columns else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    out.attrs["nan_before"] = nan_before
    out.attrs["nan_after"] = nan_after
    return out


def compute_and_save_auc() -> Dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_clean_dataset()

    y_true = data["collapse"].astype(int).to_numpy()
    y_score = data["ci_peak"].astype(float).to_numpy()

    if len(np.unique(y_true)) < 2:
        auc = 0.5
    else:
        auc = float(roc_auc_score(y_true, y_score))

    # enforce directional interpretation: larger score => more collapse risk
    if auc < 0.5:
        y_score = -y_score
        auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) >= 2 else 0.5

    auc_payload = {
        "auc": float(auc),
        "n_samples": int(len(y_true)),
        "nan_before": int(data.attrs.get("nan_before", 0)),
        "nan_after": int(data.attrs.get("nan_after", 0)),
    }

    with open(OUT_DIR / "auc.json", "w", encoding="utf-8") as f:
        json.dump(auc_payload, f, indent=2)
    with open(OUT_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(y_true.tolist(), f)
    with open(OUT_DIR / "scores.json", "w", encoding="utf-8") as f:
        json.dump(y_score.tolist(), f)

    data.to_csv(OUT_DIR / "ci_cleaned_dataset.csv", index=False)
    return auc_payload


if __name__ == "__main__":
    res = compute_and_save_auc()
    print(json.dumps(res, indent=2))
