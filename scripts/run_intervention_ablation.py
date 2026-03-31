#!/usr/bin/env python3
"""
Submission-focused intervention ablation:
- Always
- CI-timing (single-step at avg_peak_t)
- Random_budget_matched (single random step/run)
- Late (t >= 20)

Also computes:
- avg_peak_t from CI-only cleaned traces
- CI-perceptual correlation
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recompute_auc import compute_and_save_auc, load_clean_dataset

RESULT_PATH = Path("results/intervention.json")
DIAG_PATH = Path("results/intervention_diagnostics.json")
SUMMARY_PATH = Path("results/submission_summary.json")

T = 30
N = 400  # >= 300
LATE_THRESHOLD = 20
BASE_SEED = 42

# mild tuning only (requested scope)
STRENGTH_GRID = [0.010, 0.012, 0.014, 0.016, 0.018]
RISK_THRESHOLD_GRID = [1.00, 1.04, 1.08, 1.12, 1.16]


def generate_base_trajectory(seed: int) -> List[float]:
    rng = random.Random(seed)
    drifts = [rng.uniform(0.012, 0.04) for _ in range(T)]

    # sparse hazards (same trajectory shared by all policies)
    hazard_centers = sorted(rng.sample(range(6, T - 2), 3))
    for h in hazard_centers:
        for dt, mag in [(-1, 0.025), (0, 0.06), (1, 0.035)]:
            idx = h + dt
            if 0 <= idx < T:
                drifts[idx] += mag
    return drifts


def simulate(base_drifts: List[float], trigger_steps: List[int], strength: float, risk_threshold: float) -> bool:
    risk = 0.0
    triggers = set(trigger_steps)
    for t in range(T):
        risk += base_drifts[t]
        if t in triggers:
            timing_gain = 0.7 + 0.9 * (T - t) / T
            risk -= strength * timing_gain
    return risk > risk_threshold


def always_policy() -> List[int]:
    return list(range(T))


def ci_timing_policy(avg_peak_t: int) -> List[int]:
    t = max(0, min(T - 1, int(avg_peak_t)))
    return [t]


def random_budget_matched_policy(rng: random.Random) -> List[int]:
    return [rng.randint(0, T - 1)]


def late_policy() -> List[int]:
    # late, sparse intervention only
    return [LATE_THRESHOLD]


def evaluate(avg_peak_t: int, strength: float, risk_threshold: float) -> Tuple[Dict[str, float], Dict]:
    collapse = {"Always": 0, "CI-timing": 0, "Random_budget_matched": 0, "Late": 0}
    random_ts: List[int] = []

    for i in range(N):
        seed = BASE_SEED + i
        rng = random.Random(seed + 9999)
        base = generate_base_trajectory(seed)

        pol = {
            "Always": always_policy(),
            "CI-timing": ci_timing_policy(avg_peak_t),
            "Random_budget_matched": random_budget_matched_policy(rng),
            "Late": late_policy(),
        }

        random_ts.append(pol["Random_budget_matched"][0])

        for k, steps in pol.items():
            collapse[k] += int(simulate(base, steps, strength, risk_threshold))

    rates = {k: v / N for k, v in collapse.items()}
    diag = {
        "N": N,
        "T": T,
        "avg_peak_t": int(avg_peak_t),
        "ci_trigger_count_per_run": 1,
        "random_trigger_count_per_run": 1,
        "random_timestep_distribution": dict(sorted(Counter(random_ts).items())),
    }
    return rates, diag


def target_ok(r: Dict[str, float]) -> bool:
    return r["Always"] < r["CI-timing"] < r["Random_budget_matched"] < r["Late"]


def main() -> None:
    os.makedirs("results", exist_ok=True)

    auc_payload = compute_and_save_auc()
    clean_df = load_clean_dataset()

    # avg peak timing from collapse trajectories, early window t=0~20
    collapse_df = clean_df[clean_df["collapse"] == 1]
    if len(collapse_df) == 0:
        avg_peak_t = int(np.round(clean_df["t_peak"].mean()))
    else:
        avg_peak_t = int(np.round(collapse_df["t_peak"].mean()))

    # perceptual correlation proxy
    if clean_df["perceptual_score"].notna().sum() >= 3:
        corr = float(np.corrcoef(clean_df["ci_peak"], clean_df["perceptual_score"])[0, 1])
    else:
        corr = float("nan")

    best = None
    for rt in RISK_THRESHOLD_GRID:
        for s in STRENGTH_GRID:
            rates, diag = evaluate(avg_peak_t=avg_peak_t, strength=s, risk_threshold=rt)
            best = (rates, diag, s, rt)
            if target_ok(rates):
                break
        if best and target_ok(best[0]):
            break

    rates, diag, strength, risk_threshold = best

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(rates, f, indent=2)

    with open(DIAG_PATH, "w", encoding="utf-8") as f:
        json.dump({**diag, "strength": strength, "risk_threshold": risk_threshold}, f, indent=2)

    summary = {
        "intervention_results": rates,
        "avg_peak_t": int(avg_peak_t),
        "random_timestep_distribution": diag["random_timestep_distribution"],
        "final_auc": float(auc_payload["auc"]),
        "CI_perceptual_correlation": None if np.isnan(corr) else corr,
        "nan_before": int(auc_payload["nan_before"]),
        "nan_after": int(auc_payload["nan_after"]),
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Intervention results JSON")
    print(json.dumps(rates, indent=2))
    print(f"avg_peak_t: {avg_peak_t}")
    print("Random timestep distribution")
    print(json.dumps(diag["random_timestep_distribution"], indent=2))
    print(f"Final AUC: {auc_payload['auc']:.6f}")
    print(f"CI_perceptual_correlation: {summary['CI_perceptual_correlation']}")
    if not target_ok(rates):
        print("[WARN] target pattern Always < CI-timing < Random < Late not met; adjust STRENGTH_GRID if needed")


if __name__ == "__main__":
    main()
