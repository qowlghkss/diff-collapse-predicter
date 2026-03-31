"""
run_intervention_experiments.py
================================
Single-script runner for all intervention experiments:
  1. Baseline         — Control vs Fixed-t12 vs Random
  2. Timing Sweep     — intervention_step ∈ [6, 8, 10, 12, 14, 16]
  3. Boost Sweep      — boost ∈ [0.05, 0.10, 0.15, 0.20, 0.25]

Reuses CIRunner and shock_recovery_collapse from the existing pipeline.
Does not modify those classes or functions.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Resolve imports ────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CI_RUNNER_DIR = os.path.join(PROJECT_ROOT, "experiments", "ci_prediction")

if CI_RUNNER_DIR not in sys.path:
    sys.path.insert(0, CI_RUNNER_DIR)

from ci_runner import CIRunner          # existing class — not modified
import ci_runner as _cr                 # to patch NUM_STEPS / PROMPT

# ── Experiment configuration ───────────────────────────────────────────────────
MODEL_ID        = "runwayml/stable-diffusion-v1-5"
NUM_STEPS       = 25
GUIDANCE_SCALE  = 7.5
N_SIMS          = 100
SEEDS           = list(range(N_SIMS))      # seeds 0–99

TIMING_STEPS    = [6, 8, 10, 12, 14, 16]
BOOST_VALUES    = [0.05, 0.10, 0.15, 0.20, 0.25]

FIXED_T         = 12
FIXED_BOOST     = 0.15
RAND_T_LOW      = 5
RAND_T_HI       = 18

FIGURES_DIR     = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Collapse definition (mirrors Phase 2 / final_validation)
T_SHOCK      = 20
SHOCK_MAG    = 0.10
RECOVER_W    = 15
RECOVER_FRAC = 0.90


# ═══════════════════════════════════════════════════════════════════════════════
#  Core helpers
# ═══════════════════════════════════════════════════════════════════════════════

def shock_recovery_collapse(thin_traj: np.ndarray) -> bool:
    """
    Phase 2 collapse definition.
    Simulate 10% edge removal at T_SHOCK; collapse = no recovery within RECOVER_W.
    (Replicates the logic already present in the pipeline — not modifying CIRunner.)
    """
    T = len(thin_traj)
    if T_SHOCK >= T:
        return False
    pre = float(thin_traj[T_SHOCK - 1]) if T_SHOCK > 0 else float(thin_traj[0])
    threshold = RECOVER_FRAC * pre
    for dt in range(1, RECOVER_W + 1):
        t = T_SHOCK + dt
        if t >= T:
            break
        if thin_traj[t] >= threshold:
            return False
    return True


def get_random_intervention_step(seed: int) -> int:
    """Deterministic per-seed random step in [RAND_T_LOW, RAND_T_HI]."""
    rng = np.random.default_rng(seed=seed + 99999)
    return int(rng.integers(RAND_T_LOW, RAND_T_HI + 1))


def run_simulation(runner: CIRunner, seed: int,
                   intervention_step: int | None = None,
                   intervention_boost: float = 0.0) -> bool:
    """
    Run one diffusion job and return whether collapse occurred.

    Parameters
    ----------
    runner            : CIRunner instance (shared, pre-loaded)
    seed              : generator seed
    intervention_step : step index at which latent boost fires (None = control)
    intervention_boost: fractional boost magnitude (0.0 = no boost)

    Returns
    -------
    bool — True if collapse detected
    """
    runner.intervention_step  = intervention_step
    runner.intervention_boost = intervention_boost
    _, thin_traj, _ = runner.run_seed(seed)
    return shock_recovery_collapse(thin_traj)


def evaluate_condition(runner: CIRunner, seeds: list[int],
                        intervention_step: int | None,
                        intervention_boost: float,
                        label: str = "") -> float:
    """
    Run all seeds for one condition and return collapse rate.
    Prints a per-seed summary and a final rate.
    """
    collapses = 0
    for i, seed in enumerate(seeds):
        collapsed = run_simulation(runner, seed,
                                   intervention_step=intervention_step,
                                   intervention_boost=intervention_boost)
        if collapsed:
            collapses += 1
        if (i + 1) % 20 == 0 or (i + 1) == len(seeds):
            print(f"    {label}  [{i+1:>3d}/{len(seeds)}]  "
                  f"collapses so far: {collapses}")
    rate = collapses / len(seeds)
    return rate


# ═══════════════════════════════════════════════════════════════════════════════
#  Experiment 1 — Baselines
# ═══════════════════════════════════════════════════════════════════════════════

def run_baselines(runner: CIRunner) -> dict[str, float]:
    """
    Control vs Fixed-t12 vs Random intervention.
    Returns {'Control': rate, 'Fixed_t12': rate, 'Random': rate}
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1 — BASELINE COMPARISON")
    print("=" * 60)

    results: dict[str, float] = {}

    # Control
    print("\n  [1/3] Control (no intervention)")
    results["Control"] = evaluate_condition(
        runner, SEEDS,
        intervention_step=None,
        intervention_boost=0.0,
        label="Control",
    )

    # Fixed at t=12
    print(f"\n  [2/3] Fixed intervention (t={FIXED_T}, boost={FIXED_BOOST})")
    results["Fixed_t12"] = evaluate_condition(
        runner, SEEDS,
        intervention_step=FIXED_T,
        intervention_boost=FIXED_BOOST,
        label="Fixed_t12",
    )

    # Random timing per seed
    print(f"\n  [3/3] Random intervention (t∈[{RAND_T_LOW},{RAND_T_HI}], boost={FIXED_BOOST})")
    rand_collapses = 0
    for i, seed in enumerate(SEEDS):
        rand_t     = get_random_intervention_step(seed)
        collapsed  = run_simulation(runner, seed,
                                    intervention_step=rand_t,
                                    intervention_boost=FIXED_BOOST)
        if collapsed:
            rand_collapses += 1
        if (i + 1) % 20 == 0 or (i + 1) == N_SIMS:
            print(f"    Random  [{i+1:>3d}/{N_SIMS}]  "
                  f"collapses so far: {rand_collapses}")
    results["Random"] = rand_collapses / N_SIMS

    # Print summary table
    print("\n" + "-" * 35)
    print(f"  {'Strategy':<15s}  {'Collapse Rate':>12s}")
    print("-" * 35)
    for strategy, rate in results.items():
        print(f"  {strategy:<15s}  {rate*100:>11.1f}%")
    print("-" * 35)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Experiment 2 — Timing Sweep
# ═══════════════════════════════════════════════════════════════════════════════

def run_timing_sweep(runner: CIRunner) -> dict[int, float]:
    """
    Sweep intervention_step over TIMING_STEPS at fixed boost.
    Returns {step: collapse_rate}
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2 — TIMING SWEEP")
    print(f"  Sweep: {TIMING_STEPS}  |  boost={FIXED_BOOST}  |  N={N_SIMS}")
    print("=" * 60)

    results: dict[int, float] = {}

    for t in TIMING_STEPS:
        print(f"\n  Timing step = {t}")
        rate = evaluate_condition(
            runner, SEEDS,
            intervention_step=t,
            intervention_boost=FIXED_BOOST,
            label=f"t={t}",
        )
        results[t] = rate

    # Print summary table
    print("\n" + "-" * 30)
    print(f"  {'Timing':>8s}  {'Collapse Rate':>12s}")
    print("-" * 30)
    for t, rate in results.items():
        print(f"  {t:>8d}  {rate*100:>11.1f}%")
    print("-" * 30)

    # Plot
    xs = list(results.keys())
    ys = [results[x] * 100 for x in xs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, marker="o", color="#3B82F6", lw=2.5, markersize=8)
    ax.axvline(FIXED_T, color="#E05252", lw=1.5, ls="--",
               label=f"Phase 2 trigger (t={FIXED_T})")
    ax.set_xlabel("Intervention Timing (step)", fontsize=12)
    ax.set_ylabel("Collapse Rate (%)", fontsize=12)
    ax.set_title("Collapse Rate vs Intervention Timing", fontsize=13, fontweight="bold")
    ax.set_xticks(xs)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "timing_sweep.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Experiment 3 — Boost Sweep
# ═══════════════════════════════════════════════════════════════════════════════

def run_boost_sweep(runner: CIRunner) -> dict[float, float]:
    """
    Sweep boost strength over BOOST_VALUES at fixed timing.
    Returns {boost: collapse_rate}
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3 — BOOST SWEEP")
    print(f"  Sweep: {BOOST_VALUES}  |  t={FIXED_T}  |  N={N_SIMS}")
    print("=" * 60)

    results: dict[float, float] = {}

    for boost in BOOST_VALUES:
        print(f"\n  Boost = {boost:.2f}")
        rate = evaluate_condition(
            runner, SEEDS,
            intervention_step=FIXED_T,
            intervention_boost=boost,
            label=f"boost={boost:.2f}",
        )
        results[boost] = rate

    # Print summary table
    print("\n" + "-" * 30)
    print(f"  {'Boost':>8s}  {'Collapse Rate':>12s}")
    print("-" * 30)
    for boost, rate in results.items():
        print(f"  {boost:>8.2f}  {rate*100:>11.1f}%")
    print("-" * 30)

    # Plot
    xs = list(results.keys())
    ys = [results[x] * 100 for x in xs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, marker="s", color="#10B981", lw=2.5, markersize=8)
    ax.axvline(FIXED_BOOST, color="#E05252", lw=1.5, ls="--",
               label=f"Phase 2 boost ({FIXED_BOOST})")
    ax.set_xlabel("Boost Strength", fontsize=12)
    ax.set_ylabel("Collapse Rate (%)", fontsize=12)
    ax.set_title("Collapse Rate vs Boost Strength", fontsize=13, fontweight="bold")
    ax.set_xticks(xs)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "boost_sweep.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  main()
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  INTERVENTION EXPERIMENTS")
    print(f"  Model   : {MODEL_ID}")
    print(f"  Steps   : {NUM_STEPS}")
    print(f"  Seeds   : {SEEDS[0]}–{SEEDS[-1]}  (N={N_SIMS})")
    print("=" * 60)

    # Patch global NUM_STEPS before loading model
    _cr.NUM_STEPS = NUM_STEPS

    runner = CIRunner(model_id=MODEL_ID)
    runner.guidance_scale = GUIDANCE_SCALE

    # ── Run all three experiments ──────────────────────────────────
    baseline_results = run_baselines(runner)
    timing_results   = run_timing_sweep(runner)
    boost_results    = run_boost_sweep(runner)

    # ── Final combined summary ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ALL EXPERIMENTS COMPLETE — COMBINED SUMMARY")
    print("=" * 60)

    print("\n  [Baselines]")
    print(f"  {'Strategy':<15s}  {'Collapse Rate':>12s}")
    for k, v in baseline_results.items():
        print(f"  {k:<15s}  {v*100:>11.1f}%")

    print("\n  [Timing Sweep]")
    print(f"  {'Step':>6s}  {'Collapse Rate':>12s}")
    for k, v in timing_results.items():
        print(f"  {k:>6d}  {v*100:>11.1f}%")

    print("\n  [Boost Sweep]")
    print(f"  {'Boost':>6s}  {'Collapse Rate':>12s}")
    for k, v in boost_results.items():
        print(f"  {k:>6.2f}  {v*100:>11.1f}%")

    print(f"\n  Figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
