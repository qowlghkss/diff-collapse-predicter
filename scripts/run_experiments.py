"""
Final Validation Experiment — Data Generation
==============================================
5 prompts × 20 seeds × 3 conditions = 300 runs.
Conditions: control | ci_intervention (t=12) | random_intervention (t=5..18 per seed)

Model  : runwayml/stable-diffusion-v1-5
Steps  : 25
Res    : 512×512
Boost  : 0.15 (same as Phase 2)

Reuses CIRunner from experiments/ci_prediction/ci_runner.py.

Outputs to: experiments/final_validation/data/{condition}/
"""

import os
import sys
import json
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── Paths ───────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CI_RUNNER_DIR = os.path.join(PROJECT_ROOT, "experiments", "ci_prediction")

# Add ci_prediction to import path so we can reuse its modules
if CI_RUNNER_DIR not in sys.path:
    sys.path.insert(0, CI_RUNNER_DIR)

OUT_BASE   = SCRIPT_DIR
DATA_DIR   = os.path.join(OUT_BASE, "data")
IMAGES_DIR = os.path.join(OUT_BASE, "images")

CTRL_DIR   = os.path.join(DATA_DIR, "control")
CI_DIR     = os.path.join(DATA_DIR, "ci_intervention")
RAND_DIR   = os.path.join(DATA_DIR, "random_intervention")

for d in [CTRL_DIR, CI_DIR, RAND_DIR, IMAGES_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────── Experiment Config ───────────────────────────────
MODEL_ID   = "runwayml/stable-diffusion-v1-5"
NUM_STEPS  = 25
GUIDANCE   = 7.5
BOOST      = 0.15
CI_INTV_T  = 12          # CI-guided fixed intervention step
RAND_T_LOW = 5           # random intervention range [5, 18]
RAND_T_HI  = 18

PROMPTS = [
    "A detailed 3D geometric structure",
    "A mechanical gear assembly",
    "A futuristic architectural structure",
    "A coral-like organic branching structure",
    "A symmetric abstract sculpture",
]
PROMPT_IDS = [f"p{i}" for i in range(len(PROMPTS))]

SEEDS = list(range(20))   # 0–19

# Collapse definition (same as Phase 2)
T_SHOCK      = 20
SHOCK_MAG    = 0.10
RECOVER_W    = 15
RECOVER_FRAC = 0.90


def get_random_intv_step(seed: int) -> int:
    """Per-seed deterministic random intervention step in [RAND_T_LOW, RAND_T_HI]."""
    rng = np.random.default_rng(seed=seed + 9999)   # offset to avoid seed correlation
    return int(rng.integers(RAND_T_LOW, RAND_T_HI + 1))


def shock_recovery_collapse(thin_traj: np.ndarray) -> bool:
    """
    Simulate 10% edge removal at T_SHOCK.
    Collapse = thin_traj never recovers to RECOVER_FRAC × pre-shock within RECOVER_W steps.
    """
    T = len(thin_traj)
    if T_SHOCK >= T:
        return False   # shock step out of range — no collapse
    pre = float(thin_traj[T_SHOCK - 1]) if T_SHOCK > 0 else float(thin_traj[0])
    threshold = RECOVER_FRAC * pre
    for dt in range(1, RECOVER_W + 1):
        t = T_SHOCK + dt
        if t >= T:
            break
        if thin_traj[t] >= threshold:
            return False   # recovered → not collapsed
    return True   # never recovered → collapsed


def run_one(runner, prompt: str, prompt_id: str, seed: int,
            condition: str, out_dir: str,
            intervention_step=None, intervention_boost=0.0):
    """
    Run a single diffusion job with the given config.
    Saves ci_traj, thin_traj, collapse json, and (optionally) final image.
    Returns (ci_traj, thin_traj, collapsed_bool) or None if already exists.
    """
    ci_path   = os.path.join(out_dir, f"ci_traj_{prompt_id}_{seed}.npy")
    thin_path = os.path.join(out_dir, f"thin_traj_{prompt_id}_{seed}.npy")
    col_path  = os.path.join(out_dir, f"collapse_{prompt_id}_{seed}.json")

    if os.path.exists(ci_path) and os.path.exists(thin_path) and os.path.exists(col_path):
        # Resume: load existing
        ci_traj   = np.load(ci_path)
        thin_traj = np.load(thin_path)
        with open(col_path) as f:
            meta = json.load(f)
        return ci_traj, thin_traj, meta["collapse"]

    # ── patch runner ──
    import ci_runner as cr
    cr.PROMPT     = prompt
    cr.NUM_STEPS  = NUM_STEPS
    runner.guidance_scale    = GUIDANCE
    runner.intervention_step  = intervention_step
    runner.intervention_boost = intervention_boost

    ci_traj, thin_traj, final_img = runner.run_seed(seed)

    collapsed = shock_recovery_collapse(thin_traj)

    np.save(ci_path,   ci_traj)
    np.save(thin_path, thin_traj)
    with open(col_path, "w") as f:
        json.dump({
            "prompt_id": prompt_id,
            "prompt": prompt,
            "seed": seed,
            "condition": condition,
            "collapse": bool(collapsed),
            "intervention_step": intervention_step,
            "intervention_boost": intervention_boost,
        }, f, indent=2)

    if final_img is not None:
        img_path = os.path.join(IMAGES_DIR, f"{condition}_{prompt_id}_{seed}.png")
        final_img.save(img_path)

    return ci_traj, thin_traj, collapsed


# ─────────────────────────── Main ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-idx", type=int, default=None,
                        help="Run only this prompt index (0-4). Default: all.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end",   type=int, default=19)
    parser.add_argument("--condition",  type=str, default=None,
                        choices=["control", "ci", "random"],
                        help="Run only this condition. Default: all three.")
    args = parser.parse_args()

    prompt_indices = [args.prompt_idx] if args.prompt_idx is not None else list(range(len(PROMPTS)))
    seed_range     = list(range(args.seed_start, args.seed_end + 1))
    conditions     = [args.condition] if args.condition else ["control", "ci", "random"]

    print(f"[CONFIG] Model       : {MODEL_ID}")
    print(f"[CONFIG] Steps       : {NUM_STEPS}")
    print(f"[CONFIG] Guidance    : {GUIDANCE}")
    print(f"[CONFIG] Boost       : {BOOST}")
    print(f"[CONFIG] CI intv t   : {CI_INTV_T}")
    print(f"[CONFIG] Rand range  : [{RAND_T_LOW}, {RAND_T_HI}]")
    print(f"[CONFIG] T_SHOCK     : {T_SHOCK}, Mag={SHOCK_MAG*100:.0f}%, Window={RECOVER_W}, Frac={RECOVER_FRAC}")
    print(f"[CONFIG] Prompts     : {prompt_indices}")
    print(f"[CONFIG] Seeds       : {seed_range[0]}–{seed_range[-1]}")
    print(f"[CONFIG] Conditions  : {conditions}")
    print()

    # Check for existing files before loading model (saves time if all done)
    total_expected = len(prompt_indices) * len(seed_range) * len(conditions)
    total_existing = 0
    for pi in prompt_indices:
        pid = PROMPT_IDS[pi]
        for s in seed_range:
            for cond in conditions:
                out_dir = {"control": CTRL_DIR, "ci": CI_DIR, "random": RAND_DIR}[cond]
                if os.path.exists(os.path.join(out_dir, f"collapse_{pid}_{s}.json")):
                    total_existing += 1
    print(f"[INFO] Already complete: {total_existing}/{total_expected}")
    if total_existing == total_expected:
        print("[INFO] All runs complete. Nothing to do.")
        return

    # Load model
    from ci_runner import CIRunner
    import ci_runner as cr

    # Override global NUM_STEPS before creating runner (affects traj arrays)
    cr.NUM_STEPS = NUM_STEPS

    runner = CIRunner(model_id=MODEL_ID)
    runner.guidance_scale = GUIDANCE

    total_done = total_existing
    for pi in prompt_indices:
        prompt = PROMPTS[pi]
        pid    = PROMPT_IDS[pi]
        print(f"\n{'='*70}")
        print(f"PROMPT {pi}: {prompt}")
        print(f"{'='*70}")

        for seed in seed_range:
            rand_t = get_random_intv_step(seed)
            print(f"\n  Seed {seed}  (rand_intv_t={rand_t})")

            cond_map = {
                "control": dict(out_dir=CTRL_DIR, intervention_step=None, intervention_boost=0.0),
                "ci":      dict(out_dir=CI_DIR,   intervention_step=CI_INTV_T, intervention_boost=BOOST),
                "random":  dict(out_dir=RAND_DIR,  intervention_step=rand_t,    intervention_boost=BOOST),
            }

            for cond in conditions:
                cfg  = cond_map[cond]
                ci_path = os.path.join(cfg["out_dir"], f"ci_traj_{pid}_{seed}.npy")
                if os.path.exists(ci_path):
                    print(f"    [{cond:8s}] already exists — skip")
                    continue
                print(f"    [{cond:8s}] running …")
                try:
                    ci_t, thin_t, col = run_one(
                        runner, prompt, pid, seed,
                        condition=cond,
                        out_dir=cfg["out_dir"],
                        intervention_step=cfg["intervention_step"],
                        intervention_boost=cfg["intervention_boost"],
                    )
                    total_done += 1
                    print(f"    [{cond:8s}] collapse={col}  "
                          f"CI_max={np.nanmax(ci_t):.3f}  "
                          f"thin_final={thin_t[-1]}  "
                          f"[{total_done}/{total_expected}]")
                except Exception as e:
                    print(f"    [{cond:8s}] ERROR: {e}")
                    import traceback; traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"DONE.  Total runs completed: {total_done}/{total_expected}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
