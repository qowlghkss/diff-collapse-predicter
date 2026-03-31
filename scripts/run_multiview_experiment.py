import os
import sys
import json
import numpy as np
import argparse
from tqdm import tqdm

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CI_RUNNER_DIR = os.path.join(PROJECT_ROOT, "src", "diffusion")
sys.path.insert(0, CI_RUNNER_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "metrics"))

import torch
from ci_runner import CIRunner
from multiview_consistency import MultiViewConsistencyEvaluator
from PIL import Image

# ─────────────────────────── Configuration ───────────────────────────────────
MODELS = {
    "sd15": "runwayml/stable-diffusion-v1-5",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "mvdream": "ashawkey/mvdream-sd2.1-diffusers"
}

PROMPTS = {
    "main": "A highly detailed mechanical gear assembly with interlocking gears, precise teeth alignment, metallic surfaces, rotational symmetry, industrial design, sharp edges, realistic lighting",
    "mvdream_explicit": "A 3D object of a mechanical gear system, consistent geometry across views, same object identity, multiple viewpoints, front view, side view, top view, realistic rendering",
    "stress_test": "A complex mechanical engine with pipes, gears, repeating components, intricate structure, industrial machinery, high detail",
    "baseline": "A single smooth ceramic vase, minimal structure, soft lighting, clean background"
}

SEEDS = range(42, 52)  # 10 seeds: 42 to 51
NUM_STEPS = 50
INTERVENTION_T = 12
INTERVENTION_BOOST = 0.15
SHOCK_T = 20
SHOCK_BOOST = -0.20  # Negative boost for structural perturbation

OUT_BASE = os.path.join(SCRIPT_DIR, "data", "multiview")
IMG_BASE = os.path.join(SCRIPT_DIR, "figures", "multiview")
os.makedirs(OUT_BASE, exist_ok=True)
os.makedirs(IMG_BASE, exist_ok=True)

# ─────────────────────────── Execution ───────────────────────────────────────
def get_condition_params(cond):
    if cond == "control":
        return None, 0.0
    elif cond == "intervention":
        return INTERVENTION_T, INTERVENTION_BOOST
    elif cond == "shock":
        return SHOCK_T, SHOCK_BOOST
    raise ValueError(f"Unknown condition {cond}")

def run_experiment():
    results = []
    
    # Initialize Evaluator (loads DINOv2 once)
    print("Loading MultiView Consistency Evaluator...")
    evaluator = MultiViewConsistencyEvaluator()
    
    for model_key, model_id in MODELS.items():
        print(f"\n{'='*60}\nEvaluating Model: {model_id}\n{'='*60}")
        # Initialize Runner specifically for this model
        runner = CIRunner(model_id=model_id)
        
        for prompt_key, prompt_text in PROMPTS.items():
            print(f"\n  Prompt: {prompt_key}")
            
            # Monkeypatch the module level PROMPT in ci_runner because it uses a global PROMPT
            import ci_runner as cr
            cr.PROMPT = prompt_text
            cr.NUM_STEPS = NUM_STEPS
            
            for cond in ["control", "intervention", "shock"]:
                int_step, int_boost = get_condition_params(cond)
                
                print(f"    Condition: {cond}")
                
                for seed in tqdm(SEEDS, desc=f"    Seeds"):
                    out_img_path = os.path.join(IMG_BASE, f"{model_key}_{prompt_key}_{cond}_{seed}.png")
                    out_ci_path = os.path.join(OUT_BASE, f"{model_key}_{prompt_key}_{cond}_{seed}_ci.npy")
                    out_thin_path = os.path.join(OUT_BASE, f"{model_key}_{prompt_key}_{cond}_{seed}_thin.npy")
                    
                    # We can skip if already computed to allow resuming
                    if os.path.exists(out_img_path) and os.path.exists(out_ci_path):
                        ci_traj = np.load(out_ci_path)
                        img = Image.open(out_img_path)
                    else:
                        runner.intervention_step = int_step
                        runner.intervention_boost = int_boost
                        
                        ci_traj, thin_traj, img = runner.run_seed(seed)
                        
                        np.save(out_ci_path, ci_traj)
                        np.save(out_thin_path, thin_traj)
                        if img is not None:
                            img.save(out_img_path)
                            
                    mean_sim, var_sim = None, None
                    if model_key == "mvdream" and img is not None:
                        # Compute Consistency for MVDream runs
                        # Only compute if the image has width = 4 * height (since we horizontally concatenated)
                        if img.width == 4 * img.height:
                            try:
                                mean_sim, var_sim = evaluator.compute_consistency(img)
                            except Exception as e:
                                print(f"Error computing consistency for seed {seed}: {e}")
                    
                    results.append({
                        "model": model_key,
                        "prompt": prompt_key,
                        "condition": cond,
                        "seed": seed,
                        "mean_similarity": mean_sim,
                        "variance": var_sim,
                        "ci_max": float(np.nanmax(ci_traj)) if not np.all(np.isnan(ci_traj)) else None,
                    })
                    
        # Clean up model to free VRAM
        del runner
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                    
    # Save Results
    import csv
    csv_path = os.path.join(OUT_BASE, "multiview_results.csv")
    with open(csv_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nExperiment Complete. Results saved to {csv_path}")

if __name__ == "__main__":
    import gc
    import torch
    
    run_experiment()

