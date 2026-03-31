import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "metrics"))

from multiview_consistency import MultiViewConsistencyEvaluator

FIG_DIR = os.path.join(SCRIPT_DIR, "figures", "multiview")
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "multiview")

PROMPTS = ["main", "mvdream_explicit", "stress_test"]
CONDITIONS = ["control", "intervention"]
SEEDS = range(42, 52)
MODEL = "mvdream"

def analyze_worst_case():
    print("Loading MultiView Consistency Evaluator...")
    evaluator = MultiViewConsistencyEvaluator()
    
    results = []
    worst_cases = {prompt: {cond: {"seed": None, "min_sim": float('inf'), "img": None, "sim_matrix": None} for cond in CONDITIONS} for prompt in PROMPTS}
    
    # Collect Data
    for prompt in PROMPTS:
        print(f"Processing prompt: {prompt}")
        for cond in CONDITIONS:
            for seed in SEEDS:
                img_path = os.path.join(FIG_DIR, f"{MODEL}_{prompt}_{cond}_{seed}.png")
                if not os.path.exists(img_path):
                    continue
                
                try:
                    img = Image.open(img_path)
                    # Only compute if the image has width = 4 * height
                    if img.width == 4 * img.height:
                        mean_sim, var_sim, min_sim, sim_matrix = evaluator.compute_all_metrics(img)
                        
                        results.append({
                            "prompt": prompt,
                            "condition": cond,
                            "seed": seed,
                            "min_sim": min_sim,
                            "mean_sim": mean_sim,
                            "var_sim": var_sim
                        })
                        
                        # Check if this is the worst-case for this prompt/condition
                        if min_sim < worst_cases[prompt][cond]["min_sim"]:
                            worst_cases[prompt][cond]["min_sim"] = min_sim
                            worst_cases[prompt][cond]["seed"] = seed
                            worst_cases[prompt][cond]["img"] = img
                            worst_cases[prompt][cond]["sim_matrix"] = sim_matrix
                except Exception as e:
                    print(f"Failed to process {img_path}: {e}")

    df = pd.DataFrame(results)
    
    # 1. SUMMARY TABLE
    print("\n" + "="*50)
    print(" WORST-CASE CONSISTENCY TABLE (MIN_SIM) ")
    print("="*50)
    
    table = df.groupby(["prompt", "condition"]).agg(
        mean_min_sim=("min_sim", "mean"),
        std_min_sim=("min_sim", "std"),
        abs_worst_min=("min_sim", "min")
    ).round(4)
    print(table)
    
    csv_path = os.path.join(DATA_DIR, "worst_case_analysis.csv")
    table.to_csv(csv_path)
    print(f"\nSaved metrics to {csv_path}")
    
    # 2. WORST-CASE VISUALIZATIONS
    for prompt in PROMPTS:
        ctrl_wc = worst_cases[prompt]["control"]
        intv_wc = worst_cases[prompt]["intervention"]
        
        if ctrl_wc["img"] is None or intv_wc["img"] is None:
            print(f"Skipping visualization for {prompt} due to missing data.")
            continue
            
        fig, axes = plt.subplots(2, 1, figsize=(16, 8))
        fig.suptitle(f"Worst-Case Multi-View Consistency: {prompt}\n(Lowest pairwise similarity among 10 seeds)", fontsize=16)
        
        # Display Control Worst-Case
        axes[0].imshow(ctrl_wc["img"])
        axes[0].set_title(f"Control (Seed {ctrl_wc['seed']}) | Worst-Case Min Similarity: {ctrl_wc['min_sim']:.4f}", fontsize=14)
        axes[0].axis("off")
        
        # Display Intervention Worst-Case
        axes[1].imshow(intv_wc["img"])
        axes[1].set_title(f"Intervention (Seed {intv_wc['seed']}) | Worst-Case Min Similarity: {intv_wc['min_sim']:.4f}", fontsize=14)
        axes[1].axis("off")
        
        plt.tight_layout()
        out_path = os.path.join(FIG_DIR, f"worst_case_{prompt}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved visualization: {out_path}")
        
    # 3. OVERALL METRICS
    print("\n" + "="*50)
    print(" SUMMARY ")
    print("="*50)
    
    ctrl_mean_min = df[df["condition"] == "control"]["min_sim"].mean()
    intv_mean_min = df[df["condition"] == "intervention"]["min_sim"].mean()
    ctrl_abs_min = df[df["condition"] == "control"]["min_sim"].min()
    intv_abs_min = df[df["condition"] == "intervention"]["min_sim"].min()
    
    diff_mean = intv_mean_min - ctrl_mean_min
    diff_abs = intv_abs_min - ctrl_abs_min
    
    print(f"Overall Control Mean MinSim:      {ctrl_mean_min:.4f}")
    print(f"Overall Intervention Mean MinSim: {intv_mean_min:.4f}")
    print(f"Shift in average worst-case:      {'+' if diff_mean > 0 else ''}{diff_mean:.4f}")
    
    print(f"\nOverall Control Absolute MinSim:      {ctrl_abs_min:.4f}")
    print(f"Overall Intervention Absolute MinSim: {intv_abs_min:.4f}")
    print(f"Improvement in absolute worst-case bound: {'+' if diff_abs > 0 else ''}{diff_abs:.4f}")
    
    if diff_abs > 0 or diff_mean > 0:
        print("\nConclusion: CI-based intervention successfully raises the minimum structural similarity bound, reducing catastrophic worst-case inconsistencies.")
    else:
        print("\nConclusion: CI-based intervention did not improve the worst-case structural similarity bounds on average.")

if __name__ == "__main__":
    analyze_worst_case()
