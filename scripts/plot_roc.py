import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.visualization.plotting import plot_roc_curves, plot_pr_curves

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-results", type=str, default="experiments/metrics/main_model_results.json")
    parser.add_argument("--baseline-results", type=str, default="experiments/metrics/baseline_results.json")
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    if not os.path.exists(args.main_results) or not os.path.exists(args.baseline_results):
        print(f"Error: Missing results files ({args.main_results} or {args.baseline_results}).")
        return
        
    with open(args.main_results, "r") as f:
        main_res = json.load(f)
        
    with open(args.baseline_results, "r") as f:
        baseline_res = json.load(f)
        
    # Combine into a single dictionary for plotting
    model_results = {
        "Main Model (CI-LogReg)": {
            "y_true": np.array(main_res["y_true"]),
            "y_prob": np.array(main_res["y_prob"])
        }
    }
    
    # Add baselines
    for name, data in baseline_res.items():
        model_results[name] = {
            "y_true": np.array(data["y_true"]),
            "y_prob": np.array(data["y_prob"])
        }
        
    # Plot ROC
    plot_roc_curves(
        model_results, 
        title="Model Comparison — ROC Curves", 
        output_path=os.path.join(args.output_dir, "roc_comparison.png")
    )
    
    # Plot PR
    plot_pr_curves(
        model_results, 
        title="Model Comparison — Precision-Recall Curves", 
        output_path=os.path.join(args.output_dir, "pr_comparison.png")
    )

if __name__ == "__main__":
    main()
