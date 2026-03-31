import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.seed_utils import set_seed
from src.visualization.plotting import set_style

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-info", type=str, default="experiments/data/split_info.json")
    parser.add_argument("--output-dir", type=str, default="figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    set_style()
    
    with open(args.split_info, "r") as f:
        split_data = json.load(f)
        
    windows = [4, 8, 12, 16, 20, 24]
    val_aucs = []
    
    print("Running Window Length Ablation...")
    for w in windows:
        X_train, y_train = [], []
        X_val, y_val = [], []
        
        for r in split_data:
            ci_traj = np.load(r["ci_path"])
            early = ci_traj[:w]
            valid = early[~np.isnan(early)]
            feat = float(np.mean(valid)) if len(valid) > 0 else 0.0
            
            if r["split"] == "train":
                X_train.append(feat)
                y_train.append(r["label"])
            elif r["split"] == "val":
                X_val.append(feat)
                y_val.append(r["label"])
                
        X_train, y_train = np.array(X_train).reshape(-1, 1), np.array(y_train)
        X_val, y_val = np.array(X_val).reshape(-1, 1), np.array(y_val)
        
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", random_state=args.seed))
        ])
        model.fit(X_train, y_train)
        
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        val_aucs.append(auc)
        print(f"  Window {w:2d} steps | Val AUC: {auc:.4f}")
        
    # Plotting
    plt.figure(figsize=(7, 5))
    plt.plot(windows, val_aucs, "o-", color="#8B5CF6", lw=2, markersize=8)
    plt.xlabel("CI Feature Window Length (diffusion steps)", fontsize=12)
    plt.ylabel("Validation ROC-AUC", fontsize=12)
    plt.title("Ablation: Impact of Window Length on Predictivity", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, ls="--")
    
    output_path = os.path.join(args.output_dir, "ablation_window_length.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved ablation plot to {output_path}")

if __name__ == "__main__":
    main()
