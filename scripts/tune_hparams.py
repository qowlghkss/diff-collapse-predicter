import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.seed_utils import set_seed

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-info", type=str, default="experiments/data/split_info.json")
    parser.add_argument("--output", type=str, default="experiments/metrics/best_hparams.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Load data
    with open(args.split_info, "r") as f:
        data = json.load(f)
        
    X_train, y_train = [], []
    X_val, y_val = [], []
    
    for r in data:
        ci_traj = np.load(r["ci_path"])
        early = ci_traj[:16]
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
    
    best_f1 = -1
    best_params = {}
    
    # Grid search
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    weights = [None, "balanced"]
    
    print("Tuning hyperparameters...")
    for C in C_values:
        for w in weights:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=C, class_weight=w, max_iter=2000, random_state=args.seed))
            ])
            model.fit(X_train, y_train)
            
            y_prob = model.predict_proba(X_val)[:, 1]
            
            # Sub-tuning: Find best threshold for F1 on Val set
            precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
            # F1 = 2 * (P * R) / (P + R)
            f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            idx = np.argmax(f1s)
            
            current_f1 = f1s[idx]
            current_threshold = thresholds[idx] if idx < len(thresholds) else 0.5
            
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_params = {
                    "C": C,
                    "class_weight": w,
                    "threshold": float(current_threshold),
                    "val_f1": float(current_f1)
                }
                
    print(f"Best Params: {best_params}")
    
    with open(args.output, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved best hparams to {args.output}")

if __name__ == "__main__":
    main()
