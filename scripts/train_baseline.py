import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add src to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.baselines.models import RandomPredictor, MajorityPredictor, HeuristicPredictor

# Optional Task 1 import
try:
    from src.utils.seed_utils import set_seed
except ImportError:
    def set_seed(seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
    print("Warning: could not import torch-based set_seed, using basic fallback.")

def load_data(split_info_path):
    with open(split_info_path, "r") as f:
        records = json.load(f)
    
    # Extract features: mean of CI trajectory from t=0..15
    # We load the .npy files referenced in split_info
    X = []
    y = []
    splits = []
    
    print(f"Loading features for {len(records)} samples...")
    for r in records:
        try:
            ci_traj = np.load(r["ci_path"])
            # EARLY_T_END = 16 (0..15)
            early = ci_traj[:16]
            valid = early[~np.isnan(early)]
            ci_feat = float(np.mean(valid)) if len(valid) > 0 else 0.0
            
            X.append(ci_feat)
            y.append(r["label"])
            splits.append(r["split"])
        except Exception as e:
            print(f"Error loading {r['ci_path']}: {e}")
            
    df = pd.DataFrame({"X": X, "y": y, "split": splits})
    return df

def evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.5
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate baseline models.")
    parser.add_argument("--split-info", type=str, default="experiments/data/split_info.json", help="Path to split_info.json")
    parser.add_argument("--output", type=str, default="experiments/metrics/baseline_results.json", help="Path to save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    if not os.path.exists(args.split_info):
        print(f"Error: {args.split_info} not found. Run split_data.py first.")
        return
        
    df = load_data(args.split_info)
    
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    
    X_train, y_train = train_df["X"].values, train_df["y"].values
    X_val, y_val = val_df["X"].values, val_df["y"].values
    X_test, y_test = test_df["X"].values, test_df["y"].values
    
    models = {
        "Random": RandomPredictor(random_state=args.seed),
        "Majority": MajorityPredictor(),
        "Heuristic (CI Threshold)": HeuristicPredictor()
    }
    
    results = {}
    print("\nEvaluating Baselines on Test Set:")
    for name, model in models.items():
        # Heuristic optimization on Train+Val
        if name == "Heuristic (CI Threshold)":
            X_tune = np.concatenate([X_train, X_val])
            y_tune = np.concatenate([y_train, y_val])
            model.fit(X_tune, y_tune)
            print(f"  {name} optimized threshold: {model.threshold:.4f}")
            
        metrics = evaluate(model, X_train, y_train, X_test, y_test)
        
        # Add predictions for plotting
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "metrics": metrics,
            "y_true": y_test.tolist(),
            "y_prob": y_prob.tolist()
        }
        print(f"  {name:25s} | Accuracy: {metrics['accuracy']:.4f} | AUC: {metrics['auc_roc']:.4f} | F1: {metrics['f1']:.4f}")
        
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nBaseline results saved to {args.output}")

if __name__ == "__main__":
    main()
