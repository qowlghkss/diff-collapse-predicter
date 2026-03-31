import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add src to path
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.visualization.plotting import plot_training_progress
from src.utils.seed_utils import set_seed

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-info", type=str, default="experiments/data/split_info.json")
    parser.add_argument("--output-dir", type=str, default="figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    with open(args.split_info, "r") as f:
        data = json.load(f)
        
    X, y = [], []
    for r in data:
        # Load CI feature (precomputed or compute here)
        ci_traj = np.load(r["ci_path"])
        early = ci_traj[:16]
        valid = early[~np.isnan(early)]
        X.append(float(np.mean(valid)) if len(valid) > 0 else 0.0)
        y.append(r["label"])
        
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    
    print("Generating learning curves...")
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring="roc_auc", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=args.seed
    )
    
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    
    output_path = os.path.join(args.output_dir, "learning_curves.png")
    plot_training_progress(
        train_mean, val_mean, train_sizes, 
        ylabel="ROC-AUC", 
        title="Learning Curves (Logistic Regression)",
        output_path=output_path
    )

if __name__ == "__main__":
    main()
