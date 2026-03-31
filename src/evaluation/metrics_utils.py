import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, average_precision_score, 
                             roc_curve, precision_recall_curve)

def compute_all_metrics(y_true, y_prob, threshold=0.5):
    """
    Computes all standard binary classification metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    # Handle edge cases where only one class is present
    if len(np.unique(y_true)) < 2:
        auc = 0.5
        pr_auc = 0.0
    else:
        auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(auc),
        "pr_auc": float(pr_auc)
    }
    return metrics

def bootstrap_metric(y_true, y_prob, metric_fn, n_bootstrap=1000, seed=42):
    """
    Calculates 95% confidence intervals for a metric using bootstrapping.
    """
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_bootstrap):
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        # metric_fn should take (y_true, y_prob)
        scores.append(metric_fn(y_true[indices], y_prob[indices]))
    
    if not scores:
        return 0.0, 0.0, 0.0
        
    scores = np.array(scores)
    return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))
