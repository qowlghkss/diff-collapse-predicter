import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def set_style():
    """Sets a professional aesthetic for matplotlib plots."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Roboto", "Arial", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.frameon": True,
        "legend.edgecolor": "white",
        "figure.dpi": 200,
    })

def plot_roc_curves(model_results, title="Receiver Operating Characteristic (ROC)", output_path=None):
    """
    Plots multiple ROC curves on one figure.
    model_results: dict mapping {model_name: {"y_true": [], "y_prob": []}}
    """
    set_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    
    colors = plt.cm.get_cmap("tab10").colors
    for i, (name, data) in enumerate(model_results.items()):
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=colors[i % 10], label=f"{name} (AUC = {roc_auc:.3f})")
        
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", alpha=0.8)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="lower right", fontsize=10)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"Saved ROC plot to {output_path}")
    return fig

def plot_pr_curves(model_results, title="Precision-Recall (PR) Curve", output_path=None):
    """
    Plots multiple PR curves on one figure.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    
    colors = plt.cm.get_cmap("tab10").colors
    for i, (name, data) in enumerate(model_results.items()):
        precision, recall, _ = precision_recall_curve(data["y_true"], data["y_prob"])
        ax.plot(recall, precision, lw=2, color=colors[i % 10], label=name)
        
    # Baseline: random prediction is the fraction of positives
    any_y = list(model_results.values())[0]["y_true"]
    chance = np.mean(any_y)
    ax.axhline(y=chance, color="gray", lw=1, linestyle="--", alpha=0.8, label=f"Chance ({chance:.2f})")
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", fontsize=10)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"Saved PR plot to {output_path}")
    return fig

def plot_training_progress(train_scores, val_scores, x_axis, ylabel="Metric", title="Training Progress", output_path=None):
    """
    Plots training and validation curves.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(x_axis, train_scores, "o-", color="#1f77b4", lw=2, label="Train")
    ax.plot(x_axis, val_scores, "s-", color="#ff7f0e", lw=2, label="Validation")
    
    ax.set_xlabel("Training Samples" if x_axis[-1] > 100 else "Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"Saved progress plot to {output_path}")
    return fig
