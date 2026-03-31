import os
import json
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-results", type=str, default="experiments/metrics/main_model_results.json")
    parser.add_argument("--baseline-results", type=str, default="experiments/metrics/baseline_results.json")
    parser.add_argument("--output", type=str, default="experiments/metrics/statistical_tests.json")
    parser.add_argument("--hparams", type=str, default="experiments/metrics/best_hparams.json")
    args = parser.parse_args()
    
    if not os.path.exists(args.main_results) or not os.path.exists(args.baseline_results):
        print("Missing result files.")
        return
        
    with open(args.main_results, "r") as f:
        main_data = json.load(f)
    with open(args.baseline_results, "r") as f:
        baseline_data = json.load(f)
        
    # Threshold for main model from hparams or default
    threshold = 0.5
    if os.path.exists(args.hparams):
        with open(args.hparams, "r") as f:
            threshold = json.load(f).get("threshold", 0.5)
            
    y_true = np.array(main_data["y_true"])
    y_prob_main = np.array(main_data["y_prob"])
    y_pred_main = (y_prob_main >= threshold).astype(int)
    
    test_results = {}
    
    print("\nStatistical Significance Tests (vs Main Model):")
    for b_name, b_data in baseline_data.items():
        y_prob_b = np.array(b_data["y_prob"])
        y_pred_b = (y_prob_b >= 0.5).astype(int)
        
        # 1. McNemar's Test (Contingency table of correct/incorrect)
        # correct_m = (y_pred_main == y_true)
        # correct_b = (y_pred_b == y_true)
        # ... mcnemar ...
        
        # Actually McNemar is usually on (pred==true)
        # Table: [ [both_correct, main_corr_base_inc], [base_corr_main_inc, both_incorrect] ]
        c11 = np.sum((y_pred_main == y_true) & (y_pred_b == y_true))
        c12 = np.sum((y_pred_main == y_true) & (y_pred_b != y_true))
        c21 = np.sum((y_pred_main != y_true) & (y_pred_b == y_true))
        c22 = np.sum((y_pred_main != y_true) & (y_pred_b != y_true))
        
        table = [[c11, c12], [c21, c22]]
        mc_res = mcnemar(table, exact=True)
        
        # 2. Approximate T-test on bootstrapped AUC
        # (This is more common in ML papers than McNemar for AUC comparison)
        # For simplicity, we compare the probability distributions directly using Wilcoxon signed-rank test
        # (Are main_probs significantly different from baseline_probs for positive samples?)
        pos_mask = (y_true == 1)
        if np.sum(pos_mask) > 1:
            _, p_wilcoxon = stats.wilcoxon(y_prob_main[pos_mask], y_prob_b[pos_mask])
        else:
            p_wilcoxon = 1.0
            
        test_results[b_name] = {
            "mcnemar_p_value": float(mc_res.pvalue),
            "wilcoxon_pos_p_value": float(p_wilcoxon),
            "significant_mcnemar": bool(mc_res.pvalue < 0.05),
            "significant_wilcoxon": bool(p_wilcoxon < 0.05)
        }
        print(f"  {b_name:25s} | McNemar p={mc_res.pvalue:.4f} | Wilcoxon p={p_wilcoxon:.4f}")
        
    with open(args.output, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nFinal statistical results saved to {args.output}")

if __name__ == "__main__":
    main()
