# Final Validation Experiment — Report

*Generated: 2026-03-31 (submission refresh)*

## Summary

| Item | Value |
|---|---|
| Diffusion steps | 30 |
| Policies | Always, CI-timing, Random_budget_matched, Late |
| Runs per policy | 400 |
| Total runs | 1600 |
| Random budget | Exactly 1 intervention step/run |
| CI budget | Exactly 1 intervention step/run |
| Late policy | intervention at t >= 20 |

---

## Intervention Results (Fair, Budget-Matched)

| Policy | Collapse Rate |
|---|---|
| Always | **0.0000** |
| CI-timing | **0.9700** |
| Random_budget_matched | **0.9775** |
| Late | **0.9800** |

Ordering check: **Always < CI-timing < Random_budget_matched < Late**

---

## CI Timing Estimator

- CI-only analysis on control trajectories (early window t=0~20)
- `t_peak = argmax(CI_t)` per trajectory
- `avg_peak_t` over collapse trajectories: **3**
- CI intervention timing: one-shot at `t=3`

---

## AUC (Single Source of Truth)

- Method: `sklearn.metrics.roc_auc_score`
- CI traces cleaned via interpolation (`nan_before=1364`, `nan_after=0`)
- **Final AUC: 0.6771**
- Source: `results/auc.json`

---

## Perceptual Correlation

- Proxy metric: `MinSim`
- Correlation: `corr(CI_peak, perceptual_score)`
- **CI_perceptual_correlation: 0.1577**

---

## Output Files

- `results/intervention.json`
- `results/intervention_diagnostics.json`
- `results/submission_summary.json`
- `results/auc.json`
- `figures/publication/fig3_intervention.png`
