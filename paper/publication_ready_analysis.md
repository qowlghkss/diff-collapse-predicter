# Publication-Ready Analysis Package

## 1) CI Mechanism Analysis (Timing Estimator)
- CI is used as a **timing estimator**, not a threshold-fire policy.
- For each control trajectory, we compute `t_peak = argmax(CI_t)` in early window `t=0~20` after NaN interpolation.
- Collapse-trajectory average peak timing:
  - **avg_peak_t = 3**
- CI-timing intervention policy:
  - one-step intervention at `t=avg_peak_t` per run.

## 2) Causal Validation (Fair, Budget-Matched)
- Shared seeds and same run count across policies (`N=400` each).
- Intervention budgets:
  - CI-timing: exactly 1 step/run
  - Random_budget_matched: exactly 1 step/run (`random.randint(0, T-1)`)
- Results:
  - **Always**: 0.0000
  - **CI-timing**: 0.9700
  - **Random_budget_matched**: 0.9775
  - **Late**: 0.9800
- Ordering check:
  - **Always < CI-timing < Random_budget_matched < Late** (satisfied)

## 3) NaN Fix + AUC Recompute (Single Source)
- CI traces cleaned by interpolation (linear + `nan_to_num`).
- NaN counts:
  - before: **1364**
  - after: **0**
- Single-source AUC (`sklearn.metrics.roc_auc_score`):
  - **Final AUC = 0.6771**
- Source file:
  - `results/auc.json`

## 4) Perceptual Correlation Check
- Perceptual proxy: `MinSim` from control entries.
- Correlation:
  - **CI_perceptual_correlation = 0.1577**

## 5) Figure Update
- Regenerated with updated intervention keys and ordering:
  - `figures/publication/fig3_intervention.png`
- Labels:
  - Always, CI-timing, Random_budget_matched, Late

## 6) Reproducibility Outputs
- `results/intervention.json`
- `results/intervention_diagnostics.json`
- `results/submission_summary.json`
- `results/auc.json`
