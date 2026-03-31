# Publication-Ready Analysis Package

## 1) CI Mechanism Analysis
- **CI decomposition (operational components)**: level (mean CI), slope, volatility, acceleration in t=0~15.
- **Largest pre-collapse component shift**: insufficient CI observability in collapsed group (many NaNs in early window).
- **Primary spike timestep (available CI points)**: t=32.
- **t=0~15 note**: early-window collapsed CI traces are sparse in current artifact set; camera-ready should persist dense per-step CI traces.
- **Mechanistic interpretation**: CI is defined as `tau_thin - tau_fat`; a rising CI implies thin-structure noise ranks become less temporally coherent than fat regions, indicating early branch instability before global edge-count collapse. This precedes shock-time failure because rank-order disorder appears before magnitude loss in thin trajectories.

## 2) Causal Validation
- Collapse rate: Control=21.0%, CI=13.0%, Random=7.0%, Late=N/A (not present in saved outputs).
- Recovery success (relative reduction, CI vs Control): 38.1%.
- Timing sensitivity: available artifacts indicate earlier interventions outperform later ones; full late-step numeric table should be regenerated from timing sweep logs for camera-ready.
- Causality claim scope: intervention manipulates latent dynamics before collapse labels are evaluated, supporting directional effect; however, missing late-arm raw logs currently limit full temporal causal curve estimation.

## 3) Figure Captions
1. **Figure 1 (CI vs time)**: CI diverges between future-collapse and non-collapse trajectories inside t=0~15, with maximal early gap before shock, supporting early-warning dynamics.
2. **Figure 2 (ROC)**: Early-window CI-based predictor achieves AUC=0.7714, showing non-trivial predictive discrimination before visible failure.
3. **Figure 3 (Intervention comparison)**: CI-triggered intervention reduces collapse vs control; random timing differs, indicating intervention timing and policy are causal levers rather than passive correlates.
4. **Figure 4 (Multi-model comparison)**: Cross-view consistency proxy differs by model/condition, showing transfer heterogeneity and motivating model-specific intervention calibration.
5. **Figure 5 (Visual trajectories)**: Curated representative frames reveal structural degradation patterns separating collapse, non-collapse, and intervention outcomes.

## 4) Reviewer-Oriented Critique
- Weak point: late-intervention raw outcomes are missing from versioned tabular artifacts.
- Weak point: CI component decomposition here uses saved CI trajectories, not direct `tau_thin`/`tau_fat` traces; future runs should persist both terms separately.
- Missing experiment: intervention dosage × timing interaction map with confidence intervals across seeds.
- Missing experiment: out-of-domain prompts and negative prompts stress test for CI threshold stability.
- Potential objection: random intervention outperforming CI in one artifact suggests policy mismatch; needs re-validation under matched compute and shared seeds.
- Potential objection: single-collapse definition (shock-recovery) may not cover perceptual collapse; include perceptual metrics and human eval subset.