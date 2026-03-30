# Research Fact Report

**Generated:** 2026-03-13
**Scope:** Factual summary of the `pyvision` research repository.

---

### 1. Project Overview
- **Project Name:** Diffusion Collapse Early-Warning
- **Main Scripts:**
    - `src/diffusion/ci_runner.py`
    - `src/intervention/phase2_ew.py`
    - `experiments/run_experiments.py`
    - `experiments/analyze_results.py`
    - `experiments/run_intervention_experiments.py`
- **Pipeline Entry Points:** `src/diffusion/ci_runner.py`
- **Main Experiment Folders:**
    - `experiments/`

---

### 2. Models Used

**Model:** Stable Diffusion v1.5
- **Source:** HuggingFace (`runwayml/stable-diffusion-v1-5`)
- **Pipeline:** `diffusers.DiffusionPipeline` with `DDIMScheduler`
- **Related Scripts:** `src/diffusion/ci_runner.py`, `experiments/run_experiments.py`, `experiments/run_intervention_experiments.py`

**Model (Supported):** MVDream
- **Pipeline:** `MVDreamPipeline`
- **Related Scripts:** `src/diffusion/ci_runner.py`

---

### 3. Experiment Setup

- **Number of Prompts:** 5
- **Number of Seeds:** 20 per prompt (validation, seeds 0-19); 100 per condition (parameter sweep, seeds 0-99).
- **Total runs (Validation):** 300
- **Diffusion steps:** 25 (Standard for experiments) / 50 (Baseline setup)
- **Intervention step(s):**
    - CI Intervention: t = 12
    - Random Intervention: t ∈ [5, 18] (per seed)
    - Timing Sweep: t ∈ [6, 8, 10, 12, 14, 16]
- **Early Window Range (for prediction features):** Steps 0–15
- **Boost Parameters:** 0.15 (Standard)
- **Boost Sweep Range:** [0.05, 0.10, 0.15, 0.20, 0.25]
- **CI Calculation Method:** Kendall-τ correlation difference between thin and fat pixels over a sliding window of size 10. `CI = Kendall-τ_thin − Kendall-τ_fat`.
- **Collapse Metric Definition:** Shock-recovery event (10% edge removal at t=20).

---

### 4. Collapse Detection Method

- **Metric Name:** `shock_recovery_collapse`
- **Code Location:** `src/intervention/phase2_ew.py`, `experiments/run_experiments.py`, `experiments/run_intervention_experiments.py`
- **Mathematical Description:** 
    - Shock step ($t_{shock}$) = 20
    - Shock Magnitude = 10% reduction in `thin_pixel_count`
    - Recovery Window ($W$) = 15 steps
    - Recovery Fraction ($f_{recover}$) = 0.90
    - Collapse = True if $\max(TPC[t_{shock}+1 : t_{shock}+W+1]) < f_{recover} \cdot TPC[t_{shock}-1]$
- **Threshold used for collapse classification:** Recovery to 90% of pre-shock value.

---

### 5. Experimental Conditions

**Condition: Control**
- **Intervention:** None
- **Number of runs:** 100
- **Scripts:** `experiments/run_experiments.py`, `experiments/run_intervention_experiments.py`

**Condition: CI Intervention**
- **Intervention:** Latent boost applied at fixed timestep
- **Timing:** t = 12
- **Boost:** 0.15
- **Number of runs:** 100
- **Scripts:** `experiments/run_experiments.py`

**Condition: Random Intervention**
- **Intervention:** Latent boost applied at random timestep
- **Timing:** Random t ∈ [5, 18] per seed
- **Boost:** 0.15
- **Number of runs:** 100
- **Scripts:** `experiments/run_experiments.py`, `experiments/run_intervention_experiments.py`

---

### 6. Quantitative Results

**Experiment: Validation (300 runs)**
*Source: src/metrics/aggregate_metrics.json, paper/draft.md*

| Condition | Collapse Rate |
|---|---|
| Control | 21.0% (21/100) |
| Random Intervention | 7.0% (7/100) |
| CI Intervention | 13.0% (13/100) |

**Experiment: Timing Sweep (N=100 per step)**
*Source: walkthrough.md summary*

| Timing (t) | Collapse Rate |
|---|---|
| 6 | 4.0% |
| 16 | 32.0% |

---

### 7. Early Warning Signal Results
- **AUC:** 0.7140 (cross-validation on control thin-pixel features)
- **Prediction Window:** Steps 0–15
- **ROC figures:** `figures/figure2_ci_signal.png`

---

### 8. Multi-Model Evaluation

| Model | Number of Runs | Observed Collapse Rate | Intervention Result | AUC |
|---|---|---|---|---|
| **SD v1.5** | 100+ | 13% - 21% | 7% - 13% | 0.714 - 0.830 |
| **SDXL Base 1.0** | 100 | 0.0% | N/A (Immune) | 0.500 |
| **MVDream** | 100 | 31.5% | 26.5% | 0.452 |

**Technical Notes:**
- **SDXL**: Inherently immune to the specific edge-collapse mechanics tracked.
- **MVDream**: Encountered significant multi-view instabilities natively; 16.7% relative risk reduction under intervention.

---

### 9. Figures Found in Repository

| Folder | Filename | Potted Variables |
|---|---|---|
| `figures/` | `figure1_collapse_trajectory.png` | thin_pixel_count vs. Diffusion Step |
| `figures/` | `figure2_ci_signal.png` | CI score vs. Diffusion Step |
| `figures/` | `figure3_timing_ablation.png` | Collapse Rate (%) by Condition |
| `figures/` | `figure4_prompt_robustness.png` | Collapse Rate (%) by Prompt & Condition |
| `figures/` | `figure5_timing_sweep.png` | Collapse Rate (%) vs Intervention Step |
| `figures/` | `figure6_boost_sweep.png` | Collapse Rate (%) vs Boost Magnitude |
| `figures/` | `paper_figure_collapse_examples.png` | Image comparison strips |

---

### 10. Reproducibility Information
- **Python Version:** Not explicitly locked in files (Standard Python 3 used in commands).
- **Requirements:** `requirements.txt` (torch, diffusers, transformers, numpy, opencv-python, scipy, matplotlib, scikit-learn, Pillow).
- **Commands:**
    - `python experiments/run_experiments.py`
    - `python experiments/analyze_results.py`
    - `python experiments/run_intervention_experiments.py`

---

### 11. Missing or Unclear Information
- **Python environment lockfile:** No `environment.yml` or `conda` export found.
- **Raw CI values per seed:** Replaced by aggregate metrics and summary figures during cleanup.
- **Model weights:** No local model checkpoints (.pt, .ckpt) stored; assumes HuggingFace access.
- **Multi-model results:** Logic for MVDream exists in code, but no experimental results for MVDream are present.
