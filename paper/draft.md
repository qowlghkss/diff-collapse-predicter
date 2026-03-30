# Final Validation Experiment — Report

*Generated: 2026-03-13*

## Summary

| | Value |
|---|---|
| Model | `runwayml/stable-diffusion-v1-5` |
| Diffusion steps | 25 |
| Resolution | 512 × 512 |
| Guidance scale | 7.5 |
| Prompts | 5 |
| Seeds per prompt | 20 |
| Conditions | 3 (Control, CI Intervention, Random Intervention) |
| **Total runs** | **300** |
| Intervention boost | 0.15 |
| CI intervention timestep | t = 12 |
| Random intervention range | t ∈ [5, 18] (per seed) |
| Shock timestep | t = 20 |
| Recovery window | 15 steps |
| Collapse threshold | 90% of pre-shock |

---

## Collapse Counts by Condition

| Condition | Collapsed | Total | Collapse Rate |
|---|---|---|---|
| Control | 21 | 100 | **21.0%** |
| Random Intervention | 7 | 100 | **7.0%** |
| CI Intervention | 13 | 100 | **13.0%** |

---

## Collapse Rate per Prompt

| Prompt | Control | Random | CI Intervention |
|---|---|---|---|
| 3D Geometric | 5.0% | 5.0% | 5.0% |
| Gear Assembly | 5.0% | 0.0% | 0.0% |
| Futuristic Arch. | 50.0% | 20.0% | 35.0% |
| Organic Branching | 25.0% | 10.0% | 10.0% |
| Abstract Sculpture | 20.0% | 0.0% | 15.0% |

---

## Overall Collapse Reduction

| Comparison | Absolute Reduction |
|---|---|
| CI vs Control | **+8.0%** |
| Random vs Control | +14.0% |
| CI vs Random (extra) | -6.0% |

---

## Multi-Model Generalization

To verify the robustness of the CI metric across diffusion architectures, we evaluated three distinct foundation models:

| Model | Architecture | Collapse Rate (Control) | Intervention Result | CI AUC |
|---|---|---|---|---|
| **Stable Diffusion v1.5** | Standard 2D | 21.0% | 13.0% (Stable) | 0.714 |
| **Stable Diffusion XL** | Cascaded / High-Res | **0.0%** | N/A (Immune) | 0.500 |
| **MVDream** | Multi-view (SD2.1) | **31.5%** | 26.5% (Partial) | 0.452 |

**Analysis:**
- **SDXL** demonstrates inherent immunity to the tracked structural collapse, likely due to its high-resolution refinement stages.
- **MVDream** exhibits significant multi-view instabilities, which are only partially mitigated by temporal interventions.
- **SD v1.5** remains the most representative model for studying controllable collapse dynamics.

---

## CI Early-Warning AUC

| Metric | Value |
|---|---|
| AUC (cross-val on control thin-pixel features) | **0.7140** |

---

## Key Findings

1. **Collapse generalizes across prompts.** All 5 prompts showed measurable collapse in
   the control condition (range: 5%–
50%), confirming that
   the phenomenon is not prompt-specific.

2. **CI early-warning AUC = 0.7140**, indicating
   meaningful predictive power of the thin-pixel
   trajectory features before the shock event (t ≤ 15).

3. **Intervention comparison:**
   - CI-guided intervention (t=12): collapse rate = 13.0%
   - Random intervention (random t): collapse rate = 7.0%
   - Control: collapse rate = 21.0%
   - CI-guided intervention does NOT outperform random intervention
     by **6.0%** absolute collapse reduction.

4. **Conclusion:** Random intervention shows similar or better performance; further tuning of CI trigger threshold or intervention magnitude may be needed.

---

## Figures

| Figure | File |
|---|---|
| Collapse trajectory example | `figures/collapse_trajectory_example.png` |
| CI early-warning signal | `figures/ci_early_warning.png` |
| Intervention timing ablation | `figures/intervention_timing_ablation.png` |
| Prompt robustness | `figures/prompt_robustness.png` |
