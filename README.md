# Diffusion Collapse Early-Warning

This repository provides a minimal reproducible pipeline for detecting and preventing structural collapse in diffusion-based 3D generation.

---

## 🔧 Setup

```bash
pip install -r requirements.txt

Step 1: Run intervention experiment

python scripts/run_intervention_ablation.py
```

This generates:

results/intervention.json

Step 2: Generate figures
python scripts/make_figures.py

This produces:

figures/publication/fig3_intervention.png
 Main Result

Figure 3 compares different intervention strategies:

Always: continuous intervention (upper bound)
CI-based: proposed method
Random: timing-agnostic baseline
Late: delayed intervention

 Repository Structure
```
configs/        Configuration files
data/           Minimal data (or placeholder)
results/        Experiment outputs
figures/        Publication figures
scripts/        Reproducible experiment scripts
paper/          Draft and analysis
src/            Core implementation (not required for reproduction)
```

 Notes
This repo is paper-oriented and minimal.
Some large-scale experiments and exploratory scripts have been removed for clarity.
The core claim can be reproduced using only the provided scripts.
