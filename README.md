# Diffusion Collapse Early-Warning

Research code for detecting and preventing structural collapse in 3D diffusion models.

## Structure
- `src/`: Core implementation (diffusion tracking, interventions, metrics)
- `experiments/`: Scripts to reproduce paper results
- `figures/`: Final publication figures
- `paper/`: Research report and analysis

## Setup
```bash
pip install -r requirements.txt
```

## Reproducing Results
```bash
python experiments/run_experiments.py
python experiments/analyze_results.py
```

## Digital Hygiene (KISTI Silver-Aligned)
- Date of cleanup: `2026-03-31`
- Orphaned figure assets were moved from `figures/` to `archived/figures_orphaned_2026-03-31/` (19 files).
- Runtime cache and temporary artifacts were removed (`__pycache__/`, `*.pyc`, debug logs, packaged duplicate figure archive).
- `figures/` now keeps only publication-referenced images used by main reports.
- Long-term research data in `data/` remains preserved per retention policy.
