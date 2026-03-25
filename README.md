Price-Call
A production-style Python pipeline for 7-trading-day VOO vs IEF forecasting.
The system combines:
relative / tail-risk classification,
7-day price-call generation,
governance and execution controls,
alpha / fusion allocation support,
delayed realized-performance backfill.
The repository is organized so the prediction cycle can run from GitHub Actions or from a local shell.
Repository files
Core pipeline
`part1_builder.py` — builds the causal feature, label, and snapshot artifacts.
`part2_predictor.py` — fits the predictive layer and writes the canonical prediction tape.
`part2a21_alpha.py` — builds alpha / fusion support artifacts used downstream.
`part3_governance.py` — applies governance and execution logic to the prediction tape.
`part3_v1_fusion.py` — applies the fusion / allocation layer for final production outputs.
Operational runners
`run_tuesday_prediction.py` — orchestrates the Tuesday production prediction run.
`backfill_realized.py` — fills in matured realized prices and error metrics after the 7-day horizon has elapsed.
Repo support files
`index.html` — GitHub Pages dashboard for monitoring forecasts, governance, fusion allocations, and realized backfill.
`requirements.txt` — Python dependency list for local runs and GitHub Actions.
`.gitignore` — recommended ignore rules for Python, caches, environments, and generated artifacts.
`.github/workflows/` — GitHub Actions workflows for Tuesday prediction, daily backfill, and Pages deployment.
What the pipeline does
At a high level, the system:
builds causal features and revealed labels,
generates a decision-time probability forecast,
builds alpha / fusion support artifacts,
applies governance and fusion logic,
writes final production artifacts,
later backfills realized outcomes for audit and monitoring.
The project is designed around a 7-trading-day horizon for the VOO-versus-IEF decision problem.
Current canonical artifact flow
Typical artifact families produced by the pipeline include:
`artifacts_part1/`
`artifacts_part2_v77/`
`artifacts_part2a_alpha/`
`artifacts_part3/`
`artifacts_part3_v1/`
The current production Part 3 outputs are:
`artifacts_part3/prediction_log.csv`
`artifacts_part3_v1/v1_final_production_tape.csv`
`artifacts_part3_v1/v1_final_production_governance.csv`
`artifacts_part3_v1/v1_fusion_allocations.csv`
These are the files the current runner and dashboard are expected to use.
Python version
Use Python 3.12 for GitHub Actions and local reproducibility.
Installation
Create and activate a virtual environment, then install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
On Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```
Manual runs
Run the production prediction pipeline
```bash
python run_tuesday_prediction.py --force
```
Use `--force` for manual testing outside the normal Tuesday decision window.
Backfill matured realized results
```bash
python backfill_realized.py
```
This updates the prediction log once enough market history exists to evaluate a prior 7-day call.
GitHub repository contents
Recommended tracked source files:
`backfill_realized.py`
`part1_builder.py`
`part2_predictor.py`
`part2a21_alpha.py`
`part3_governance.py`
`part3_v1_fusion.py`
`run_tuesday_prediction.py`
`index.html`
`README.md`
`requirements.txt`
`.gitignore`
`.github/workflows/` workflow files
Recommended `.gitignore` behavior
Generated artifacts generally should not be committed unless you explicitly want versioned result snapshots.
In most cases, ignore:
caches,
notebook checkpoints,
virtual environments,
OS-specific files,
temporary logs and outputs.
Operational notes
The Tuesday runner is the entry point for the live prediction cycle.
The backfill script is a separate daily maintenance step.
The dashboard should point to the same canonical artifact family as the runner.
Artifact naming should remain stable across local runs, GitHub Actions, and GitHub Pages monitoring.
Status
This repository layout reflects the current working pipeline and the currently working V1 production artifact family.
