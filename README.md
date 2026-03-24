# Price-Call

A production-style Python pipeline for **7-trading-day VOO vs IEF forecasting**, combining:
- relative/tail-risk classification,
- 7-day price-call generation,
- governance and execution controls,
- delayed realized-performance backfill.

The repository is organized so the full prediction cycle can run on a schedule from GitHub Actions or manually from a local shell.

## Repository files

### Core pipeline
- `part1_builder.py` — builds the causal feature, label, and snapshot artifacts.
- `part2_predictor.py` — fits the predictive layer and produces the canonical prediction tape.
- `part2a21_alpha.py` — builds alpha/fusion support artifacts consumed downstream.
- `part3_governance.py` — applies governance and execution logic to the canonical tape.
- `part3_v1_fusion.py` — applies the fusion/allocation layer for the final production outputs.

### Operational runners
- `run_tuesday_prediction.py` — orchestrates the production prediction run.
- `backfill_realized.py` — fills in matured realized prices and error metrics after the 7-day horizon has elapsed.

## What the pipeline does

At a high level, the system:
1. builds features and revealed labels,
2. generates a decision-time probability forecast,
3. applies governance and execution constraints,
4. writes final production artifacts,
5. later backfills realized outcomes for audit and monitoring.

The project is designed around a **7-trading-day horizon** for the VOO-versus-IEF decision problem.

## Expected artifact flow

Typical artifact families produced by the pipeline include:
- `artifacts_part1/`
- `artifacts_part2*/`
- `artifacts_part2a_alpha/`
- `artifacts_part3_v3b/`

The current production-style Part 3 outputs are expected to include files such as:
- `v3c_final_production_tape.csv`
- `v3c_final_production_governance.csv`
- `v3c_fusion_allocations.csv`
- `prediction_log.csv`
- `part3_summary.json`

## Python version

Use **Python 3.12** for GitHub Actions and local reproducibility.

## Installation

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

## Manual runs

### Run the production prediction pipeline

```bash
python run_tuesday_prediction.py --force
```

Use `--force` for manual testing. In scheduled operation, the script can be allowed to respect its normal day/time logic.

### Backfill matured realized results

```bash
python backfill_realized.py
```

This updates the prediction log once enough market history exists to evaluate a prior 7-day call.

## Suggested GitHub repository contents

Upload these source files:
- `backfill_realized.py`
- `part1_builder.py`
- `part2_predictor.py`
- `part2a21_alpha.py`
- `part3_governance.py`
- `part3_v1_fusion.py`
- `run_tuesday_prediction.py`
- `README.md`
- `requirements.txt`
- `.gitignore`
- `.github/workflows/` workflow files

## Recommended `.gitignore` behavior

Generated artifacts generally should **not** be committed unless you explicitly want versioned result snapshots. In most cases, ignore:
- artifact directories,
- caches,
- notebook checkpoints,
- virtual environments,
- OS-specific files.

## Operational notes

- The Tuesday runner is the entry point for the live prediction cycle.
- The backfill script is a separate maintenance step and should run daily.
- Artifact naming should remain stable across all stages so GitHub Actions, local runs, and any monitoring layer point to the same canonical files.

## Status

This repository layout is intended for a clean GitHub deployment of the current working pipeline, without relying on the notebook-export wrapper files.
