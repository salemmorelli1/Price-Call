# Price-Call

A production-style Python pipeline for **7-trading-day VOO vs IEF forecasting**.

The system combines:
- relative / tail-risk classification,
- 7-day price-call generation,
- governance and defense controls,
- alpha / fusion allocation support,
- realized-performance backfill,
- downstream validation, regime, portfolio, execution, and attribution layers.

The repository is organized so the prediction cycle can run from **GitHub Actions** or from a local shell.

## Repository files

### Core production pipeline
- `part0_data_infrastructure.py` — downloads and prepares market and macro data, then writes Part 0 artifacts.
- `part1_builder.py` — builds the causal feature, label, and snapshot artifacts.
- `part2_predictor.py` — fits the predictive layer and writes the canonical Part 2 prediction tape.
- `part2a21_alpha.py` — builds alpha / fusion support artifacts used downstream.
- `part3_governance.py` — applies governance, defense, and fusion logic to produce final production outputs.
- `part4_gui.py` — Gradio-based dashboard for local or notebook-side monitoring.
- `part5_validator.py` — validates file integrity, family selection, and production contract assumptions.
- `part6_regime_engine.py` — builds regime features and regime-state outputs.
- `part7_portfolio_construction.py` — portfolio construction and risk-budgeting layer.
- `part8_execution_model.py` — execution and transaction-cost modeling layer.
- `part9_live_attribution.py` — live attribution and statistical monitoring layer.

### Operational runners
- `run_tuesday_prediction.py` — orchestrates the canonical production run.
- `backfill_realized.py` — fills in matured realized prices and error metrics after the 7-day horizon has elapsed.

### Repo support files
- `requirements.txt` — Python dependency list for local runs and GitHub Actions.
- `.gitignore` — ignore rules for caches, environments, notebook exports, and rebuildable artifacts.
- `.github/workflows/` — GitHub Actions workflows for Tuesday production runs and daily realized backfill.

## What the pipeline does

At a high level, the system:

1. builds market and macro infrastructure artifacts,
2. builds causal features and revealed labels,
3. generates a decision-time probability forecast and 7-day price calls,
4. builds alpha / fusion support artifacts,
5. applies governance and defense logic,
6. validates the production stack,
7. runs regime, portfolio, execution, and attribution layers,
8. later backfills realized outcomes for audit and monitoring.

The project is designed around a **7-trading-day horizon** for the VOO-versus-IEF decision problem.

## Canonical production execution order

The current validated production order is:

```text
Part 0 -> Part 1 -> Part 2 -> Part 2A -> Part 3 -> Part 5 -> Part 6 -> Part 7 -> Part 8 -> Part 9
This updates the prediction log once enough market history exists to evaluate a prior 7-day call.

## GitHub repository contents

Recommended tracked source files:

- `backfill_realized.py`
- `part1_builder.py`
- `part2_predictor.py`
- `part2a21_alpha.py`
- `part3_governance.py`
- `part3_v1_fusion.py`
- `run_tuesday_prediction.py`
- `index.html`
- `README.md`
- `requirements.txt`
- `.gitignore`
- `.github/workflows/` workflow files

## Recommended `.gitignore` behavior

Generated artifacts generally should **not** be committed unless you explicitly want versioned result snapshots.

In most cases, ignore:
- caches,
- notebook checkpoints,
- virtual environments,
- OS-specific files,
- temporary logs and outputs.

## Operational notes

- The Tuesday runner is the entry point for the live prediction cycle.
- The backfill script is a separate daily maintenance step.
- The dashboard should point to the same canonical artifact family as the runner.
- Artifact naming should remain stable across local runs, GitHub Actions, and GitHub Pages monitoring.

## Status

This repository layout reflects the **current working pipeline** and the **currently working V1 production artifact family**.
