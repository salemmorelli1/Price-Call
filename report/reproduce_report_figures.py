"""Reproduce the Price-Call report metrics and figures.

Run from the repository root:
    python report/reproduce_report_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "artifacts_part3" / "prediction_log.csv"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def metric_block(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    aligned = pd.concat((actual, predicted), axis=1).dropna()
    y = aligned.iloc[:, 0].astype(float)
    yhat = aligned.iloc[:, 1].astype(float)
    error = yhat - y
    return {
        "n": int(len(aligned)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "bias": float(error.mean()),
        "mape_pct": float((error.abs() / y.abs()).mean() * 100),
        "correlation": float(yhat.corr(y)),
    }


def main() -> None:
    df = pd.read_csv(LOG)
    df["target_date"] = pd.to_datetime(df["target_date"])
    realized = df[df["px_voo_realized"].notna() & df["px_ief_realized"].notna()].copy()

    metrics = {
        "VOO": metric_block(realized["px_voo_realized"], realized["px_voo_call_1d"]),
        "IEF": metric_block(realized["px_ief_realized"], realized["px_ief_call_1d"]),
    }
    prev_voo = realized["px_voo_realized"].shift(1)
    predicted_direction = np.sign(realized["px_voo_call_1d"] - prev_voo)
    realized_direction = np.sign(realized["px_voo_realized"] - prev_voo)
    valid = prev_voo.notna()
    metrics["direction_hit_rate"] = float(
        (predicted_direction[valid] == realized_direction[valid]).mean()
    )
    (OUT / "report_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    colors = {"VOO": ("#d97706", "#0e7490"), "IEF": ("#7c3aed", "#059669")}
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, ticker in zip(axes, ("VOO", "IEF"), strict=True):
        lower = ticker.lower()
        predicted_color, realized_color = colors[ticker]
        ax.plot(df["target_date"], df[f"px_{lower}_call_1d"], label=f"{ticker} predicted",
                color=predicted_color, marker="o", linewidth=2)
        ax.plot(realized["target_date"], realized[f"px_{lower}_realized"],
                label=f"{ticker} realized", color=realized_color, marker="s", linewidth=2)
        ax.set_ylabel(f"{ticker} close ($)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[-1].set_xlabel("Target date")
    fig.tight_layout()
    fig.savefig(OUT / "actual_vs_predicted.png", dpi=220)
    plt.close(fig)

    realized["voo_error"] = realized["px_voo_call_1d"] - realized["px_voo_realized"]
    realized["ief_error"] = realized["px_ief_call_1d"] - realized["px_ief_realized"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(realized["target_date"], realized["voo_error"], color="#d97706", marker="o")
    axes[1].plot(realized["target_date"], realized["ief_error"], color="#0e7490", marker="s")
    for ax, ticker in zip(axes, ("VOO", "IEF"), strict=True):
        ax.axhline(0, color="#64748b", linewidth=1)
        ax.set_ylabel(f"{ticker} error ($)")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Target date")
    fig.tight_layout()
    fig.savefig(OUT / "signed_errors.png", dpi=220)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
