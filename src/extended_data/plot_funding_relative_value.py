"""Create reliability charts for the funding-relative-value backtest.

This module only reads the saved backtest outputs.  It does not rerun or alter
the strategy, which keeps the diagnostics separate from parameter selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIMARY = "frv_lb21_k3_lag3"
REVERSED = "frv_lb21_k3_lag3_sign_reversed"
CURVE_CONFIGS = [
    PRIMARY,
    "frv_lb21_k3_lag1",
    "frv_lb63_k3_lag1",
    "frv_lb63_k3_lag3",
    REVERSED,
]


def _read_results(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "funding_relative_value_results.csv"
    frame = pd.read_csv(path)
    numeric = [
        "lookback_events",
        "k",
        "lag_events",
        "return",
        "cagr",
        "mdd",
        "fees",
        "funding",
        "max_gross",
        "max_open_gross_ratio",
        "development_cagr",
        "validation_cagr",
        "historical_holdout_cagr",
        "extension_cagr",
    ]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    return frame


def _read_equity(results_dir: Path, config: str) -> pd.DataFrame:
    path = results_dir / f"funding_relative_value_{config}_equity.csv"
    frame = pd.read_csv(path, parse_dates=["time"])
    frame = frame.sort_values("time").drop_duplicates("time")
    frame["equity"] = pd.to_numeric(frame["equity"], errors="coerce")
    frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1.0
    return frame


def _style(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)


def plot_equity_drawdown(results_dir: Path, output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, height_ratios=[2, 1])
    colors = {PRIMARY: "#1464a0", REVERSED: "#b23a48"}
    labels = {
        PRIMARY: "21-event / k=3 / lag=3",
        "frv_lb21_k3_lag1": "21-event / k=3 / lag=1",
        "frv_lb63_k3_lag1": "63-event / k=3 / lag=1",
        "frv_lb63_k3_lag3": "63-event / k=3 / lag=3",
        REVERSED: "sign-reversed placebo",
    }
    for config in CURVE_CONFIGS:
        curve = _read_equity(results_dir, config)
        color = colors.get(config)
        width = 2.2 if config == PRIMARY else 1.0
        alpha = 0.95 if config in (PRIMARY, REVERSED) else 0.7
        axes[0].plot(curve["time"], curve["equity"], label=labels[config], color=color, lw=width, alpha=alpha)
        axes[1].plot(curve["time"], 100 * curve["drawdown"], color=color, lw=width, alpha=alpha)
    axes[0].axhline(1.0, color="#666", lw=0.8)
    axes[0].set_ylabel("Equity (initial = 1)")
    axes[0].set_title("Funding-relative-value: equity and marked-to-market drawdown")
    axes[0].legend(loc="upper left", ncol=2, frameon=False)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("UTC")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].tick_params(axis="x", rotation=30)
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    path = output_dir / "funding_relative_value_equity_drawdown.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _heatmap(ax: plt.Axes, matrix: pd.DataFrame, title: str, fmt: str) -> None:
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), [str(x) for x in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), [str(x) for x in matrix.index])
    ax.set_xlabel("lag events")
    ax.set_ylabel("lookback events")
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, format(values[i, j], fmt), ha="center", va="center", fontsize=9)
    _style(ax)
    return image


def plot_parameter_heatmap(results: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    images = []
    for col, k in enumerate((2, 3)):
        # The sign-reversed placebo shares the primary parameters and would
        # otherwise create duplicate cells in the parameter pivot.
        subset = results[(results["k"] == k) & (results["config"] != REVERSED)]
        cagr = subset.pivot(index="lookback_events", columns="lag_events", values="cagr") * 100
        mdd = subset.pivot(index="lookback_events", columns="lag_events", values="mdd") * 100
        images.append(_heatmap(axes[0, col], cagr, f"CAGR (%) | k={k}", ".1f"))
        images.append(_heatmap(axes[1, col], mdd, f"MDD (%) | k={k}", ".1f"))
    for ax, image in zip(axes.flat, images):
        fig.colorbar(image, ax=ax, shrink=0.82, pad=0.03)
    fig.suptitle("Parameter sensitivity at fixed one-way fee = 5 bp", y=1.02)
    path = output_dir / "funding_relative_value_parameter_heatmap.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_phase_cagr(results: pd.DataFrame, output_dir: Path) -> Path:
    phases = ["development_cagr", "validation_cagr", "historical_holdout_cagr", "extension_cagr"]
    phase_labels = ["development", "validation", "holdout", "extension"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for _, row in results.iterrows():
        values = 100 * row[phases].to_numpy(dtype=float)
        highlighted = row["config"] == PRIMARY
        reversed_ = row["config"] == REVERSED
        ax.plot(
            phase_labels,
            values,
            marker="o",
            lw=2.5 if highlighted else 1.0,
            alpha=1.0 if highlighted or reversed_ else 0.55,
            color="#1464a0" if highlighted else ("#b23a48" if reversed_ else None),
            label=row["config"] if highlighted or reversed_ else None,
        )
    ax.axhline(0, color="#555", lw=0.8)
    ax.axhline(10, color="#b8860b", lw=0.8, ls="--", label="10% target")
    ax.set_ylabel("Annualized CAGR (%)")
    ax.set_title("CAGR by chronological period (all tested configurations)")
    ax.legend(frameon=False, loc="best")
    _style(ax)
    fig.tight_layout()
    path = output_dir / "funding_relative_value_phase_cagr.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_attribution(results_dir: Path, output_dir: Path) -> Path:
    curve = _read_equity(results_dir, PRIMARY)
    daily = curve.set_index("time").resample("1D").last().dropna(subset=["equity"])
    price_pnl = daily.equity - 1.0 - daily.funding + daily.fees
    rolling = (daily.equity / daily.equity.shift(90)).pow(365.25 / 90.0) - 1.0
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(daily.index, price_pnl, label="price PnL", lw=1.2)
    axes[0].plot(daily.index, daily.funding, label="funding", lw=1.1)
    axes[0].plot(daily.index, -daily.fees, label="-fees", lw=1.1)
    axes[0].axhline(0, color="#555", lw=0.8)
    axes[0].set_ylabel("Cumulative PnL")
    axes[0].set_title("Primary account attribution")
    axes[0].legend(frameon=False)
    axes[1].plot(daily.index, 100 * rolling, color="#1464a0", lw=1.2)
    axes[1].axhline(10, color="#b8860b", ls="--", lw=0.8, label="10% target")
    axes[1].axhline(0, color="#555", lw=0.8)
    axes[1].set_ylabel("90-day CAGR (%)")
    axes[1].set_xlabel("UTC")
    axes[1].legend(frameon=False)
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    path = output_dir / "funding_relative_value_attribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_fee_sensitivity(results_dir: Path, output_dir: Path) -> Path:
    frame = pd.read_csv(results_dir / "funding_relative_value_fee_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frame.fee_bp_one_way, 100 * frame.cagr, marker="o", label="CAGR")
    ax.plot(frame.fee_bp_one_way, 100 * frame.mdd, marker="o", label="MDD")
    ax.axhline(10, color="#b8860b", ls="--", lw=0.8, label="10% CAGR target")
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xlabel("One-way fee (bp)")
    ax.set_ylabel("Percent")
    ax.set_title("Primary fee sensitivity")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    path = output_dir / "funding_relative_value_fee_sensitivity.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _write_summary(results: pd.DataFrame, output_dir: Path) -> Path:
    primary = results.loc[results["config"] == PRIMARY].iloc[0]
    phase = primary[["development_cagr", "validation_cagr", "historical_holdout_cagr", "extension_cagr"]]
    summary = pd.DataFrame(
        {
            "metric": [
                "primary_full_sample_cagr",
                "primary_mdd",
                "primary_fees",
                "primary_funding",
                "primary_max_marked_gross",
                "primary_max_open_gross_ratio",
                "phase_cagr_min",
                "phase_cagr_max",
                "configs_above_10pct_cagr",
            ],
            "value": [
                primary["cagr"],
                primary["mdd"],
                primary["fees"],
                primary["funding"],
                primary["max_gross"],
                primary["max_open_gross_ratio"],
                phase.min(),
                phase.max(),
                int((results["cagr"] >= 0.10).sum()),
            ],
        }
    )
    path = output_dir / "funding_relative_value_reliability_summary.csv"
    summary.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = _read_results(results_dir)
    paths = [
        plot_equity_drawdown(results_dir, output_dir),
        plot_parameter_heatmap(results, output_dir),
        plot_phase_cagr(results, output_dir),
        plot_attribution(results_dir, output_dir),
        plot_fee_sensitivity(results_dir, output_dir),
        _write_summary(results, output_dir),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
