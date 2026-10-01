"""Plot the baseline and the selected funding entry filter.

The script only reads saved outputs.  It is deliberately separate from the
backtest so charting cannot change the experiment or select a policy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


POLICIES = {
    "baseline": "funding_filter_baseline_equity.csv",
    "cost1_persist3": "funding_filter_cost1_persist3_equity.csv",
}
LABELS = {
    "baseline": "baseline",
    "cost1_persist3": "spread-cost + 3-event persistence",
}
COLORS = {"baseline": "#6c757d", "cost1_persist3": "#1464a0"}
PHASES = [
    ("development_cagr", "development"),
    ("validation_cagr", "validation"),
    ("historical_holdout_cagr", "holdout"),
    ("extension_cagr", "extension"),
]


def _style(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)


def _equity(results_dir: Path, policy: str) -> pd.DataFrame:
    frame = pd.read_csv(results_dir / POLICIES[policy], parse_dates=["time"])
    frame = frame.sort_values("time").drop_duplicates("time")
    frame["equity"] = pd.to_numeric(frame["equity"], errors="coerce")
    frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1.0
    return frame.dropna(subset=["time", "equity"])


def plot_equity_drawdown(results_dir: Path, output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, height_ratios=[2, 1])
    for policy in POLICIES:
        frame = _equity(results_dir, policy)
        style = {"lw": 2.2 if policy == "cost1_persist3" else 1.3, "color": COLORS[policy]}
        axes[0].plot(frame["time"], frame["equity"], label=LABELS[policy], **style)
        axes[1].plot(frame["time"], 100 * frame["drawdown"], label=LABELS[policy], **style)
    axes[0].axhline(1.0, color="#555", lw=0.8)
    axes[0].set_ylabel("Equity (initial = 1)")
    axes[0].set_title("Funding entry filter: equity and marked-to-market drawdown")
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].axhline(0, color="#555", lw=0.8)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("UTC")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].tick_params(axis="x", rotation=30)
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    path = output_dir / "funding_entry_exit_equity_drawdown.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _read_experiments(results_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(results_dir / "funding_entry_exit_experiments.csv")
    return frame.set_index("policy")


def plot_phase_comparison(results_dir: Path, output_dir: Path) -> Path:
    frame = _read_experiments(results_dir).loc[list(POLICIES)]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = list(range(len(PHASES)))
    width = 0.36
    for offset, policy in zip((-width / 2, width / 2), POLICIES):
        values = [100 * frame.loc[policy, column] for column, _ in PHASES]
        ax.bar([i + offset for i in x], values, width, label=LABELS[policy], color=COLORS[policy])
    ax.axhline(0, color="#555", lw=0.8)
    ax.axhline(10, color="#b8860b", ls="--", lw=0.8, label="10% target")
    ax.set_xticks(x, [label for _, label in PHASES])
    ax.set_ylabel("Annualized CAGR (%)")
    ax.set_title("CAGR by chronological phase")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    path = output_dir / "funding_entry_exit_phase_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_fee_sensitivity(results_dir: Path, output_dir: Path) -> Path:
    frame = pd.read_csv(results_dir / "funding_entry_exit_cost1_persist3_fee_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(frame["fee_bp_one_way"], 100 * frame["cagr"], marker="o", lw=2, label="CAGR")
    ax.plot(frame["fee_bp_one_way"], 100 * frame["mdd"], marker="o", lw=1.4, label="MDD")
    ax.axhline(10, color="#b8860b", ls="--", lw=0.8, label="10% target")
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xlabel("One-way fee (bp)")
    ax.set_ylabel("Percent")
    ax.set_title("Cost1 + persistence candidate: fee sensitivity")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    path = output_dir / "funding_entry_exit_fee_sensitivity.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_summary(results_dir: Path, output_dir: Path) -> Path:
    frame = _read_experiments(results_dir).loc[list(POLICIES)]
    rows = []
    for policy in POLICIES:
        for column, phase in PHASES:
            rows.append({"policy": policy, "metric": f"{phase}_cagr", "value": frame.loc[policy, column]})
        rows.extend(
            {
                "policy": policy,
                "metric": metric,
                "value": frame.loc[policy, metric],
            }
            for metric in ("cagr", "mdd", "trades", "fees", "turnover", "time_in_market")
        )
    path = output_dir / "funding_entry_exit_chart_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_equity_drawdown(results_dir, output_dir),
        plot_phase_comparison(results_dir, output_dir),
        plot_fee_sensitivity(results_dir, output_dir),
        write_summary(results_dir, output_dir),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
