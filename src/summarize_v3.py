"""Create compact diagnostics from the completed v3 run."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
START = pd.Timestamp("2026-03-01", tz="UTC")
TRAIN_END = pd.Timestamp("2026-05-01", tz="UTC")
VALID_END = pd.Timestamp("2026-07-01", tz="UTC")
END = pd.Timestamp("2026-09-01", tz="UTC")
INITIAL = 100_000.0


def load_equity(path: Path) -> pd.Series:
    d = pd.read_csv(path, index_col=0)
    d.index = pd.to_datetime(d.index, utc=True)
    return d.iloc[:, 0].astype(float).sort_index()


def period_stats(eq: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, float, np.ndarray]:
    before = eq[eq.index < start]
    base = float(before.iloc[-1]) if len(before) else INITIAL
    daily = eq[(eq.index >= start) & (eq.index < end)].resample("1D").last().dropna()
    finish = float(eq[eq.index < end].iloc[-1])
    previous = base
    returns = []
    for value in daily.to_numpy(float):
        returns.append(value / previous - 1.0)
        previous = value
    returns = np.asarray(returns, dtype=float)
    if len(returns):
        assert np.isclose(np.prod(1.0 + returns) - 1.0, finish / base - 1.0, atol=1e-10)
    return base, finish, returns


def main() -> None:
    grid = pd.read_csv(RESULTS / "v3_grid_2bp.csv")
    keys = ["model", "frequency_min", "center_hours", "scale_days", "center_kind",
            "logic", "entry_sigma", "exit_sigma", "hold_hours", "exit_rule"]
    model = grid.groupby("model").agg(configs=("model", "size"),
        median_valid=("valid_return", "median"), median_holdout=("holdout_return", "median"),
        mean_valid=("valid_return", "mean"), mean_holdout=("holdout_return", "mean"),
        positive_holdout=("holdout_return", lambda x: int((x > 0).sum())))
    model.to_csv(RESULTS / "v3_model_summary_2bp.csv")
    logic = grid.groupby(["logic", "exit_rule"]).agg(configs=("model", "size"),
        median_valid=("valid_return", "median"), median_holdout=("holdout_return", "median"),
        mean_valid=("valid_return", "mean"), mean_holdout=("holdout_return", "mean"),
        positive_holdout=("holdout_return", lambda x: int((x > 0).sum())))
    logic.to_csv(RESULTS / "v3_logic_summary_2bp.csv")

    rows, target_rows, boot_rows = [], [], []
    chosen = json.loads((RESULTS / "v3_selected_configs.json").read_text(encoding="utf-8"))
    for rank, cfg in enumerate(chosen, 1):
        path = RESULTS / f"v3_selected_{rank}_{cfg['model']}_{int(cfg['frequency_min'])}m_2bp_equity.csv"
        eq = load_equity(path)
        tr_path = path.with_name(path.name.replace("_equity.csv", "_trades.csv"))
        if not tr_path.exists():
            continue
        trades = pd.read_csv(tr_path)
        trades["exit_time"] = pd.to_datetime(trades.exit_time, utc=True)
        name = path.name.replace("_equity.csv", "")
        periods = [("formation", START, TRAIN_END), ("valid", TRAIN_END, VALID_END),
                   ("holdout", VALID_END, END)]
        periods += [(f"2026-{month:02d}", pd.Timestamp(f"2026-{month:02d}-01", tz="UTC"),
                     pd.Timestamp(f"2026-{month + 1:02d}-01", tz="UTC")) for month in range(3, 9)]
        for period, a, b in periods:
            start_equity, end_equity, _ = period_stats(eq, a, b)
            rows.append({"rank": rank, "strategy": name, "period": period,
                         "return": end_equity / start_equity - 1,
                         "start_equity": start_equity, "end_equity": end_equity})
            if period not in {"valid", "holdout"}:
                continue
            tt = trades[(trades.exit_time >= a) & (trades.exit_time < b)]
            if len(tt):
                for target, g in tt.groupby("target"):
                    target_rows.append({"rank": rank, "strategy": name, "period": period,
                        "target": target, "trades": len(g),
                        "net_pnl_including_funding": g.net_pnl_including_funding.sum(),
                        "win_rate": (g.net_pnl_including_funding > 0).mean()})
        holdout_base, holdout_end, daily = period_stats(eq, VALID_END, END)
        if len(daily) >= 10:
            rng = np.random.default_rng(20260924 + rank)
            block = 4
            n_blocks = int(np.ceil(len(daily) / block))
            samples = []
            for _ in range(4000):
                ix = rng.integers(0, len(daily), n_blocks)
                block_ix = (ix[:, None] + np.arange(block)) % len(daily)
                samples.append(float(np.prod(1 + daily[block_ix].ravel()[:len(daily)]) - 1))
            boot_rows.append({"rank": rank, "strategy": name, "period": "holdout",
                "days": len(daily), "return": holdout_end / holdout_base - 1,
                "block_days": block, "bootstrap_ci_2_5": np.percentile(samples, 2.5),
                "bootstrap_ci_97_5": np.percentile(samples, 97.5)})
    pd.DataFrame(rows).to_csv(RESULTS / "v3_selected_month_periods.csv", index=False)
    pd.DataFrame(target_rows).to_csv(RESULTS / "v3_selected_by_target.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(RESULTS / "v3_bootstrap_ci.csv", index=False)


if __name__ == "__main__":
    main()
