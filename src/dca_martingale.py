"""Long-only DCA/martingale basket backtest for the 20-symbol sample."""
from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from research_v2 import END, MONTHS, RESULTS, START, SYMBOLS

BAR_MINUTES = 15
FEE_BP = 5.0
BASE_FRACTION = 0.05
MAX_POSITION_FRACTION = 0.75
VALID_START = pd.Timestamp("2026-05-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-07-01", tz="UTC")


@dataclass(frozen=True)
class Params:
    ma_hours: int
    first_drop_pct: float
    add_step_pct: float
    multiplier: float
    take_profit_pct: float
    stop_loss_pct: float
    max_legs: int


def load_bars() -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    index = pd.date_range(START, END, freq=f"{BAR_MINUTES}min", inclusive="left", tz="UTC")
    fields = {name: np.empty((len(index), len(SYMBOLS))) for name in ("open", "high", "low", "close")}
    for j, symbol in enumerate(SYMBOLS):
        frames = [pd.read_parquet(Path(__file__).parents[1] / "data" / "klines" / f"symbol={symbol}" / f"month={month}.parquet",
                                  columns=["open_time_utc_ms", "open", "high", "low", "close"])
                  for month in MONTHS]
        frame = pd.concat(frames, ignore_index=True)
        frame.index = pd.to_datetime(frame.pop("open_time_utc_ms"), unit="ms", utc=True)
        bars = frame.resample(f"{BAR_MINUTES}min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).reindex(index)
        for name in fields:
            fields[name][:, j] = bars[name].to_numpy(float)
    return index, fields["open"], fields["high"], fields["low"], fields["close"]


def load_funding(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    rates = np.zeros((len(index), len(SYMBOLS)))
    marks = np.full_like(rates, np.nan)
    root = Path(__file__).parents[1] / "results" / "funding_api"
    positions = {t: i for i, t in enumerate(index)}
    for j, symbol in enumerate(SYMBOLS):
        frame = pd.read_parquet(root / f"{symbol}.parquet")
        times = pd.to_datetime(frame.funding_time_utc_ms.astype("int64"), unit="ms", utc=True).dt.floor(f"{BAR_MINUTES}min")
        for t, rate, mark in zip(times, frame.funding_rate.astype(float), frame.mark_price.astype(float)):
            i = positions.get(t)
            if i is not None:
                rates[i, j] = rate
                marks[i, j] = mark
    return rates, marks


@njit(cache=True)
def _simulate(open_, high, low, close, ma, funding_rate, funding_mark,
              first_drop, add_step, multiplier, take_profit, stop_loss,
              max_legs, fee_rate, base_fraction, max_position):
    n = len(close)
    equity = np.empty(n)
    trades = np.zeros((n, 10))
    trade_count = 0
    cash = 1.0
    qty = 0.0
    average = 0.0
    cost_basis = 0.0
    last_entry_price = 0.0
    last_notional = 0.0
    legs = 0
    entry_i = -1
    trade_fees = 0.0
    trade_funding = 0.0
    armed = False
    peak = 0.0
    pending = 0.0

    for i in range(n):
        px = open_[i]
        if qty > 0.0:
            stop = average * (1.0 - stop_loss)
            target = average * (1.0 + take_profit)
            if px <= stop or px >= target:
                exit_px = px
                reason = 1.0 if px <= stop else 2.0
                proceeds = qty * exit_px
                fee = proceeds * fee_rate
                cash += proceeds - fee
                trade_fees += fee
                trades[trade_count, 0] = entry_i
                trades[trade_count, 1] = i
                trades[trade_count, 2] = legs
                trades[trade_count, 3] = cost_basis
                trades[trade_count, 4] = proceeds - cost_basis
                trades[trade_count, 5] = trade_fees
                trades[trade_count, 6] = trade_funding
                trades[trade_count, 7] = trades[trade_count, 4] - trade_fees + trade_funding
                trades[trade_count, 8] = reason
                trades[trade_count, 9] = average
                trade_count += 1
                qty = 0.0; average = 0.0; cost_basis = 0.0; legs = 0; pending = 0.0
                entry_i = -1; trade_fees = 0.0; trade_funding = 0.0; armed = False

        if pending > 0.0 and qty > 0.0:
            available = max(0.0, max_position - qty * px)
            affordable = max(0.0, cash / (1.0 + fee_rate))
            notional = min(pending, available, affordable)
            if notional > 1e-12:
                fee = notional * fee_rate
                cash -= notional + fee
                old_qty = qty
                qty += notional / px
                average = (old_qty * average + notional) / qty
                cost_basis += notional
                last_entry_price = px
                last_notional = notional
                legs += 1
                trade_fees += fee
            pending = 0.0
        elif pending > 0.0 and qty == 0.0:
            affordable = max(0.0, cash / (1.0 + fee_rate))
            notional = min(pending, max_position, affordable)
            if notional > 1e-12:
                fee = notional * fee_rate
                cash -= notional + fee
                qty = notional / px
                average = px
                cost_basis = notional
                last_entry_price = px
                last_notional = notional
                legs = 1
                entry_i = i
                trade_fees = fee
                trade_funding = 0.0
            pending = 0.0

        if qty > 0.0 and funding_rate[i] != 0.0:
            mark = funding_mark[i] if np.isfinite(funding_mark[i]) else close[i]
            flow = -qty * mark * funding_rate[i]
            cash += flow
            trade_funding += flow

        if qty > 0.0:
            stop = average * (1.0 - stop_loss)
            target = average * (1.0 + take_profit)
            exit_px = 0.0
            reason = 0.0
            if low[i] <= stop:
                exit_px = stop
                reason = 1.0
            elif high[i] >= target:
                exit_px = target
                reason = 2.0
            if reason > 0.0:
                proceeds = qty * exit_px
                fee = proceeds * fee_rate
                cash += proceeds - fee
                trade_fees += fee
                trades[trade_count, 0] = entry_i
                trades[trade_count, 1] = i
                trades[trade_count, 2] = legs
                trades[trade_count, 3] = cost_basis
                trades[trade_count, 4] = proceeds - cost_basis
                trades[trade_count, 5] = trade_fees
                trades[trade_count, 6] = trade_funding
                trades[trade_count, 7] = trades[trade_count, 4] - trade_fees + trade_funding
                trades[trade_count, 8] = reason
                trades[trade_count, 9] = average
                trade_count += 1
                qty = 0.0; average = 0.0; cost_basis = 0.0; legs = 0; pending = 0.0
                entry_i = -1; trade_fees = 0.0; trade_funding = 0.0; armed = False

        equity[i] = cash + qty * close[i]
        if i == n - 1:
            if qty > 0.0:
                proceeds = qty * close[i]
                fee = proceeds * fee_rate
                cash += proceeds - fee
                trade_fees += fee
                trades[trade_count, 0] = entry_i
                trades[trade_count, 1] = i
                trades[trade_count, 2] = legs
                trades[trade_count, 3] = cost_basis
                trades[trade_count, 4] = proceeds - cost_basis
                trades[trade_count, 5] = trade_fees
                trades[trade_count, 6] = trade_funding
                trades[trade_count, 7] = trades[trade_count, 4] - trade_fees + trade_funding
                trades[trade_count, 8] = 3.0
                trades[trade_count, 9] = average
                trade_count += 1
                equity[i] = cash
            continue

        if qty == 0.0 and pending == 0.0 and np.isfinite(ma[i]):
            if not armed:
                if close[i] > ma[i]:
                    armed = True
                    peak = close[i]
            else:
                if close[i] > ma[i] and close[i] > peak:
                    peak = close[i]
                if close[i] <= peak * (1.0 - first_drop):
                    pending = base_fraction
                    armed = False
        elif qty > 0.0 and pending == 0.0 and legs < max_legs:
            if close[i] <= last_entry_price * (1.0 - add_step):
                pending = last_notional * multiplier

    return equity, trades, trade_count


def simulate(close, open_, high, low, ma, rates, marks, params: Params):
    return _simulate(open_, high, low, close, ma, rates, marks,
                     params.first_drop_pct, params.add_step_pct, params.multiplier,
                     params.take_profit_pct, params.stop_loss_pct, params.max_legs,
                     FEE_BP / 10000.0, BASE_FRACTION, MAX_POSITION_FRACTION)


def period_metrics(index, equity, trade_array, trade_count, start, end):
    a = index.searchsorted(start)
    b = index.searchsorted(end)
    base = 1.0 if a == 0 else float(equity[a - 1])
    sample = equity[a:b]
    if len(sample) == 0:
        return {"return": np.nan, "max_drawdown": np.nan, "trades": 0, "win_rate": np.nan,
                "profit_factor": np.nan, "fees": 0.0, "funding": 0.0, "legs": np.nan,
                "stops": 0, "take_profits": 0}
    peak = np.maximum.accumulate(np.r_[base, sample])[1:]
    trades = trade_array[:trade_count]
    selected = trades[(trades[:, 1] >= a) & (trades[:, 1] < b)]
    wins = selected[selected[:, 7] > 0, 7].sum()
    losses = selected[selected[:, 7] < 0, 7].sum()
    return {
        "return": float(sample[-1] / base - 1.0),
        "max_drawdown": float(np.min(sample / peak - 1.0)),
        "trades": int(len(selected)),
        "win_rate": float(np.mean(selected[:, 7] > 0)) if len(selected) else np.nan,
        "profit_factor": float(wins / -losses) if losses < 0 else np.nan,
        "fees": float(selected[:, 5].sum()) if len(selected) else 0.0,
        "funding": float(selected[:, 6].sum()) if len(selected) else 0.0,
        "legs": float(selected[:, 2].mean()) if len(selected) else np.nan,
        "stops": int(np.sum(selected[:, 8] == 1.0)),
        "take_profits": int(np.sum(selected[:, 8] == 2.0)),
    }


def grid():
    values = itertools.product((12, 24, 48), (0.02, 0.03, 0.04), (0.01, 0.02),
                                (1.25, 1.5), (0.01, 0.02), (0.06, 0.10), (4, 6))
    return [Params(*v) for v in values]


def evaluate_config(params, index, arrays, rates, marks):
    open_, high, low, close = arrays
    ma_bars = params.ma_hours * 60 // BAR_MINUTES
    rows = []
    for j, symbol in enumerate(SYMBOLS):
        ma = pd.Series(close[:, j], index=index).rolling(ma_bars, min_periods=ma_bars).mean().to_numpy()
        equity, trades, count = simulate(close[:, j], open_[:, j], high[:, j], low[:, j], ma,
                                         rates[:, j], marks[:, j], params)
        valid = period_metrics(index, equity, trades, count, VALID_START, HOLDOUT_START)
        holdout = period_metrics(index, equity, trades, count, HOLDOUT_START, END)
        rows.append({"symbol": symbol, **{f"valid_{k}": v for k, v in valid.items()},
                     **{f"holdout_{k}": v for k, v in holdout.items()}})
    frame = pd.DataFrame(rows)
    valid_returns = frame.valid_return.to_numpy(float)
    valid_dd = frame.valid_max_drawdown.to_numpy(float)
    return frame, {
        "valid_mean_return": float(np.nanmean(valid_returns)),
        "valid_median_return": float(np.nanmedian(valid_returns)),
        "valid_p10_return": float(np.nanpercentile(valid_returns, 10)),
        "valid_worst_return": float(np.nanmin(valid_returns)),
        "valid_median_drawdown": float(np.nanmedian(valid_dd)),
        "valid_worst_drawdown": float(np.nanmin(valid_dd)),
        "valid_positive_coins": int(np.sum(valid_returns > 0)),
        "valid_trades": int(frame.valid_trades.sum()),
        "valid_score": float(0.30 * np.nanmean(valid_returns) + 0.30 * np.nanmedian(valid_returns)
                            + 0.20 * np.nanpercentile(valid_returns, 10)
                            + 0.20 * np.nanmedian(valid_dd)),
        "holdout_mean_return": float(frame.holdout_return.mean()),
        "holdout_median_return": float(frame.holdout_return.median()),
        "holdout_positive_coins": int(np.sum(frame.holdout_return > 0)),
        "holdout_trades": int(frame.holdout_trades.sum()),
    }


def trade_frame(symbol, index, trades, count):
    rows = []
    reasons = {1.0: "stop_loss", 2.0: "take_profit", 3.0: "end_of_sample"}
    for row in trades[:count]:
        rows.append({"symbol": symbol, "entry_time": index[int(row[0])], "exit_time": index[int(row[1])],
                     "legs": int(row[2]), "cost_basis": row[3], "gross_pnl": row[4], "fees": row[5],
                     "funding": row[6], "net_pnl": row[7], "exit_reason": reasons[row[8]],
                     "average_entry": row[9]})
    return rows


def report_safe(grid_table, selected):
    return f"""# Long-only DCA / Martingale backtest

Selected parameters: MA={selected.ma_hours}h, first drop={selected.first_drop_pct:.1%}, add step={selected.add_step_pct:.1%}, multiplier={selected.multiplier:.2f}, take profit={selected.take_profit_pct:.1%}, stop loss={selected.stop_loss_pct:.1%}, max legs={selected.max_legs}.
Validation mean/median/p10 return: {grid_table.iloc[0].valid_mean_return:.2%}/{grid_table.iloc[0].valid_median_return:.2%}/{grid_table.iloc[0].valid_p10_return:.2%}; worst validation return: {grid_table.iloc[0].valid_worst_return:.2%}.
Holdout mean/median return: {grid_table.iloc[0].holdout_mean_return:.2%}/{grid_table.iloc[0].holdout_median_return:.2%}; profitable holdout symbols: {int(grid_table.iloc[0].holdout_positive_coins)}/20.
The configuration is selected on validation only; a negative validation score is not hidden by holdout results.
"""


def run():
    RESULTS.mkdir(exist_ok=True)
    index, open_, high, low, close = load_bars()
    rates, marks = load_funding(index)
    arrays = (open_, high, low, close)
    grid_rows, cache = [], {}
    configs = grid()
    for number, params in enumerate(configs, 1):
        coins, summary = evaluate_config(params, index, arrays, rates, marks)
        row = asdict(params) | summary
        grid_rows.append(row)
        cache[tuple(asdict(params).values())] = coins
        if number % 24 == 0:
            print(f"grid {number}/{len(configs)}", flush=True)
    grid_table = pd.DataFrame(grid_rows)
    eligible = grid_table[(grid_table.valid_trades >= 40) & (grid_table.valid_positive_coins >= 8)]
    ranked = (eligible if len(eligible) else grid_table).sort_values(
        ["valid_score", "valid_median_return", "valid_positive_coins"], ascending=False).reset_index(drop=True)
    grid_table["eligible"] = grid_table.index.isin(eligible.index)
    grid_table.sort_values(["valid_score", "valid_median_return", "valid_positive_coins"], ascending=False).to_csv(
        RESULTS / "dca_martingale_grid.csv", index=False)
    best_row = ranked.iloc[0]
    best = Params(int(best_row.ma_hours), float(best_row.first_drop_pct), float(best_row.add_step_pct),
                  float(best_row.multiplier), float(best_row.take_profit_pct),
                  float(best_row.stop_loss_pct), int(best_row.max_legs))
    best_coins = cache[tuple(asdict(best).values())]

    coin_rows = []
    for _, coin in best_coins.iterrows():
        coin_rows.append({"selection": "global", "config_source": "validation_median", **coin.to_dict(), **asdict(best)})
    for symbol in SYMBOLS:
        coin_candidates = []
        for params in configs:
            key = tuple(asdict(params).values())
            coins = cache[key]
            coin_candidates.append((float(coins.loc[coins.symbol == symbol, "valid_return"].iloc[0]), params, coins.loc[coins.symbol == symbol].iloc[0]))
        _, coin_best, metrics_row = max(coin_candidates, key=lambda x: x[0])
        coin_rows.append({"selection": "per_coin_validation", "config_source": "individual_validation_return",
                          **metrics_row.to_dict(), **asdict(coin_best)})
    coin_table = pd.DataFrame(coin_rows)
    coin_table.to_csv(RESULTS / "dca_martingale_coin_results.csv", index=False)

    trade_rows = []
    equity_table = pd.DataFrame({"time": index})
    portfolio = np.zeros(len(index))
    for j, symbol in enumerate(SYMBOLS):
        ma = pd.Series(close[:, j], index=index).rolling(best.ma_hours * 60 // BAR_MINUTES,
                                                         min_periods=best.ma_hours * 60 // BAR_MINUTES).mean().to_numpy()
        equity, trades, count = simulate(close[:, j], open_[:, j], high[:, j], low[:, j], ma,
                                         rates[:, j], marks[:, j], best)
        portfolio += equity
        trade_rows.extend(trade_frame(symbol, index, trades, count))
    equity_table["equal_weight_equity"] = portfolio / len(SYMBOLS)
    equity_table.to_csv(RESULTS / "dca_martingale_equity.csv", index=False)
    pd.DataFrame(trade_rows).to_csv(RESULTS / "dca_martingale_trades.csv", index=False)
    ranked.head(1).to_json(RESULTS / "dca_martingale_selected.json", orient="records", force_ascii=False, indent=2)
    manifest = {"strategy": "long_only_dca_martingale", "bar_minutes": BAR_MINUTES,
                "symbols": SYMBOLS, "sample": [str(START), str(END)],
                "validation": [str(VALID_START), str(HOLDOUT_START)], "holdout": [str(HOLDOUT_START), str(END)],
                "grid_rows": len(configs), "fee_bp": FEE_BP, "base_fraction": BASE_FRACTION,
                "max_position_fraction": MAX_POSITION_FRACTION, "funding": "local results/funding_api",
                "selection": "0.30 mean return + 0.30 median return + 0.20 p10 return + 0.20 median drawdown, with trade and positive-coin gates"}
    (RESULTS / "dca_martingale_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(__file__).parents[1].joinpath("REPORT_DCA_MARTINGALE_ZH.md").write_text(
        report_safe(ranked, best), encoding="utf-8")
    print(ranked.head(10).to_string(index=False))
    print(coin_table[coin_table.selection == "global"].to_string(index=False))


if __name__ == "__main__":
    run()
