"""Small, predeclared experiments motivated by recent stat-arb research.

This is a small, predeclared screen rather than a parameter optimizer.  It
tests short-horizon reversal, a four-hour control, funding-rate reversal, and a
formation-period distance-pair baseline on the existing local minute lake.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from research_v2 import END, MONTHS, RESULTS, START, SYMBOLS, load_funding_events, load_prices

FEE_BPS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)
HORIZONS = (15, 30, 60, 240)
PAIR_Z_BARS = 28 * 24 * 4


def _period_return(values: pd.Series, times: pd.Series, start: str, end: str) -> float:
    t = pd.to_datetime(times, utc=True)
    x = values[(t >= pd.Timestamp(start, tz="UTC")) & (t < pd.Timestamp(end, tz="UTC"))]
    return float((1.0 + x).prod() - 1.0) if len(x) else np.nan


def reversal_events(open1: pd.DataFrame, close1: pd.DataFrame, close15: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight reversal of each completed candle's intra-bar direction."""
    rows = []
    open15 = open1.resample("15min", label="left", closed="left").first().reindex(close15.index)
    r = close15 / open15 - 1.0
    for h in HORIZONS:
        step = max(1, h // 15)
        for t in close15.index[:-step:step]:
            entry_t = t + pd.Timedelta(minutes=15)
            exit_t = entry_t + pd.Timedelta(minutes=h)
            if exit_t >= END or entry_t not in open1.index or exit_t not in close1.index:
                continue
            sig = r.loc[t]
            px0, px1 = open1.loc[entry_t], open1.loc[exit_t]
            fwd = px1 / px0 - 1.0
            valid = sig.notna() & fwd.notna()
            if not valid.any():
                continue
            gross = (-np.sign(sig[valid]) * fwd[valid]).mean()
            magnitude = sig[valid].abs().mean()
            rows.append({"strategy": "sign_reversal", "horizon_min": h, "lookback_min": 15, "signal_time": t, "entry_time": entry_t, "exit_time": exit_t, "n": int(valid.sum()), "gross_return": float(gross), "mean_abs_signal": float(magnitude)})
    return pd.DataFrame(rows)


def funding_reversal(open1: pd.DataFrame, close1: pd.DataFrame, events: list[tuple[pd.Timestamp, np.ndarray, np.ndarray]], q: int = 5) -> pd.DataFrame:
    """Rank the *previous* funding rate, then hold the equal-weight basket 8h.

    Funding is a proxy for basis here.  The rate used for the signal is known at
    the previous settlement; the next settlement's rate is the cash flow paid
    while the position is held.  Entry and exit are delayed to the next 15m
    opens, so the settlement mark itself cannot be traded.
    """
    if len(events) < 3:
        return pd.DataFrame()
    rows = []
    ev = [(t.floor("h"), rate, mark) for t, rate, mark in events]
    for i in range(1, len(ev) - 1):
        t, signal_rate, _ = ev[i]
        signal_rate = ev[i - 1][1]
        next_t, settle_rate, settle_mark = ev[i + 1]
        entry_t = t + pd.Timedelta(minutes=15)
        exit_t = next_t + pd.Timedelta(minutes=15)
        if entry_t not in open1.index or exit_t not in close1.index:
            continue
        s = pd.Series(signal_rate, index=SYMBOLS).replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 2 * q:
            continue
        order = s.sort_values(); low, high = order.index[:q], order.index[-q:]
        fwd = (open1.loc[exit_t] / open1.loc[entry_t] - 1.0).reindex(s.index)
        price = 0.5 * (fwd.loc[low].mean() - fwd.loc[high].mean())
        entry = open1.loc[entry_t]
        mark = pd.Series(settle_mark, index=SYMBOLS)
        rate = pd.Series(settle_rate, index=SYMBOLS)
        funding = 0.5 * ((-mark.loc[low] * rate.loc[low] / entry.loc[low]).mean() + (mark.loc[high] * rate.loc[high] / entry.loc[high]).mean())
        rows.append({"strategy": f"funding_reversal_q{q}", "horizon_min": int((exit_t - entry_t).total_seconds() / 60), "signal_time": t, "entry_time": entry_t, "exit_time": exit_t, "n": len(s), "price_return": float(price), "funding_return": float(funding), "gross_return": float(price + funding), "signal_rate_mean_bp": float(s.mean() * 1e4)})
    return pd.DataFrame(rows)


def cross_sectional_reversal(open1: pd.DataFrame, close1: pd.DataFrame, close15: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    """Long the bottom and short the top of the prior *holding-period* return."""
    rows = []
    for h in HORIZONS:
        step = max(1, h // 15)
        r = close15.pct_change(step)
        for k, t in enumerate(close15.index[:-step:step]):
            entry_t = t + pd.Timedelta(minutes=15); exit_t = entry_t + pd.Timedelta(minutes=h)
            if exit_t >= END or entry_t not in open1.index or exit_t not in close1.index: continue
            sig = r.loc[t].dropna(); px0, px1 = open1.loc[entry_t], open1.loc[exit_t]
            fwd = (px1 / px0 - 1.0).reindex(sig.index).dropna(); sig = sig.reindex(fwd.index)
            if len(sig) < 2 * q: continue
            order = sig.sort_values(); losers, winners = order.index[:q], order.index[-q:]
            gross = 0.5 * (fwd.loc[losers].mean() - fwd.loc[winners].mean())
            rows.append({"strategy": f"cross_sectional_q{q}", "horizon_min": h, "lookback_min": h, "signal_time": t, "entry_time": entry_t, "exit_time": exit_t, "n": len(sig), "gross_return": float(gross), "loser_mean": float(fwd.loc[losers].mean()), "winner_mean": float(fwd.loc[winners].mean())})
    return pd.DataFrame(rows)


def select_distance_pairs(close1: pd.DataFrame) -> pd.DataFrame:
    """Gatev-style formation distance, with the formation fixed before May."""
    h = close1.loc[close1.index < pd.Timestamp("2026-05-01", tz="UTC")].resample("1h").last().dropna(how="any")
    logp = np.log(h); norm = logp - logp.iloc[0]
    rows = []
    for a, b in combinations(SYMBOLS, 2):
        d = float(((norm[a] - norm[b]) ** 2).sum())
        rows.append({"a": a, "b": b, "distance": d})
    return pd.DataFrame(rows).sort_values("distance").head(5).reset_index(drop=True)


def pair_trades(open1: pd.DataFrame, close1: pd.DataFrame, close15: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair_id, pair in pairs.iterrows():
        a, b = pair.a, pair.b
        form = close1.loc[close1.index < pd.Timestamp("2026-05-01", tz="UTC"), [a, b]].resample("1h").last().dropna()
        x, y = np.log(form[b]), np.log(form[a]); beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)); alpha = float(y.mean() - beta * x.mean())
        spread = np.log(close15[a]) - beta * np.log(close15[b]) - alpha
        mean = spread.rolling(PAIR_Z_BARS, min_periods=PAIR_Z_BARS).mean().shift(1); sd = spread.rolling(PAIR_Z_BARS, min_periods=PAIR_Z_BARS).std(ddof=1).shift(1); z = (spread - mean) / sd
        pos = 0; entry = None; entry_px = None; entry_z = None
        for t in close15.index:
            if t < pd.Timestamp("2026-05-01", tz="UTC"):
                continue
            entry_t = t + pd.Timedelta(minutes=15)
            if entry_t >= END or entry_t not in open1.index: continue
            zz = z.loc[t]
            if pos == 0 and np.isfinite(zz) and abs(zz) >= 2:
                pos = -1 if zz > 0 else 1; entry = entry_t; entry_px = open1.loc[entry_t, [a, b]].to_numpy(float); entry_z = float(zz); continue
            if pos and (abs(zz) <= 0.5 or (entry is not None and (entry_t - entry).total_seconds() >= 4 * 3600)):
                exit_t = entry_t; exit_px = open1.loc[exit_t, [a, b]].to_numpy(float); ret = pos * 0.5 * ((exit_px[0] / entry_px[0] - 1) - (exit_px[1] / entry_px[1] - 1))
                rows.append({"strategy": "distance_pair", "pair_id": int(pair_id), "a": a, "b": b, "entry_time": entry, "exit_time": exit_t, "entry_z": entry_z, "gross_return": float(ret)})
                pos = 0; entry = None
        if pos and entry is not None:
            exit_t = close15.index[-1]; exit_px = close1.loc[exit_t, [a, b]].to_numpy(float); ret = pos * 0.5 * ((exit_px[0] / entry_px[0] - 1) - (exit_px[1] / entry_px[1] - 1))
            rows.append({"strategy": "distance_pair", "pair_id": int(pair_id), "a": a, "b": b, "entry_time": entry, "exit_time": exit_t, "entry_z": entry_z, "gross_return": float(ret)})
    return pd.DataFrame(rows)


def summarize(trades: pd.DataFrame, strategy: str, fee_bp: float, horizon_min: int | None = None) -> dict:
    if trades.empty: return {"strategy": strategy, "horizon_min": horizon_min, "fee_bp": fee_bp, "trades": 0}
    cost = 2 * fee_bp / 10000.0
    allocation = 1.0 / trades.pair_id.nunique() if "pair_id" in trades else 1.0
    net = (trades.gross_return - cost) * allocation
    t = pd.to_datetime(trades.get("exit_time", trades.get("signal_time", trades.get("entry_time"))), utc=True)
    order = np.argsort(t.to_numpy()); net = net.iloc[order].reset_index(drop=True); t = t.iloc[order].reset_index(drop=True)
    eq = (1.0 + net).cumprod(); dd = eq / eq.cummax() - 1.0
    return {"strategy": strategy, "horizon_min": horizon_min, "fee_bp": fee_bp, "trades": int(len(net)), "allocation_per_pair": allocation, "mean_price_bp": float(trades.get("price_return", trades.gross_return).mean() * 1e4), "mean_funding_bp": float(trades.get("funding_return", pd.Series(0.0, index=trades.index)).mean() * 1e4), "mean_gross_bp": float(trades.gross_return.mean() * 1e4), "return": float(eq.iloc[-1] - 1), "may_jun_return": _period_return(net, t, "2026-05-01", "2026-07-01"), "jul_aug_return": _period_return(net, t, "2026-07-01", "2026-09-01"), "max_drawdown": float(dd.min()), "win_rate": float((net > 0).mean()), "break_even_one_way_bp": float(trades.gross_return.mean() * 1e4 / 2)}


def block_bootstrap_ci(trades: pd.DataFrame, fee_bp: float, block: int = 24, reps: int = 1000) -> tuple[float, float]:
    """Deterministic block bootstrap for the non-overlapping four-hour control."""
    net = (trades.gross_return.to_numpy(float) - 2 * fee_bp / 10000.0)
    rng = np.random.default_rng(20260925); blocks = [net[i:i + block] for i in range(0, len(net), block)]
    vals = []
    for _ in range(reps):
        sample = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])[:len(net)]
        vals.append(float(np.prod(1.0 + sample) - 1.0))
    return tuple(float(x) for x in np.quantile(vals, [0.025, 0.975]))


def main() -> None:
    open1, close1, _ = load_prices(); close15 = close1.resample("15min", label="left", closed="left").last().dropna(how="any")
    event = reversal_events(open1, close1, close15); cs = cross_sectional_reversal(open1, close1, close15); funding = funding_reversal(open1, close1, load_funding_events()); pairs = select_distance_pairs(close1); pt = pair_trades(open1, close1, close15, pairs)
    event.to_csv(RESULTS / "new_reversal_events.csv", index=False); cs.to_csv(RESULTS / "new_cross_sectional_events.csv", index=False); pairs.to_csv(RESULTS / "new_distance_pairs.csv", index=False); pt.to_csv(RESULTS / "new_distance_pair_trades.csv", index=False)
    funding.to_csv(RESULTS / "new_funding_reversal_events.csv", index=False)
    rows = []
    for h in HORIZONS:
        for fee in FEE_BPS:
            rows.append(summarize(event[event.horizon_min == h], "sign_reversal", fee, h))
            rows.append(summarize(cs[cs.horizon_min == h], "cross_sectional_q5", fee, h))
    for fee in FEE_BPS: rows.append(summarize(funding, "funding_reversal_q5", fee, 480))
    for fee in FEE_BPS: rows.append(summarize(pt, "distance_pair", fee, None))
    out = pd.DataFrame(rows); out.to_csv(RESULTS / "new_strategy_summary.csv", index=False)
    boot = {str(f): block_bootstrap_ci(cs[cs.horizon_min == 240], f) for f in (0.0, 0.5, 1.0)}
    (RESULTS / "new_strategy_bootstrap.json").write_text(json.dumps(boot, indent=2), encoding="utf-8")
    manifest = {"sources": ["https://arxiv.org/abs/2608.21888", "https://github.com/nadav2/short-horizon-reversion", "https://arxiv.org/abs/1903.06033", "https://arxiv.org/abs/2109.10662", "https://ieeexplore.ieee.org/document/9200323/", "https://github.com/Stermer72/pairs-trading", "https://doi.org/10.1002/fut.22425"], "assumptions": {"signal": "completed bar or funding settlement; next 15m bar open", "horizons_min": HORIZONS, "fees": FEE_BPS, "funding": "previous settlement rank; next settlement cash flow, funding is a basis proxy", "pair_formation": "Mar-Apr 2026 fixed before May-June/July-August", "pair_z_bars": PAIR_Z_BARS}}
    (RESULTS / "new_strategy_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out.to_string(index=False)); print("pairs", pairs.to_dict("records"))


if __name__ == "__main__": main()
