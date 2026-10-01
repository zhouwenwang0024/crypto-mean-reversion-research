"""Small, causal entry/exit ablations for the funding relative-value rule.

This module keeps the production candidate untouched.  It reuses its ledger and
adds only entry spread/cost filters, rank persistence, cooldown, and an early
funding-spread reversion exit.  The experiment intentionally fixes the baseline
configuration to lb=21, k=3, lag=3 so that the filters are the only changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from funding_relative_value import (FEE_BP, PERIODS, RESULTS, SYMBOLS,
                                        event_slots, load_funding, load_prices,
                                        market_betas, neutral_weights,
                                        rank_signal)
except ModuleNotFoundError:
    from .funding_relative_value import (FEE_BP, PERIODS, RESULTS, SYMBOLS,
                                         event_slots, load_funding, load_prices,
                                         market_betas, neutral_weights,
                                         rank_signal)

GROSS = 0.90
MAX_SINGLE = 0.25
CFG = {"name": "frv_lb21_k3_lag3", "lookback_events": 21,
       "hold_events": 21, "k": 3, "lag_events": 3}
FEE_RATE = FEE_BP / 10_000.0
COST_COVER_THRESHOLD = 4.0 * FEE_RATE / CFG["hold_events"]


def _drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    return float(np.min(equity / peak - 1.0)) if len(equity) else 0.0


def _signal_stats(rates: np.ndarray, slots: np.ndarray, event_pos: int,
                  lookback: int, lag: int, k: int):
    sig = rank_signal(rates, slots, event_pos, lookback, k, lag)
    if sig is None:
        return None
    longs, shorts = sig
    row_pos = event_pos - lag
    row = np.mean(rates[slots[row_pos - lookback + 1:row_pos + 1]], axis=0)
    spread = float(row[shorts].mean() - row[longs].mean())
    return longs, shorts, spread


def _persistent(rates: np.ndarray, slots: np.ndarray, event_pos: int,
                cfg: dict, p: int) -> bool:
    """Require at least 2/3 membership overlap for each prior signal."""
    if p <= 1:
        return True
    now = _signal_stats(rates, slots, event_pos, cfg["lookback_events"],
                         cfg["lag_events"], cfg["k"])
    if now is None:
        return False
    longs, shorts, _ = now
    for back in range(1, p):
        old = _signal_stats(rates, slots, event_pos - back,
                            cfg["lookback_events"], cfg["lag_events"], cfg["k"])
        if old is None:
            return False
        old_l, old_s, _ = old
        if len(set(longs) & set(old_l)) < max(1, int(np.ceil(cfg["k"] * 2 / 3))):
            return False
        if len(set(shorts) & set(old_s)) < max(1, int(np.ceil(cfg["k"] * 2 / 3))):
            return False
    return True


def run_policy(index, op, close, rates, marks, start: int, end: int,
               policy: dict, fee_bp: float = FEE_BP):
    """Run one filter policy with the same signed cash ledger as the baseline."""
    cfg = CFG; n = close.shape[1]; fee_rate = fee_bp / 10_000.0
    slots = event_slots(rates); pos_by_slot = {int(x): i for i, x in enumerate(slots)}
    q = np.zeros(n); cash = 1.0; total_funding = total_fees = turnover = 0.0
    pending_entry = {}; pending_exit = {}; active = {}
    orders = []; trades = []; bars = []; skipped = 0; max_open = 0.0; cap_bad = 0
    last_exit_event = -10**9

    def emit(t, delta, px, kind):
        nonlocal cash, total_fees, turnover
        fee = float(np.abs(delta * px).sum() * fee_rate)
        cash -= float(delta @ px) + fee
        total_fees += fee; turnover += float(np.abs(delta * px).sum())
        for j in np.flatnonzero(np.abs(delta) > 1e-14):
            orders.append({"time": str(index[t]), "kind": kind,
                           "symbol": SYMBOLS[j], "symbol_index": int(j),
                           "quantity_change": float(delta[j]), "price": float(px[j]),
                           "fee": float(abs(delta[j] * px[j]) * fee_rate)})
        return fee

    for t in range(start, end):
        valid = np.isfinite(rates[t]) & np.isfinite(marks[t])
        if active and valid.any():
            flow = float((-q[valid] * marks[t, valid] * rates[t, valid]).sum())
            cash += flow; total_funding += flow; active["funding"] += flow

        if t in pending_exit and active:
            signal_t, reason = pending_exit.pop(t)
            p = active; pnl = float(np.dot(p["q"], op[t] - p["entry_px"]))
            fee = emit(t, -p["q"], op[t], "exit_" + reason); q -= p["q"]
            trades.append({"entry_time": str(index[p["entry_t"]]), "exit_time": str(index[t]),
                           "gross_pnl": pnl, "funding": p["funding"],
                           "fee": p["entry_fee"] + fee,
                           "net_pnl": pnl + p["funding"] - p["entry_fee"] - fee,
                           "reason": reason, "hold_events": cfg["hold_events"]})
            # The order fills one bar after the completed event.  Record the
            # event immediately preceding that fill so cooldown is measured
            # from the actual exit, rather than from the event that scheduled
            # a 21-event hold.
            last_exit_event = int(np.searchsorted(slots, t, side="left") - 1)
            active = {}; pending_exit.clear()

        if t in pending_entry:
            signal_t, longs, shorts, weights = pending_entry.pop(t)
            if not active:
                equity = cash + float(q @ op[t]); gross = float(np.abs(q * op[t]).sum())
                room = max(0.0, (GROSS * equity - gross) / (1.0 + GROSS * fee_rate))
                if equity > 0 and room > 0:
                    scale = min(equity, room / GROSS); notionals = weights * scale
                    target = np.divide(notionals, op[t], out=np.zeros(n), where=op[t] > 0)
                    fee = emit(t, target, op[t], "entry"); q += target
                    active = {"q": target.copy(), "entry_px": op[t].copy(), "entry_t": t,
                              "entry_fee": fee, "funding": 0.0,
                              "longs": longs, "shorts": shorts}

        event_pos = pos_by_slot.get(t)
        if event_pos is not None and start <= t < end:
            # Match the production engine: the last completed event has no
            # next event and therefore cannot create a new signal/order.
            stats = None
            if event_pos + 1 < len(slots):
                stats = _signal_stats(rates, slots, event_pos, cfg["lookback_events"],
                                      cfg["lag_events"], cfg["k"])
            if active and stats is not None:
                _, _, spread = stats
                revert = policy.get("exit_spread")
                if revert is not None and spread <= revert:
                    fill_t = t + 1
                    if fill_t < end and (not pending_exit or fill_t < min(pending_exit)):
                        pending_exit[fill_t] = (t, "spread_revert")
            if active and not pending_exit:
                due = event_pos + cfg["hold_events"]
                if due < len(slots):
                    pending_exit[int(slots[due]) + 1] = (t, "scheduled")
            if stats is not None:
                longs, shorts, spread = stats
                threshold = float(policy.get("spread_threshold", 0.0))
                good = spread >= threshold
                p = int(policy.get("persistence", 1))
                good = good and _persistent(rates, slots, event_pos, cfg, p)
                cooldown = int(policy.get("cooldown_events", 0))
                good = good and event_pos - last_exit_event > cooldown
                if good:
                    beta = market_betas(close, slots, event_pos)
                    weights, feasible = neutral_weights(beta, longs, shorts, GROSS, MAX_SINGLE)
                    if feasible and weights is not None:
                        fill_t = t + 1
                        if fill_t < end and fill_t not in pending_entry:
                            pending_entry[fill_t] = (t, longs, shorts, weights)
                    else:
                        skipped += 1

        equity_open = cash + float(q @ op[t]); gross_open = float(np.abs(q * op[t]).sum())
        if gross_open > GROSS * max(equity_open, 0.0) and gross_open > 0:
            factor = min(1.0, max(0.0, GROSS * (max(equity_open, 0.0) - gross_open * fee_rate) /
                                   (gross_open * (1.0 - GROSS * fee_rate))))
            delta = q * factor - q; emit(t, delta, op[t], "risk_reduce"); q += delta
            if active: active["q"] *= factor
        post_eq = cash + float(q @ op[t]); post_gross = float(np.abs(q * op[t]).sum())
        if post_eq > 0:
            max_open = max(max_open, post_gross / post_eq)
            cap_bad += int(post_gross > GROSS * post_eq + 1e-10)
        equity = cash + float(q @ close[t])
        bars.append({"time": index[t], "equity": equity, "cash": cash,
                     "funding": total_funding, "fees": total_fees,
                     "gross_exposure": float(np.abs(q * close[t]).sum()),
                     "net_exposure": float(q @ close[t]), "open": int(bool(active))})

    if active:
        t = end - 1; p = active; pnl = float(np.dot(p["q"], close[t] - p["entry_px"]))
        fee = emit(t, -p["q"], close[t], "terminal_exit"); q -= p["q"]
        trades.append({"entry_time": str(index[p["entry_t"]]), "exit_time": str(index[t]),
                       "gross_pnl": pnl, "funding": p["funding"],
                       "fee": p["entry_fee"] + fee,
                       "net_pnl": pnl + p["funding"] - p["entry_fee"] - fee,
                       "reason": "terminal", "hold_events": cfg["hold_events"]})
        bars[-1].update({"equity": cash, "cash": cash, "gross_exposure": 0.0,
                         "net_exposure": 0.0, "open": 0})
    bar = pd.DataFrame(bars); od = pd.DataFrame(orders); tr = pd.DataFrame(trades)
    curve = bar.equity.to_numpy(float)
    days = max((bar.time.iloc[-1] - bar.time.iloc[0]).total_seconds() / 86400.0, 1.0)
    price_pnl = float(curve[-1] - 1.0 - total_funding + total_fees)
    replay = 1.0 - float((od.quantity_change * od.price + od.fee).sum()) if len(od) else 1.0
    replay += total_funding
    row = {"policy": policy["name"], "return": float(curve[-1] - 1.0),
           "cagr": float(curve[-1] ** (365.25 / days) - 1.0) if curve[-1] > 0 else np.nan,
           "mdd": _drawdown(curve), "trades": len(tr), "fees": total_fees,
           "funding": total_funding, "gross_pnl": price_pnl, "turnover": turnover,
           "max_open_gross_ratio": max_open, "cap_open_violations": cap_bad,
           "time_in_market": float(bar.open.mean()), "skipped_infeasible": skipped,
           "replay_error": replay - curve[-1]}
    for name, (a, b) in PERIODS.items():
        ts = pd.to_datetime(bar.time, utc=True)
        mask = (ts >= pd.Timestamp(a, tz="UTC")) & (ts < pd.Timestamp(b, tz="UTC"))
        ids = np.flatnonzero(mask.to_numpy())
        if len(ids):
            base = float(bar.equity.iloc[ids[0] - 1]) if ids[0] else 1.0
            ret = float(bar.equity.iloc[ids[-1]] / base - 1.0)
            period_days = (pd.Timestamp(b, tz="UTC") - pd.Timestamp(a, tz="UTC")).total_seconds() / 86400.0
            row[name + "_return"] = ret
            row[name + "_cagr"] = (1.0 + ret) ** (365.25 / period_days) - 1.0 if ret > -1 else np.nan
        else:
            row[name + "_return"] = np.nan; row[name + "_cagr"] = np.nan
    return row, bar, od, tr


POLICIES = [
    {"name": "baseline"},
    {"name": "spread_cost_1x", "spread_threshold": COST_COVER_THRESHOLD},
    {"name": "spread_20bp", "spread_threshold": 0.00020},
    {"name": "spread_30bp", "spread_threshold": 0.00030},
    {"name": "persistent_2", "persistence": 2},
    {"name": "persistent_3", "persistence": 3},
    {"name": "cost1_persist2", "spread_threshold": COST_COVER_THRESHOLD, "persistence": 2},
    {"name": "cost1_persist3", "spread_threshold": COST_COVER_THRESHOLD, "persistence": 3},
    {"name": "cooldown_1", "cooldown_events": 1},
    {"name": "cooldown_2", "cooldown_events": 2},
    {"name": "revert_exit_5bp", "exit_spread": 0.00005},
    {"name": "revert_exit_cost", "exit_spread": COST_COVER_THRESHOLD},
]


def main():
    index, op, close, _ = load_prices(); rates, marks, _ = load_funding(index)
    start = int(index.searchsorted(pd.Timestamp("2024-06-01", tz="UTC")))
    end = int(index.searchsorted(pd.Timestamp("2026-09-01", tz="UTC")))
    rows = []
    for policy in POLICIES:
        row, bar, od, tr = run_policy(index, op, close, rates, marks, start, end, policy)
        if abs(row["replay_error"]) > 1e-8:
            raise AssertionError(row)
        rows.append(row)
        prefix = "funding_filter_" + policy["name"]
        bar.to_csv(RESULTS / (prefix + "_equity.csv"), index=False)
        od.to_csv(RESULTS / (prefix + "_orders.csv"), index=False)
        tr.to_csv(RESULTS / (prefix + "_trades.csv"), index=False)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "funding_entry_exit_experiments.csv", index=False)
    (RESULTS / "funding_entry_exit_experiments_manifest.json").write_text(
        json.dumps({"baseline": CFG, "fee_bp_one_way": FEE_BP,
                    "cost_cover_threshold": COST_COVER_THRESHOLD,
                    "policies": POLICIES}, indent=2), encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
