"""Funding-conditioned cross-sectional relative value.

The signal is deliberately small and causal: at each completed funding event,
rank the *previous* funding event, buy the lowest-rate contracts and short the
highest-rate contracts.  The resulting basket is dollar neutral and, when the
selected beta ranges overlap, neutral to a rolling equal-weight market beta.
Orders are filled at the next five-minute open and funding is settled before
that fill.  This is a research candidate, not an option premium strategy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from validate_5m_mean_reversion import (FEE_BP, PERIODS, RESULTS, SYMBOLS,
                                              load_funding, load_prices)
except ModuleNotFoundError:  # package import from repository root
    from .validate_5m_mean_reversion import (FEE_BP, PERIODS, RESULTS, SYMBOLS,
                                              load_funding, load_prices)

GROSS_FRACTION = 0.90
MAX_SINGLE_FRACTION = 0.25
BETA_LOOKBACK_EVENTS = 63
PRIMARY_LOOKBACK_EVENTS = 21
PRIMARY_HOLD_EVENTS = 21
PRIMARY_K = 3
PRIMARY_LAG_EVENTS = 3
CONFIGS = tuple(
    {"name": f"frv_lb{lb}_k{k}_lag{lag}", "lookback_events": lb,
     "hold_events": 21, "k": k, "lag_events": lag}
    for lb in (21, 63) for k in (2, 3) for lag in (1, 3)
)


def _drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    return float(np.min(equity / peak - 1.0)) if len(equity) else 0.0


def event_slots(rates: np.ndarray) -> np.ndarray:
    """Return the common completed funding-event bars."""
    x = np.asarray(rates, float)
    if x.ndim != 2:
        raise ValueError("rates must be two-dimensional")
    return np.flatnonzero(np.isfinite(x).all(axis=1))


def market_betas(close: np.ndarray, slots: np.ndarray, event_pos: int,
                 lookback_events: int = BETA_LOOKBACK_EVENTS) -> np.ndarray:
    """Causal beta to the equal-weight market, through the prior event close."""
    if event_pos < lookback_events:
        return np.ones(close.shape[1])
    end = event_pos  # excludes the current event close
    start = end - lookback_events
    logp = np.log(close[slots[start:end]])
    rets = np.diff(logp, axis=0)
    market = np.mean(rets, axis=1)
    den = float(market @ market)
    if den <= 1e-14 or not np.isfinite(den):
        return np.ones(close.shape[1])
    centered = market - market.mean()
    var = float(centered @ centered)
    if var <= 1e-14:
        return np.ones(close.shape[1])
    # market mean is removed from both series; the intercept is irrelevant to
    # the hedge ratio and this remains well-defined for all symbols.
    centered_assets = rets - rets.mean(axis=0)
    beta = (centered[:, None] * centered_assets).sum(axis=0) / var
    return np.where(np.isfinite(beta), beta, 1.0)


def _side_alloc(beta: np.ndarray, target: float, max_single: float) -> np.ndarray | None:
    """Closest-to-equal nonnegative allocations with a requested beta mean."""
    n = len(beta)
    if n == 0 or target < beta.min() - 1e-10 or target > beta.max() + 1e-10:
        return None
    if n == 1:
        return np.ones(1) if abs(target - beta[0]) <= 1e-8 else None
    b = beta.astype(float); mean = float(b.mean()); d = b - mean
    ss = float(d @ d)
    if ss <= 1e-14:
        return np.full(n, 1.0 / n) if abs(target - mean) <= 1e-8 else None
    # Minimum squared deviation from equal weights subject to sum=1 and
    # beta-weighted mean=target.  A small cap check prevents concentration.
    alloc = np.full(n, 1.0 / n) + ((target - mean) / ss) * d
    if np.min(alloc) < -1e-9:
        return None
    alloc = np.maximum(alloc, 0.0); alloc /= alloc.sum()
    return alloc if float(alloc.max()) <= max_single + 1e-9 else None


def neutral_weights(beta: np.ndarray, long_ids: np.ndarray,
                    short_ids: np.ndarray, gross: float = GROSS_FRACTION,
                    max_single: float = MAX_SINGLE_FRACTION) -> tuple[np.ndarray | None, bool]:
    """Build signed dollar weights; return ``(weights, feasible)``.

    Each side carries half of gross.  The common beta target is chosen from the
    overlap of the two side beta ranges, closest to their equal-weight means.
    """
    beta = np.asarray(beta, float)
    if len(long_ids) == 0 or len(short_ids) == 0:
        return None, False
    lo = max(float(beta[long_ids].min()), float(beta[short_ids].min()))
    hi = min(float(beta[long_ids].max()), float(beta[short_ids].max()))
    if lo > hi + 1e-10:
        return None, False
    preferred = (float(beta[long_ids].mean()) + float(beta[short_ids].mean())) / 2.0
    target = float(np.clip(preferred, lo, hi))
    side_cap = max_single / (gross / 2.0)
    la = _side_alloc(beta[long_ids], target, side_cap)
    sa = _side_alloc(beta[short_ids], target, side_cap)
    if la is None or sa is None:
        return None, False
    w = np.zeros(len(beta)); half = gross / 2.0
    w[long_ids] = half * la; w[short_ids] = -half * sa
    if abs(w.sum()) > 1e-10 or abs(float(w @ beta)) > 1e-7 or np.max(np.abs(w)) > max_single + 1e-9:
        return None, False
    return w, True


def rank_signal(rates: np.ndarray, slots: np.ndarray, event_pos: int,
                lookback_events: int, k: int, lag_events: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Rank the lagged rolling mean funding rate cross-sectionally."""
    rank_pos = event_pos - lag_events
    if (rank_pos < lookback_events - 1 or k < 1 or
            2 * k > rates.shape[1]):
        return None
    row = np.mean(rates[slots[rank_pos - lookback_events + 1:rank_pos + 1]], axis=0)
    if not np.isfinite(row).all():
        return None
    order = np.argsort(row, kind="mergesort")
    return order[:k], order[-k:][::-1]


def _period_summary(bar: pd.DataFrame, start: str, end: str) -> float:
    t = pd.to_datetime(bar.time, utc=True)
    mask = (t >= pd.Timestamp(start, tz="UTC")) & (t < pd.Timestamp(end, tz="UTC"))
    if not mask.any():
        return np.nan
    ids = np.flatnonzero(mask.to_numpy()); before = float(bar.equity.iloc[ids[0] - 1]) if ids[0] else 1.0
    return float(bar.equity.iloc[ids[-1]] / before - 1.0)


def _period_cagr(bar: pd.DataFrame, start: str, end: str) -> float:
    ret = _period_summary(bar, start, end)
    if not np.isfinite(ret) or ret <= -1.0:
        return np.nan
    days = (pd.Timestamp(end, tz="UTC") - pd.Timestamp(start, tz="UTC")).total_seconds() / 86400.0
    return float((1.0 + ret) ** (365.25 / days) - 1.0)


def run(index: pd.DatetimeIndex, op: np.ndarray, close: np.ndarray,
        rates: np.ndarray, marks: np.ndarray, cfg: dict, start: int, end: int,
        *, fee_bp: float = FEE_BP, gross_fraction: float = GROSS_FRACTION,
        max_single_fraction: float = MAX_SINGLE_FRACTION) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run one fixed configuration and return summary, bars, orders, trades."""
    n = close.shape[1]; fee_rate = fee_bp / 10_000.0
    slots = event_slots(rates); pos_by_slot = {int(x): i for i, x in enumerate(slots)}
    q = np.zeros(n); cash = 1.0; total_funding = total_fees = turnover = 0.0
    pending_entry: dict[int, tuple[int, np.ndarray, np.ndarray, np.ndarray]] = {}
    pending_exit: dict[int, tuple[int, str]] = {}
    active: dict = {}; orders: list[dict] = []; trades: list[dict] = []; bars: list[dict] = []
    skipped_infeasible = 0; max_open_ratio = 0.0; cap_open_violations = 0

    def emit(t: int, delta: np.ndarray, px: np.ndarray, kind: str) -> float:
        nonlocal cash, total_fees, turnover
        fee = float(np.abs(delta * px).sum() * fee_rate)
        # Linear futures cash ledger: every signed notional changes cash;
        # replaying these rows plus funding reproduces the terminal equity.
        cash -= float(delta @ px) + fee
        total_fees += fee; turnover += float(np.abs(delta * px).sum())
        for j in np.flatnonzero(np.abs(delta) > 1e-14):
            orders.append({"time": str(index[t]), "kind": kind, "symbol": SYMBOLS[j],
                           "symbol_index": int(j), "quantity_change": float(delta[j]),
                           "price": float(px[j]), "fee": float(abs(delta[j] * px[j]) * fee_rate)})
        return fee

    for t in range(start, end):
        valid = np.isfinite(rates[t]) & np.isfinite(marks[t])
        if active and valid.any():
            flow = float((-q[valid] * marks[t, valid] * rates[t, valid]).sum())
            cash += flow; total_funding += flow; active["funding"] += flow

        # Signals generated at completed event closes are filled at this bar's
        # open.  Exit precedes entry when both are scheduled together.
        if t in pending_exit and active:
            signal_t, reason = pending_exit.pop(t)
            p = active; pnl = float(np.dot(p["q"], op[t] - p["entry_px"]))
            delta = -p["q"]; fee = emit(t, delta, op[t], "exit_" + reason)
            q += delta
            trades.append({"entry_time": str(index[p["entry_t"]]), "exit_time": str(index[t]),
                           "gross_pnl": pnl, "funding": p["funding"], "fee": p["entry_fee"] + fee,
                           "net_pnl": pnl + p["funding"] - p["entry_fee"] - fee,
                           "reason": reason, "hold_events": int(p["hold_events"])})
            active = {}
        if t in pending_entry:
            signal_t, ids_long, ids_short, weights = pending_entry.pop(t)
            if not active:
                equity = cash + float(q @ op[t]); gross = float(np.abs(q * op[t]).sum())
                room = max(0.0, (gross_fraction * equity - gross) / (1.0 + gross_fraction * fee_rate))
                if equity > 0 and room > 0:
                    scale = min(equity, room / gross_fraction)
                    notionals = weights * scale
                    target = np.divide(notionals, op[t], out=np.zeros(n), where=op[t] > 0)
                    fee = emit(t, target, op[t], "entry"); q += target
                    active = {"q": target.copy(), "entry_px": op[t].copy(), "entry_t": t,
                              "entry_fee": fee, "funding": 0.0,
                              "hold_events": cfg["hold_events"]}

        event_pos = pos_by_slot.get(t)
        if event_pos is not None and start <= t < end:
            # This is a completed event close.  Schedule an exit at a future
            # completed event and an entry from the lagged ranking.
            if active and event_pos >= 0 and not pending_exit:
                due_pos = event_pos + cfg["hold_events"]
                if due_pos < len(slots):
                    pending_exit[int(slots[due_pos]) + 1] = (t, "scheduled")
            if event_pos + 1 < len(slots):
                sig = rank_signal(rates, slots, event_pos, cfg["lookback_events"], cfg["k"], cfg["lag_events"])
                if sig is not None:
                    longs, shorts = sig; beta = market_betas(close, slots, event_pos)
                    weights, feasible = neutral_weights(beta, longs, shorts, gross_fraction, max_single_fraction)
                    if feasible and weights is not None:
                        fill_t = int(t) + 1
                        if fill_t < end and fill_t not in pending_entry:
                            pending_entry[fill_t] = (t, longs, shorts, weights)
                    else:
                        skipped_infeasible += 1

        # Hard cap after mark-to-market drift.  Reduce all legs at this bar's
        # open, paying the same five-bp fee, if the cap is exceeded.
        equity_open = cash + float(q @ op[t]); gross_open = float(np.abs(q * op[t]).sum())
        if gross_open > gross_fraction * max(equity_open, 0.0) and gross_open > 0:
            factor = min(1.0, max(0.0, gross_fraction * (max(equity_open, 0.0) - gross_open * fee_rate) /
                                   (gross_open * (1.0 - gross_fraction * fee_rate))))
            target = q * factor; delta = target - q
            emit(t, delta, op[t], "risk_reduce"); q = target
            if active: active["q"] *= factor
        post_equity = cash + float(q @ op[t]); post_gross = float(np.abs(q * op[t]).sum())
        if post_equity > 0:
            max_open_ratio = max(max_open_ratio, post_gross / post_equity)
            cap_open_violations += int(post_gross > gross_fraction * post_equity + 1e-10)

        equity = cash + float(q @ close[t])
        bars.append({"time": index[t], "equity": equity, "cash": cash,
                     "funding": total_funding, "fees": total_fees,
                     "gross_exposure": float(np.abs(q * close[t]).sum()),
                     "net_exposure": float(q @ close[t]), "open": int(bool(active))})

    if active:
        t = end - 1; p = active; pnl = float(np.dot(p["q"], close[t] - p["entry_px"]))
        delta = -p["q"]; fee = emit(t, delta, close[t], "terminal_exit"); q += delta
        trades.append({"entry_time": str(index[p["entry_t"]]), "exit_time": str(index[t]),
                       "gross_pnl": pnl, "funding": p["funding"], "fee": p["entry_fee"] + fee,
                       "net_pnl": pnl + p["funding"] - p["entry_fee"] - fee,
                       "reason": "terminal", "hold_events": int(p["hold_events"])})
        bars[-1].update({"equity": cash, "cash": cash, "gross_exposure": 0.0,
                         "net_exposure": 0.0, "open": 0})
        active = {}
    if np.max(np.abs(q)) > 1e-10 or active:
        raise AssertionError("non-flat terminal inventory")

    bar = pd.DataFrame(bars); od = pd.DataFrame(orders); tr = pd.DataFrame(trades)
    curve = bar.equity.to_numpy(float); period_days = max((bar.time.iloc[-1] - bar.time.iloc[0]).total_seconds() / 86400.0, 1.0)
    # Account-level attribution remains exact when the hard gross cap performs
    # partial risk reductions; trade-level sums omit those partial closes.
    price_pnl = float(curve[-1] - 1.0 - total_funding + total_fees)
    row = {"config": cfg["name"], "lookback_events": cfg["lookback_events"], "hold_events": cfg["hold_events"],
           "k": cfg["k"], "lag_events": cfg["lag_events"], "fee_bp_one_way": fee_bp,
           "return": float(curve[-1] - 1.0), "cagr": float(curve[-1] ** (365.25 / period_days) - 1.0) if curve[-1] > 0 else np.nan,
           "mdd": _drawdown(curve), "trades": len(tr), "fees": total_fees, "funding": total_funding,
           "turnover": turnover, "gross_pnl": price_pnl,
           "net_pnl": float(curve[-1] - 1.0),
           "max_gross": float(bar.gross_exposure.max()), "max_open_gross_ratio": max_open_ratio,
           "cap_open_violations": cap_open_violations,
           "max_abs_net": float(np.max(np.abs(bar.net_exposure))), "time_in_market": float(bar.open.mean()),
           "skipped_infeasible": skipped_infeasible}
    # Signed entry/exit notionals already equal round-trip PnL; only funding
    # is an external cash flow in this ledger.
    replay = 1.0 - float((od.quantity_change * od.price + od.fee).sum()) if len(od) else 1.0
    replay += total_funding
    row["replay_error"] = float(replay - curve[-1])
    for name, (a, b) in PERIODS.items():
        row[name + "_return"] = _period_summary(bar, a, b)
        row[name + "_cagr"] = _period_cagr(bar, a, b)
    return row, bar, od, tr


def replay_orders(orders: pd.DataFrame, trades: pd.DataFrame, funding: float) -> float:
    """Independent cash replay used by tests and the audit manifest."""
    cash = 1.0
    if len(orders): cash -= float((orders.quantity_change * orders.price + orders.fee).sum())
    return float(cash + funding)


def main() -> None:
    index, op, close, _ = load_prices(); rates, marks, funding_events = load_funding(index)
    start = int(index.searchsorted(pd.Timestamp("2024-06-01", tz="UTC")))
    end = int(index.searchsorted(pd.Timestamp("2026-09-01", tz="UTC")))
    rows = []; manifest = {"strategy": "funding-conditioned cross-sectional relative value",
        "funding_events": funding_events, "gross_fraction": GROSS_FRACTION,
        "max_single_fraction": MAX_SINGLE_FRACTION, "beta_lookback_events": BETA_LOOKBACK_EVENTS,
        "fee_bp_one_way": FEE_BP, "configs": CONFIGS,
        "signal": "completed event, funding rank from lagged event; next 5m open fill",
        "funding": "actual rates and marks settled before each fill",
        "selection": "all 8 configs evaluated; highlighted candidate is exploratory and not out-of-sample validated"}
    for cfg in CONFIGS:
        row, bar, od, tr = run(index, op, close, rates, marks, cfg, start, end)
        if abs(row["replay_error"]) > 1e-8:
            raise AssertionError(row)
        rows.append(row); prefix = f"funding_relative_value_{cfg['name']}"
        bar.to_csv(RESULTS / f"{prefix}_equity.csv", index=False)
        od.to_csv(RESULTS / f"{prefix}_orders.csv", index=False)
        tr.to_csv(RESULTS / f"{prefix}_trades.csv", index=False)
    primary = next(c for c in CONFIGS if c["lookback_events"] == PRIMARY_LOOKBACK_EVENTS and
                   c["k"] == PRIMARY_K and c["lag_events"] == PRIMARY_LAG_EVENTS)
    rev_cfg = dict(primary, name=primary["name"] + "_sign_reversed")
    rev_row, rev_bar, rev_od, rev_tr = run(index, op, close, -rates, marks, rev_cfg, start, end)
    if abs(rev_row["replay_error"]) > 1e-8:
        raise AssertionError(rev_row)
    rev_row["diagnostic"] = "funding_sign_reversed_placebo"
    rows.append(rev_row)
    rev_prefix = f"funding_relative_value_{rev_cfg['name']}"
    rev_bar.to_csv(RESULTS / f"{rev_prefix}_equity.csv", index=False)
    rev_od.to_csv(RESULTS / f"{rev_prefix}_orders.csv", index=False)
    rev_tr.to_csv(RESULTS / f"{rev_prefix}_trades.csv", index=False)
    result = pd.DataFrame(rows); result.to_csv(RESULTS / "funding_relative_value_results.csv", index=False)
    (RESULTS / "funding_relative_value_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__": main()
