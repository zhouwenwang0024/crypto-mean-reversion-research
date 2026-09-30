"""Causal 5-minute backtest for the monthly pair-stat-arb formation.

The formation CSV is treated as immutable input.  A signal is observed at a
completed 5-minute close and filled at the following bar open.  Each open lot
keeps its formation alpha, beta, mean and scale until it is closed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from validate_5m_mean_reversion import RESULTS, SYMBOLS, load_funding, load_prices

FEE_BP = 5.0
ENTRY_Z = 2.5
EXIT_Z = 0.5
REARM_Z = 1.0
MAX_HOLD_BARS = 14 * 288
PAIR_BUDGET = 0.30
MAX_PAIRS = 3


def _ts(value: str | pd.Timestamp) -> pd.Timestamp:
    out = pd.Timestamp(value)
    return out.tz_localize("UTC") if out.tzinfo is None else out.tz_convert("UTC")


def _formation_rows(selected: pd.DataFrame, model: str) -> dict[pd.Timestamp, list[dict]]:
    out: dict[pd.Timestamp, list[dict]] = {}
    for row in selected[selected.model.eq(model)].to_dict("records"):
        month = _ts(row["model_end_time"]).normalize()
        item = {k: row[k] for k in ("a", "b", "alpha", "beta", "cal_mean", "cal_scale", "half_life_hours")}
        item.update({"a": int(item["a"]), "b": int(item["b"]), "alpha": float(item["alpha"]),
                     "beta": float(item["beta"]), "mean": float(item["cal_mean"]),
                     "scale": float(item["cal_scale"])})
        if not np.isfinite([item["alpha"], item["beta"], item["mean"], item["scale"]]).all() or item["scale"] <= 0:
            raise ValueError(f"invalid frozen parameters: {row}")
        out.setdefault(month, []).append(item)
    for month in out:
        out[month].sort(key=lambda x: (x["a"], x["b"]))
    return out


def _event_window(zone: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp("2025-10-11", tz=zone).tz_convert("UTC")
    return start, start + pd.Timedelta(days=1)


def _drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    return float(np.min(equity / peak - 1.0)) if len(equity) else 0.0


def _z(close: np.ndarray, row: dict) -> float:
    spread = np.log(close[row["a"]]) - row["alpha"] - row["beta"] * np.log(close[row["b"]])
    return float((spread - row["mean"]) / row["scale"])


def run_model(index: pd.DatetimeIndex, op: np.ndarray, cl: np.ndarray,
              rates: np.ndarray, marks: np.ndarray, selected: pd.DataFrame,
              model: str, start: int, end: int, *, fee_bp: float = FEE_BP,
              use_funding: bool = True, direction_mutant: bool = False,
              hold_bars: int = MAX_HOLD_BARS, entry_z: float = ENTRY_Z,
              exit_z: float = EXIT_Z, rearm_z: float = REARM_Z,
              force_month_boundary: bool = True, gross_limit: float | None = None,
              delay_bars: int = 0, observe_bars: int = 1,
              phase_bars: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    formations = _formation_rows(selected, model)
    month_rows: dict[pd.Timestamp, list[dict]] = {}
    for month, rows in formations.items():
        month_rows[month] = rows
    cash = 1.0
    month_period = index.tz_convert(None).to_period("M")
    positions: dict[tuple[int, int], dict] = {}
    blocked: set[tuple[int, int]] = set()
    pending_exit: dict[tuple[int, int], int] = {}
    pending_entry: dict[tuple[int, int], tuple[int, dict, int]] = {}
    orders: list[dict] = []
    trades: list[dict] = []
    bars: list[dict] = []
    total_funding = total_fees = total_gross = 0.0
    fee_rate = float(fee_bp) / 10_000.0

    def fee(delta: np.ndarray, px: np.ndarray) -> float:
        return float(np.abs(delta * px).sum() * fee_rate)

    def order_rows(t: int, delta: np.ndarray, px: np.ndarray, kind: str, pid: str, f: float) -> None:
        for j in np.flatnonzero(np.abs(delta) > 0):
            orders.append({"time": str(index[t]), "kind": kind, "position_id": pid,
                           "symbol": SYMBOLS[j], "symbol_index": int(j),
                           "quantity_change": float(delta[j]), "price": float(px[j]),
                           "fee": float(abs(delta[j] * px[j]) * fee_rate), "model": model})

    def close_position(key: tuple[int, int], t: int, px: np.ndarray, forced: bool = False) -> None:
        nonlocal cash, total_fees, total_gross
        p = positions.pop(key)
        pnl = float(np.dot(p["q"], px - p["entry_px"]))
        delta = -p["q"]
        f = fee(delta, px)
        cash += pnl - f
        total_fees += f; total_gross += pnl
        kind = "forced_exit" if forced else "exit"
        order_rows(t, delta, px, kind, p["id"], f)
        trades.append({"model": model, "position_id": p["id"], "a": p["row"]["a"], "b": p["row"]["b"],
                       "entry_time": str(index[p["entry_t"]]), "exit_time": str(index[t]),
                       "entry_z": p["entry_z"], "exit_z": p.get("exit_z", np.nan),
                       "gross_pnl": pnl, "entry_fee": p["entry_fee"], "exit_fee": f,
                       "fee": p["entry_fee"] + f, "funding": p["funding"],
                       "net_pnl": pnl + p["funding"] - p["entry_fee"] - f,
                       "reason": p.get("reason", kind), "hold_bars": int(t - p["entry_t"])})

    for t in range(start, end):
        px_open, px_close = op[t], cl[t]
        # Funding is settled before this bar's executions.
        funding = 0.0
        if use_funding and positions:
            valid = np.isfinite(rates[t]) & np.isfinite(marks[t])
            if valid.any():
                q = np.zeros(len(SYMBOLS))
                for p in positions.values(): q += p["q"]
                funding = float((-q[valid] * marks[t, valid] * rates[t, valid]).sum())
                cash += funding; total_funding += funding
                for p in positions.values(): p["funding"] += float((-p["q"][valid] * marks[t, valid] * rates[t, valid]).sum())

        # Execute intents created on the preceding close.  Exits always precede entries.
        for key, signal_t in list(pending_exit.items()):
            if signal_t + 1 + delay_bars == t and key in positions:
                close_position(key, t, px_open)
                pending_exit.pop(key, None)
        for key, payload in list(pending_entry.items()):
            signal_t, row, direction = payload
            if signal_t + 1 + delay_bars != t:
                continue
            pending_entry.pop(key, None)
            # A month-boundary reset must not carry a signal formed on the
            # last old-month close into the new formation regime.  The
            # cross-month long-horizon mode explicitly disables this rule.
            if force_month_boundary and (index[signal_t].year, index[signal_t].month) != (index[t].year, index[t].month):
                continue
            if key in positions or len(positions) >= MAX_PAIRS:
                continue
            used = {s for p in positions.values() for s in p["symbols"]}
            if key[0] in used or key[1] in used:
                continue
            # Size from the executable open mark.  Using this bar's close here
            # would leak the future five-minute return into the order budget.
            equity = cash + sum(float(np.dot(p["q"], px_open - p["entry_px"])) for p in positions.values())
            budget = PAIR_BUDGET * equity
            if gross_limit is not None:
                used_gross = sum(float(np.abs(p["q"] * px_open).sum()) for p in positions.values())
                room = (gross_limit * equity - used_gross) / (1.0 + gross_limit * fee_rate)
                budget = min(budget, max(0.0, room))
            if budget <= 0:
                continue
            w = np.array([1.0, -row["beta"]]); denom = float(np.abs(w).sum())
            if direction_mutant:
                direction = -direction
            q = np.zeros(len(SYMBOLS)); q[key[0]] = direction * budget / denom / px_open[key[0]]
            q[key[1]] = direction * w[1] * budget / denom / px_open[key[1]]
            f = fee(q, px_open); cash -= f; total_fees += f
            pid = f"{model}:{index[t].isoformat()}:{key[0]}-{key[1]}"
            positions[key] = {"id": pid, "row": row, "q": q, "entry_px": px_open.copy(),
                              "entry_t": t, "entry_z": _z(cl[signal_t], row),
                              "entry_fee": f, "funding": 0.0,
                              "symbols": {key[0], key[1]}, "direction": direction}
            order_rows(t, q, px_open, "entry", pid, f)

        # Signal decisions use this completed close and only this month's frozen rows.
        month = month_period[t].to_timestamp().tz_localize("UTC")
        rows = month_rows.get(month, [])
        for key, p in list(positions.items()):
            row = p["row"]
            z = _z(px_close, row); p["exit_z"] = z
            next_month = t + 1 < end and month_period[t + 1] != month_period[t]
            timed = t + 1 + delay_bars - p["entry_t"] >= hold_bars
            month_exit = force_month_boundary and next_month
            observed = (t - phase_bars) % observe_bars == 0
            if (observed and abs(z) <= exit_z) or timed or month_exit:
                p["reason"] = "mean" if abs(z) <= exit_z else ("timeout" if timed else "month_boundary")
                pending_exit.setdefault(key, t)
                blocked.add(key)
        for row in rows:
            if (t - phase_bars) % observe_bars:
                continue
            key = (row["a"], row["b"])
            if key in blocked:
                if abs(_z(px_close, row)) <= rearm_z:
                    blocked.remove(key)
                else:
                    continue
            if key in positions or key in pending_entry or key in pending_exit:
                continue
            used = {s for p in positions.values() for s in p["symbols"]}
            if key[0] in used or key[1] in used or len(positions) + len(pending_entry) >= MAX_PAIRS:
                continue
            z = _z(px_close, row)
            if abs(z) >= entry_z:
                pending_entry[key] = (t, row, 1 if z < 0 else -1)

        marked = cash + sum(float(np.dot(p["q"], px_close - p["entry_px"])) for p in positions.values())
        gross = sum(float(np.abs(p["q"] * px_close).sum()) for p in positions.values())
        net = sum(float(np.dot(p["q"], px_close)) for p in positions.values())
        bars.append({"time": index[t], "equity": marked, "cash": cash, "funding": funding,
                     "fees": total_fees, "gross_exposure": gross, "net_exposure": net,
                     "open_positions": len(positions)})

    # Terminal close is the only intentionally non-next-open fill.
    if positions:
        t = end - 1
        for key in list(positions):
            positions[key]["reason"] = "terminal"
            close_position(key, t, cl[t], forced=True)
        bars[-1]["equity"] = cash; bars[-1]["cash"] = cash; bars[-1]["gross_exposure"] = 0.0; bars[-1]["net_exposure"] = 0.0; bars[-1]["open_positions"] = 0
    pending_entry.clear(); pending_exit.clear()
    if positions or pending_entry or pending_exit:
        raise AssertionError("non-flat terminal state")
    bar_df = pd.DataFrame(bars)
    trade_df, order_df = pd.DataFrame(trades), pd.DataFrame(orders)
    curve = bar_df.equity.to_numpy(float)
    trade_net = float(trade_df.net_pnl.sum()) if len(trade_df) else 0.0
    summary = {"model": model, "start": str(index[start]), "end": str(index[end - 1]),
               "return": float(curve[-1] - 1.0), "mdd": _drawdown(curve), "trades": len(trade_df),
               "gross_pnl": total_gross, "fees": total_fees, "funding": total_funding,
               "turnover": float(order_df.eval("abs(quantity_change * price)").sum()) if len(order_df) else 0.0,
               "fee_bp_one_way": float(fee_bp), "funding_enabled": bool(use_funding),
               "entry_z": float(entry_z), "exit_z": float(exit_z),
               "rearm_z": float(rearm_z), "hold_bars": int(hold_bars),
               "force_month_boundary": bool(force_month_boundary),
               "gross_limit_on_new_orders": gross_limit, "delay_bars": delay_bars,
               "observe_bars": observe_bars, "phase_bars": phase_bars,
               "direction_mutant": bool(direction_mutant),
               "reconciliation_error": float(curve[-1] - 1.0 - trade_net),
               "max_gross": float(bar_df.gross_exposure.max()), "max_open_positions": int(bar_df.open_positions.max())}
    return bar_df, trade_df, order_df, summary


def replay_orders(bar_df: pd.DataFrame, trades: pd.DataFrame, orders: pd.DataFrame,
                  *, use_funding: bool = True) -> float:
    """Independent quantity/cash replay of the order ledger (futures cash model)."""
    q: dict[int, float] = {}; entry: dict[int, float] = {}; cash = 1.0
    for row in orders.itertuples(index=False):
        j, dq, px = int(row.symbol_index), float(row.quantity_change), float(row.price)
        old = q.get(j, 0.0)
        if old and old * dq < 0:
            close = min(abs(old), abs(dq))
            cash += np.sign(old) * close * (px - entry[j])
            if abs(dq) >= abs(old): entry.pop(j, None)
        new = old + dq
        if abs(new) > 1e-14 and abs(old) <= 1e-14: entry[j] = px
        q[j] = new
        cash -= float(row.fee)
    if use_funding:
        cash += float(bar_df.funding.sum())
    if any(abs(v) > 1e-12 for v in q.values()):
        raise AssertionError("order replay not flat")
    return float(cash)


def attribute(bar_df: pd.DataFrame, trade_df: pd.DataFrame, zone: str, model: str) -> dict:
    a, b = _event_window(zone); mask = (bar_df.time >= a) & (bar_df.time < b)
    ret = bar_df.equity.pct_change().fillna(bar_df.equity.iloc[0] - 1.0)
    zero = (1.0 + ret.where(~mask, 0.0)).prod() - 1.0
    indices = np.flatnonzero(mask.to_numpy())
    first = int(indices[0]) if len(indices) else None
    before = float(bar_df.equity.iloc[first - 1]) if first else 1.0
    after = float(bar_df.equity.iloc[int(indices[-1])]) if len(indices) else before
    return {"model": model, "timezone": zone,
            "event_start_utc": str(a), "event_end_utc_exclusive": str(b), "event_bars": int(mask.sum()),
            "event_equity_change": after - before,
            "event_funding": float(bar_df.loc[mask, "funding"].sum()),
            "event_return_zeroed": float(zero), "full_return": float(bar_df.equity.iloc[-1] - 1.0),
            "crossing_trades": int(((pd.to_datetime(trade_df.entry_time, utc=True) < b) & (pd.to_datetime(trade_df.exit_time, utc=True) >= a)).sum()) if len(trade_df) else 0}


def main() -> None:
    index, op, cl, _ = load_prices(); rates, marks, _ = load_funding(index)
    selected = pd.read_csv(RESULTS / "statarb_formation_selected.csv")
    start = int(index.searchsorted(_ts("2024-06-01"))); end = int(index.searchsorted(_ts("2026-09-01")))
    summaries, events = [], []
    models = sorted(selected.model.unique())
    for model in models:
        bars, trades, orders, summary = run_model(index, op, cl, rates, marks, selected, model, start, end)
        replay = replay_orders(bars, trades, orders)
        summary["independent_replay_error"] = float(replay - bars.equity.iloc[-1])
        if abs(summary["reconciliation_error"]) > 1e-9 or abs(summary["independent_replay_error"]) > 1e-9:
            raise AssertionError(summary)
        prefix = RESULTS / f"statarb_5m_{model}"
        bars.to_csv(str(prefix) + "_equity.csv", index=False); trades.to_csv(str(prefix) + "_trades.csv", index=False); orders.to_csv(str(prefix) + "_orders.csv", index=False)
        summaries.append(summary)
        events.extend(attribute(bars, trades, z, model) for z in ("Asia/Shanghai", "UTC"))
    pd.DataFrame(summaries).to_csv(RESULTS / "statarb_5m_summary.csv", index=False)
    cost_rows = []
    for model in models:
        for bp in (0.0, 2.0, 5.0, 10.0):
            for use_funding in (True, False):
                _, _, _, summary = run_model(index, op, cl, rates, marks, selected, model, start, end,
                                             fee_bp=bp, use_funding=use_funding)
                cost_rows.append(summary)
    pd.DataFrame(cost_rows).to_csv(RESULTS / "statarb_5m_cost_sensitivity.csv", index=False)
    funding_rows = []
    for model in models:
        f_bars, f_trades, f_orders, summary = run_model(index, op, cl, rates, marks, selected, model, start, end,
                                                        fee_bp=FEE_BP, use_funding=False)
        summary["independent_replay_error"] = float(replay_orders(f_bars, f_trades, f_orders, use_funding=False)
                                                     - f_bars.equity.iloc[-1])
        if abs(summary["reconciliation_error"]) > 1e-9 or abs(summary["independent_replay_error"]) > 1e-9:
            raise AssertionError(summary)
        funding_rows.append(summary)
    pd.DataFrame(funding_rows).to_csv(RESULTS / "statarb_5m_funding_sensitivity.csv", index=False)
    mutant_rows = []
    for model in models:
        _, _, _, summary = run_model(index, op, cl, rates, marks, selected, model, start, end, direction_mutant=True)
        mutant_rows.append(summary)
    pd.DataFrame(events).to_csv(RESULTS / "statarb_5m_event_attribution.csv", index=False)
    checks = {"models": models, "entry_z": ENTRY_Z, "exit_z": EXIT_Z,
              "rearm_z": REARM_Z, "max_hold_days": 14, "pair_budget": PAIR_BUDGET, "fee_bp": FEE_BP,
              "event_data_retained": True, "selection_uses_return": False,
              "independent_replay_max_error": float(max(abs(x["independent_replay_error"]) for x in summaries)),
              "cost_sensitivity_rows": len(cost_rows), "funding_sensitivity_rows": len(funding_rows),
              "direction_mutant_rows": len(mutant_rows),
              "direction_mutant_returns": {x["model"]: x["return"] for x in mutant_rows}}
    (RESULTS / "statarb_5m_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False)); print(json.dumps(checks, ensure_ascii=False))


if __name__ == "__main__":
    main()
