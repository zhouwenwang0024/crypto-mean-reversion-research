"""Combined portfolio backtest for the five neutral peer models.

Signals use a frozen monthly peer basket and are filled at the next five-minute
open.  At most three target positions are open, each using 30% of current
equity divided across the target and its peer hedge.  The same order ledger is
replayed independently for every fee/funding convention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_target_predictor_backtest as bt  # noqa: E402

START, END = "2024-06-01", "2026-09-01"
FEE_BPS = (0.0, 2.0, 5.0, 10.0)
ENTRY_Z, EXIT_Z, REARM_Z = 3.0, 0.5, 1.0
OBSERVE_BARS, HOLD_DAYS, MAX_POSITIONS = 12, 28, 3
TARGET_BUDGET = 0.30


def _event_window(zone: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp("2025-10-11", tz=zone).tz_convert("UTC")
    return start, start + pd.Timedelta(days=1)


def _drawdown(curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(np.r_[1.0, curve])[1:]
    return float(np.min(curve / peak - 1.0)) if len(curve) else 0.0


def _event_row(bars: pd.DataFrame, trades: pd.DataFrame, zone: str) -> dict:
    a, b = _event_window(zone); t = pd.to_datetime(bars.time, utc=True)
    mask = (t >= a) & (t < b); ret = bars.equity.pct_change().fillna(bars.equity.iloc[0] - 1.0)
    ids = np.flatnonzero(mask.to_numpy()); before = float(bars.equity.iloc[ids[0] - 1]) if len(ids) and ids[0] else 1.0
    after = float(bars.equity.iloc[ids[-1]]) if len(ids) else before
    if len(trades):
        et = pd.to_datetime(trades.entry_time, utc=True); xt = pd.to_datetime(trades.exit_time, utc=True)
        crossing = trades.loc[(et < b) & (xt >= a)]
    else:
        crossing = trades
    return {"event_timezone": zone, "event_start_utc": str(a), "event_end_utc_exclusive": str(b),
            "event_bars": int(mask.sum()), "event_equity_change": after - before,
            "event_return_zeroed": float((1.0 + ret.where(~mask, 0.0)).prod() - 1.0),
            "crossing_trades": int(len(crossing)),
            "crossing_trade_net_pnl": float(crossing.net_pnl.sum()) if len(crossing) else 0.0}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _month(t: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(t.year, t.month, 1, tz="UTC")


def run_portfolio(index: pd.DatetimeIndex, op: np.ndarray, close: np.ndarray,
                  rates: np.ndarray, marks: np.ndarray, z: np.ndarray,
                  hedge_map: dict[tuple[int, int], dict[int, float]], start: int, end: int,
                  *, fee_bp: float = 5.0, use_funding: bool = True,
                  entry_z: float = ENTRY_Z, exit_z: float = EXIT_Z, rearm_z: float = REARM_Z,
                  observe_bars: int = OBSERVE_BARS, hold_bars: int = HOLD_DAYS * 288,
                  max_positions: int = MAX_POSITIONS, target_budget: float = TARGET_BUDGET,
                  force_month_boundary: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Run one causal account; ``z`` and ``hedge_map`` are frozen inputs."""
    ncoin = close.shape[1]; symbols = bt.SYMBOLS[:ncoin]; fee_rate = fee_bp / 10_000.0
    cash = 1.0; positions: dict[int, dict] = {}; blocked: set[int] = set()
    pending_entry: dict[int, tuple[int, int, dict[int, float]]] = {}
    pending_exit: dict[int, tuple[int, str]] = {}
    raw_orders: list[dict] = []; net_orders: list[dict] = []; trades: list[dict] = []; bars: list[dict] = []
    fee_entry: dict[str, float] = {}; fee_exit: dict[str, float] = {}
    fees_total = funding_total = gross_total = 0.0

    def emit(t: int, delta: np.ndarray, px: np.ndarray, kind: str, pid: str) -> float:
        nonlocal fees_total
        fee = float(np.abs(delta * px).sum() * fee_rate); fees_total += fee
        for j in np.flatnonzero(np.abs(delta) > 1e-14):
            raw_orders.append({"_t": t, "time": str(index[t]), "kind": kind, "position_id": pid,
                           "target": symbols[int(pid.split(":", 1)[0])], "target_index": int(pid.split(":", 1)[0]),
                           "symbol": symbols[j], "symbol_index": int(j), "quantity_change": float(delta[j]),
                           "price": float(px[j]), "fee": float(abs(delta[j] * px[j]) * fee_rate), "fee_bp": fee_bp})
        return fee

    def settle(t: int) -> None:
        """Net same-bar fills by symbol and refund gross fee overcharge."""
        nonlocal cash, fees_total
        fills = [r for r in raw_orders if r["_t"] == t]
        if not fills: return
        gross = sum(float(r["fee"]) for r in fills); net = 0.0
        for symbol, group in pd.DataFrame(fills).groupby("symbol_index", sort=True):
            rows = group.to_dict("records"); px = float(rows[0]["price"])
            dq = float(sum(r["quantity_change"] for r in rows)); net_fee = abs(dq * px) * fee_rate if abs(dq) > 1e-14 else 0.0
            net += net_fee
            if abs(dq) > 1e-14:
                net_orders.append({"time": rows[0]["time"], "kind": "forced_exit" if any(r["kind"] == "forced_exit" for r in rows) else ("exit" if any(r["kind"] == "exit" for r in rows) else "entry"),
                                   "position_id": "net:" + rows[0]["time"], "target": rows[0]["target"], "target_index": rows[0]["target_index"],
                                   "symbol": rows[0]["symbol"], "symbol_index": int(symbol), "quantity_change": dq, "price": px,
                                   "fee": net_fee, "fee_bp": fee_bp})
            gross_notional = sum(abs(float(r["quantity_change"]) * px) for r in rows)
            for r in rows:
                allocated = net_fee * abs(float(r["quantity_change"]) * px) / gross_notional if gross_notional > 0 else 0.0
                target_map = fee_entry if r["kind"] == "entry" else fee_exit
                target_map[r["position_id"]] = target_map.get(r["position_id"], 0.0) + allocated
        cash += gross - net; fees_total += net - gross

    def close_pos(t: int, target: int, px: np.ndarray, reason: str) -> None:
        nonlocal cash, gross_total
        pos = positions.pop(target); delta = -pos["q"]
        fee = emit(t, delta, px, "forced_exit" if reason == "terminal" else "exit", f"{target}:{index[pos['entry_t']].isoformat()}")
        leg_pnl = pos["q"] * (px - pos["entry_px"]); pnl = float(leg_pnl.sum()); cash += pnl - fee; gross_total += pnl
        pid = f"{target}:{index[pos['entry_t']].isoformat()}"
        trades.append({"target": symbols[target], "target_index": target, "position_id": pid,
                       "entry_time": str(index[pos["entry_t"]]), "exit_time": str(index[t]),
                       "entry_z": pos["entry_z"], "exit_z": float(z[t, target]) if np.isfinite(z[t, target]) else np.nan,
                       "gross_pnl": pnl, "target_leg_pnl": float(leg_pnl[target]), "hedge_leg_pnl": float(pnl - leg_pnl[target]),
                       "entry_fee": 0.0, "exit_fee": 0.0, "fee": 0.0,
                       "net_pnl": pnl + pos["funding"], "funding": pos["funding"],
                       "hold_bars": t - pos["entry_t"], "reason": reason,
                       "hedge_symbols": ",".join(symbols[j] for j in np.flatnonzero(np.abs(pos["q"]) > 1e-14) if j != target)})

    for t in range(start, end):
        month, next_month = _month(index[t]), _month(index[t]) + pd.offsets.MonthBegin(1)
        px_open, px_close = op[t], close[t]
        if use_funding and positions:
            q_total = np.sum([p["q"] for p in positions.values()], axis=0)
            valid = np.isfinite(rates[t]) & np.isfinite(marks[t]); flow = float((-q_total[valid] * marks[t, valid] * rates[t, valid]).sum())
            cash += flow; funding_total += flow
            for p in positions.values():
                p["funding"] += float((-p["q"][valid] * marks[t, valid] * rates[t, valid]).sum())
        # Execute exits before entries at this bar's open.
        for target, (signal_t, reason) in list(pending_exit.items()):
            if signal_t + 1 == t and target in positions:
                close_pos(t, target, px_open, reason); pending_exit.pop(target)
        for target, (signal_t, direction, weights) in sorted(list(pending_entry.items())):
            if signal_t + 1 != t: continue
            pending_entry.pop(target)
            signal_month = _month(index[signal_t])
            if force_month_boundary and signal_month != month or target in positions or len(positions) >= max_positions:
                continue
            if not np.isfinite(px_open).all() or px_open[target] <= 0: continue
            equity_open = cash + float(sum((p["q"] * (px_open - p["entry_px"])).sum() for p in positions.values()))
            budget = target_budget * equity_open; denom = 1.0 + sum(abs(w) for w in weights.values())
            q = np.zeros(ncoin); q[target] = direction * budget / denom / px_open[target]
            for j, weight in weights.items(): q[j] = direction * weight * budget / denom / px_open[j]
            fee = emit(t, q, px_open, "entry", f"{target}:{index[t].isoformat()}"); cash -= fee
            positions[target] = {"q": q, "entry_px": px_open.copy(), "entry_t": t,
                                 "entry_z": float(z[signal_t, target]), "entry_fee": fee, "funding": 0.0}
        settle(t)
        # Make decisions at the completed bar close.
        for target, pos in list(positions.items()):
            z_valid = np.isfinite(z[t, target]); zt = float(z[t, target]) if z_valid else np.nan
            boundary = force_month_boundary and index[t] + pd.Timedelta(minutes=5) >= next_month
            observed = (t - start) % observe_bars == 0
            if boundary or t + 1 - pos["entry_t"] >= hold_bars or (observed and z_valid and abs(zt) <= exit_z):
                reason = "mean" if observed and z_valid and abs(zt) <= exit_z else ("month_boundary" if boundary else "timeout")
                pending_exit.setdefault(target, (t, reason)); blocked.add(target)
        for target in range(ncoin):
            if target in positions or target in pending_entry or not np.isfinite(z[t, target]) or (t - start) % observe_bars:
                continue
            zt = float(z[t, target])
            if target in blocked:
                if abs(zt) <= rearm_z: blocked.remove(target)
            elif abs(zt) >= entry_z:
                weights = hedge_map.get((t, target))
                if weights: pending_entry[target] = (t, -1 if zt > 0 else 1, dict(weights))
        q_total = np.sum([p["q"] for p in positions.values()], axis=0) if positions else np.zeros(ncoin)
        marked = cash + float(sum((p["q"] * (px_close - p["entry_px"])).sum() for p in positions.values()))
        bars.append({"time": index[t], "equity": marked, "cash": cash, "fees": fees_total, "funding": funding_total,
                     "open_positions": len(positions), "gross_exposure": float(np.abs(q_total * px_close).sum()),
                     "net_exposure": float((q_total * px_close).sum())})
    for target in list(positions): close_pos(end - 1, target, close[end - 1], "terminal")
    settle(end - 1)
    for tr in trades:
        pid = tr["position_id"]; tr["entry_fee"] = fee_entry.get(pid, 0.0); tr["exit_fee"] = fee_exit.get(pid, 0.0)
        tr["fee"] = tr["entry_fee"] + tr["exit_fee"]; tr["net_pnl"] = tr["gross_pnl"] + tr["funding"] - tr["fee"]
    if bars: bars[-1].update(equity=cash, cash=cash, fees=fees_total, open_positions=0, gross_exposure=0.0, net_exposure=0.0)
    bar_df, trade_df, order_df = pd.DataFrame(bars), pd.DataFrame(trades), pd.DataFrame(net_orders)
    summary = {"return": float(bar_df.equity.iloc[-1] - 1), "mdd": _drawdown(bar_df.equity.to_numpy(float)),
               "trades": len(trade_df), "fees": fees_total, "funding": funding_total, "gross_pnl": gross_total,
               "turnover": float((order_df.quantity_change.abs() * order_df.price).sum()) if len(order_df) else 0.0,
               "max_open_positions": int(bar_df.open_positions.max()) if len(bar_df) else 0,
               "max_gross_exposure": float(bar_df.gross_exposure.max()) if len(bar_df) else 0.0,
               "reconciliation_error": float(bar_df.equity.iloc[-1] - 1 - (trade_df.net_pnl.sum() if len(trade_df) else 0.0))}
    return bar_df, trade_df, order_df, summary


def replay_portfolio(index: pd.DatetimeIndex, op: np.ndarray, close: np.ndarray, rates: np.ndarray,
                     marks: np.ndarray, orders: pd.DataFrame, start: int, end: int,
                     fee_bp: float, use_funding: bool) -> float:
    ncoin = close.shape[1]; q = np.zeros(ncoin); entry = np.full(ncoin, np.nan); cash = 1.0; rate = fee_bp / 10000.0
    grouped = {_utc(k): g.to_dict("records") for k, g in orders.groupby("time")} if len(orders) else {}
    for t in range(start, end):
        valid = np.isfinite(rates[t]) & np.isfinite(marks[t])
        if use_funding: cash += float((-q[valid] * marks[t, valid] * rates[t, valid]).sum())
        for row in grouped.get(pd.Timestamp(index[t]), []):
            j, dq = int(row["symbol_index"]), float(row["quantity_change"])
            px = float(close[t, j] if row["kind"] == "forced_exit" else op[t, j])
            old_q = q[j]; new_q = old_q + dq
            if abs(old_q) <= 1e-14 and abs(dq) > 0:
                entry[j] = px
            elif old_q * dq < 0:
                closed = min(abs(old_q), abs(dq)); cash += np.sign(old_q) * closed * (px - entry[j])
                if abs(dq) > abs(old_q): entry[j] = px
            elif old_q * dq > 0:
                entry[j] = (old_q * entry[j] + dq * px) / new_q
            q[j] = new_q; cash -= abs(dq * px) * rate
            if abs(q[j]) <= 1e-12: q[j] = 0.0; entry[j] = np.nan
    if np.max(np.abs(q)) > 1e-10: raise AssertionError("portfolio replay not flat")
    return float(cash)


def _utc(value: object) -> pd.Timestamp:
    t = pd.Timestamp(value); return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--predictions", type=Path, default=RESULTS / "neutral_peer_selected.csv")
    parser.add_argument("--hold-days", type=int, default=HOLD_DAYS); parser.add_argument("--entry-z", type=float, default=ENTRY_Z)
    parser.add_argument("--exit-z", type=float, default=EXIT_Z); parser.add_argument("--observe-bars", type=int, default=OBSERVE_BARS)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    index, op, close, _ = bt.load_prices(); rates, marks, _ = bt.load_funding(index)
    start, end = int(index.searchsorted(_utc(START))), int(index.searchsorted(_utc(END)))
    model_names = sorted(pd.read_csv(args.predictions, usecols=["model"])["model"].dropna().astype(str).unique())
    summaries, events, costs = [], [], []; suffix = f"_{args.label}" if args.label else ""
    for model in model_names:
        pred, _, hedge, means, scales = bt.load_predictions(args.predictions, index, close, model=model)
        z = bt._frozen_z(pred, close, means, scales)
        bars, trades, orders, summary = run_portfolio(index, op, close, rates, marks, z, hedge, start, end,
                                                      hold_bars=args.hold_days * 288, entry_z=args.entry_z,
                                                      exit_z=args.exit_z, observe_bars=args.observe_bars)
        replay = replay_portfolio(index, op, close, rates, marks, orders, start, end, 5.0, True)
        summary.update({"model": model, "fee_bp": 5.0, "funding_enabled": True, "independent_replay_error": replay - bars.equity.iloc[-1]})
        if abs(summary["reconciliation_error"]) > 1e-8 or abs(summary["independent_replay_error"]) > 1e-8: raise AssertionError(summary)
        summaries.append(summary)
        for zone in ("UTC", "Asia/Shanghai"):
            row = _event_row(bars, trades, zone); row["model"] = model; events.append(row)
        trades.to_csv(RESULTS / f"neutral_peer_portfolio_{model}{suffix}_trades.csv", index=False)
        orders.to_csv(RESULTS / f"neutral_peer_portfolio_{model}{suffix}_orders.csv", index=False)
        for bp in FEE_BPS:
            for funding in (True, False):
                value = replay_portfolio(index, op, close, rates, marks, orders, start, end, bp, funding)
                costs.append({"model": model, "fee_bp": bp, "funding_enabled": funding,
                              "return": value - 1.0, "trades": len(trades), "fixed_signal_orders": True})
    summary_path = RESULTS / f"neutral_peer_portfolio_summary{suffix}.csv"; events_path = RESULTS / f"neutral_peer_portfolio_events{suffix}.csv"
    costs_path = RESULTS / f"neutral_peer_portfolio_costs{suffix}.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False); pd.DataFrame(events).to_csv(events_path, index=False)
    pd.DataFrame(costs).to_csv(costs_path, index=False)
    order_paths = [RESULTS / f"neutral_peer_portfolio_{model}{suffix}_orders.csv" for model in model_names]
    archive = ROOT / "data" / "combined_5m_2024-02_to_2026-08.zip"
    manifest = {"source": str(args.predictions), "source_runner_sha256": _sha256(Path(__file__)),
                "selected_sha256": _sha256(args.predictions), "archive_sha256": _sha256(archive) if archive.exists() else None,
                "data_period": "2024-02 through 2026-08, evaluated 2024-06-01 through 2026-09-01",
                "funding_events": 11040, "start": START, "end": END,
                "entry_z": args.entry_z, "exit_z": args.exit_z, "rearm_z": REARM_Z, "observe_bars": args.observe_bars,
                "hold_days": args.hold_days, "max_positions": MAX_POSITIONS, "target_budget": TARGET_BUDGET,
                "sizing": "each entry uses 30% of marked current equity at the executable next-open; target and hedge legs share that budget",
                "force_month_boundary": True, "models": model_names,
                "netting": "same UTC timestamp and symbol signed quantities are aggregated; fee is abs(net quantity*price)*fee_bp",
                "funding_attribution": "cash funding is computed on aggregate q; trade attribution uses each position q",
                "event_exclusion": "2025-10-11 is reported in UTC and Asia/Shanghai and is never used for selection",
                "independent_replay_max_abs_error": float(max(abs(x["independent_replay_error"]) for x in summaries)),
                "files": {p.name: _sha256(p) for p in [summary_path, events_path, costs_path] + order_paths}}
    (RESULTS / f"neutral_peer_portfolio_manifest{suffix}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False)); print(pd.DataFrame(costs).to_string(index=False))


if __name__ == "__main__":
    main()
