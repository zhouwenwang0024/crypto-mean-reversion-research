"""Adaptive residual grid: a perpetual-futures proxy for a short strangle.

The strategy has no option premium.  It buys a residual below a frozen center
and sells it above the center, then closes at the center.  The center drifts
slowly only while flat; each open lot freezes its center and band.  This makes
range harvesting explicit and leaves the trend tail visible.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from run_statarb_backtest import load_funding, load_prices
except ModuleNotFoundError:  # package import from the repository root
    from .run_statarb_backtest import load_funding, load_prices

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SYMBOLS = "BTC ETH BNB SOL XRP DOGE ADA TRX LINK SUI AVAX LTC BCH DOT HBAR XLM FIL UNI NEAR AAVE".split()
FEE_BP = 5.0
MODEL = "johansen"
PAIR_BUDGET = 0.20
MAX_PAIRS = 3
GROSS_CAP = 0.90
MAX_HOLD_HOURS = 14 * 24
ENTRY_Z = 1.0
EXIT_Z = 0.10
STOP_Z = 2.0
REARM_Z = 0.75
COOLDOWN_HOURS = 12
CENTER_RATCHET = 0.15
SCALE_FLOOR = 0.0025
CONFIGS = (
    {"name": "resid_grid_1h_hl7d", "observe_hours": 1, "center_hours": 168, "scale_hours": 168},
    {"name": "resid_grid_4h_hl7d", "observe_hours": 4, "center_hours": 168, "scale_hours": 168},
    {"name": "resid_grid_1h_hl14d", "observe_hours": 1, "center_hours": 336, "scale_hours": 336},
    {"name": "resid_grid_4h_hl14d", "observe_hours": 4, "center_hours": 336, "scale_hours": 336},
)
PERIODS = {"development": ("2024-06-01", "2025-03-01"),
           "validation": ("2025-03-01", "2025-09-01"),
           "historical_holdout": ("2025-09-01", "2026-03-01"),
           "extension": ("2026-03-01", "2026-09-01")}


def formation_rows(selected: pd.DataFrame) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in selected[selected.model.eq(MODEL)].to_dict("records"):
        stamp = pd.Timestamp(r["model_end_time"])
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        month = str(stamp.strftime("%Y-%m"))
        row = {"a": int(r["a"]), "b": int(r["b"]), "alpha": float(r["alpha"]),
               "beta": float(r["beta"]), "mean": float(r["cal_mean"]),
               "scale": max(SCALE_FLOOR, float(r["cal_scale"]))}
        row["key"] = (month, row["a"], row["b"]); out.setdefault(month, []).append(row)
    return out


def residual(close: np.ndarray, row: dict) -> float:
    return float(np.log(close[row["a"]]) - row["alpha"] - row["beta"] * np.log(close[row["b"]]))


def execute(cash: float, q: np.ndarray, delta: np.ndarray, px: np.ndarray, fee_rate: float) -> tuple[float, float]:
    fee = float(np.abs(delta * px).sum() * fee_rate)
    return cash - float(delta @ px) - fee, fee


def run(index, op, cl, vol, rates, marks, selected, cfg, start, end):
    rows_by_month = formation_rows(selected); periods = index.tz_convert(None).to_period("M")
    latest_rows: list[dict] = []
    observe = cfg["observe_hours"] * 12; warmup = max(cfg["center_hours"], cfg["scale_hours"], 168) * 12
    alpha_c = 1.0 - np.exp(-np.log(2.0) / (cfg["center_hours"] * 12))
    alpha_s = 1.0 - np.exp(-np.log(2.0) / (cfg["scale_hours"] * 12))
    states: dict[tuple, dict] = {}; blocked: dict[tuple, int] = {}; rearm_ready: dict[tuple, bool] = {}; positions: dict[tuple, dict] = {}
    q = np.zeros(len(SYMBOLS)); cash = 1.0; fees = funding_cash = turnover = 0.0
    trades = stops = timeouts = risk_reductions = 0; bars = []; orders = []; fee_rate = FEE_BP / 10_000.0

    def order(delta: np.ndarray, px: np.ndarray, kind: str, key: tuple | None = None):
        nonlocal cash, fees, turnover, trades
        cash, fee = execute(cash, q, delta, px, fee_rate); fees += fee; turnover += float(np.abs(delta * px).sum())
        if np.any(np.abs(delta) > 0): trades += int(np.count_nonzero(delta))
        for j in np.flatnonzero(np.abs(delta) > 0):
            orders.append({"time": str(index[t]), "kind": kind, "symbol": SYMBOLS[j], "quantity_change": float(delta[j]), "price": float(px[j]), "fee": float(abs(delta[j] * px[j]) * fee_rate), "pair_key": str(key)})

    for t in range(start, end):
        valid_f = np.isfinite(rates[t]) & np.isfinite(marks[t])
        fund = float((-q[valid_f] * marks[t, valid_f] * rates[t, valid_f]).sum()) if valid_f.any() else 0.0
        cash += fund; funding_cash += fund
        equity_open = cash + float(q @ op[t]); gross = float(np.abs(q * op[t]).sum())
        if gross > GROSS_CAP * max(equity_open, 0.0) and gross > 0:
            factor = max(0.0, GROSS_CAP * max(equity_open, 0.0) / gross)
            target = q * factor; delta = target - q
            order(delta, op[t], "risk_reduce"); q = target; risk_reductions += 1
            for p in positions.values(): p["q"] *= factor
            if factor == 0.0:
                for key in positions: blocked[key] = t + COOLDOWN_HOURS * 12; rearm_ready[key] = False
                positions.clear()
        if t >= start + warmup and (t - start) % observe == 0:
            month = str(periods[t].strftime("%Y-%m"))
            prior_months = [m for m in rows_by_month if m <= month]
            if prior_months:
                latest_rows = rows_by_month[max(prior_months)]
            current = latest_rows
            exit_keys = []
            for key, p in positions.items():
                z = (residual(cl[t - 1], p["row"]) - p["center"]) / p["scale"]
                hit_mean = z >= -EXIT_Z if p["direction"] > 0 else z <= EXIT_Z
                hit_stop = z <= -STOP_Z if p["direction"] > 0 else z >= STOP_Z
                hit_timeout = t - p["entry_t"] >= MAX_HOLD_HOURS * 12
                if hit_mean or hit_stop or hit_timeout:
                    exit_keys.append((key, "mean" if hit_mean else ("stop" if hit_stop else "timeout")))
            for key, reason in exit_keys:
                p = positions.pop(key); delta = -p["q"]; order(delta, op[t], "exit_" + reason, key); q += delta
                if reason == "stop": stops += 1
                if reason == "timeout": timeouts += 1
                states[key]["center"] = (1.0 - CENTER_RATCHET) * p["center"] + CENTER_RATCHET * residual(cl[t - 1], p["row"])
                blocked[key] = t + COOLDOWN_HOURS * 12
                rearm_ready[key] = False
            used = {s for p in positions.values() for s in p["symbols"]}
            for row in current:
                key = row["key"]
                state = states.setdefault(key, {"center": row["mean"], "scale": max(SCALE_FLOOR, row["scale"])})
                if key not in positions and t >= blocked.get(key, -1):
                    s = residual(cl[t - 1], row)
                    state["center"] += np.clip(alpha_c * (s - state["center"]), -0.25 * state["scale"], 0.25 * state["scale"])
                    state["scale"] = max(SCALE_FLOOR, np.sqrt((1.0 - alpha_s) * state["scale"] ** 2 + alpha_s * (s - state["center"]) ** 2))
                if key in positions or len(positions) >= MAX_PAIRS:
                    continue
                if t < blocked.get(key, -1):
                    continue
                if row["a"] in used or row["b"] in used:
                    continue
                z = (residual(cl[t - 1], row) - state["center"]) / state["scale"]
                if not rearm_ready.get(key, True):
                    if abs(z) <= REARM_Z: rearm_ready[key] = True
                    else: continue
                if abs(z) < ENTRY_Z:
                    continue
                if not np.all(np.isfinite(vol[t - 1, [row["a"], row["b"]]])) or not np.all(vol[t - 1, [row["a"], row["b"]]] > 0):
                    continue
                direction = 1 if z < 0 else -1
                equity_open = cash + float(q @ op[t]); gross = float(np.abs(q * op[t]).sum())
                room = max(0.0, GROSS_CAP * equity_open - gross); budget = min(PAIR_BUDGET * equity_open, room)
                if equity_open <= 0 or budget <= 0: continue
                beta = row["beta"]; denom = 1.0 + abs(beta); target = np.zeros(len(SYMBOLS))
                target[row["a"]] = direction * budget / denom / op[t, row["a"]]
                target[row["b"]] = -direction * beta * budget / denom / op[t, row["b"]]
                order(target, op[t], "entry", key); q += target
                positions[key] = {"q": target.copy(), "row": row, "center": state["center"], "scale": state["scale"], "direction": direction, "symbols": {row["a"], row["b"]}, "entry_t": t}
                used |= {row["a"], row["b"]}
            # Existing lots keep their frozen center and width until exit.
            expected = np.zeros(len(SYMBOLS))
            for p in positions.values(): expected += p["q"]
            if not np.allclose(q, expected, atol=1e-10): raise AssertionError("inventory mismatch")
        equity = cash + float(q @ cl[t]); bars.append({"time": index[t], "equity": equity, "cash": cash, "funding": fund, "fees": fees, "gross_exposure": float(np.abs(q * cl[t]).sum()), "net_exposure": float(q @ cl[t]), "open_positions": len(positions)})
    if positions:
        t = end - 1
        for key, p in list(positions.items()):
            delta = -p["q"]; order(delta, cl[t], "terminal_exit", key); q += delta
        positions.clear(); bars[-1].update({"equity": cash, "cash": cash, "fees": fees, "gross_exposure": 0.0, "net_exposure": 0.0, "open_positions": 0})
    if not np.allclose(q, 0.0, atol=1e-10): raise AssertionError("non-flat terminal inventory")
    bar = pd.DataFrame(bars); curve = bar.equity.to_numpy(float); peak = np.maximum.accumulate(np.r_[1.0, curve])[1:]
    order_df = pd.DataFrame(orders)
    replay_cash = 1.0 + funding_cash
    if len(order_df): replay_cash -= float((order_df.quantity_change * order_df.price + order_df.fee).sum())
    row = {"config": cfg["name"], "fee_bp_one_way": FEE_BP, "return": float(curve[-1] - 1.0), "mdd": float(np.min(curve / peak - 1.0)), "fees": fees, "funding": funding_cash, "turnover": turnover, "order_legs": trades, "stop_count": stops, "timeout_count": timeouts, "risk_reductions": risk_reductions, "max_gross": float(bar.gross_exposure.max()), "max_abs_net": float(np.max(np.abs(bar.net_exposure))), "time_in_market": float((bar.open_positions > 0).mean()), "replay_error": float(replay_cash - curve[-1])}
    for name, (a, b) in PERIODS.items():
        ta, tb = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC"); mask = (bar.time >= ta) & (bar.time < tb); prior = bar.loc[bar.time < ta, "equity"]
        row[name + "_return"] = float(bar.loc[mask, "equity"].iloc[-1] / (prior.iloc[-1] if len(prior) else 1.0) - 1.0) if mask.any() else np.nan
    return row, bar, order_df


def main() -> None:
    index, op, cl, vol = load_prices(); rates, marks, funding_events = load_funding(index); selected = pd.read_csv(RESULTS / "statarb_formation_selected.csv")
    start = int(index.searchsorted(pd.Timestamp("2024-06-01", tz="UTC"))); end = int(index.searchsorted(pd.Timestamp("2026-09-01", tz="UTC"))); rows = []
    for cfg in CONFIGS:
        row, bar, orders = run(index, op, cl, vol, rates, marks, selected, cfg, start, end); rows.append(row)
        orders.to_csv(RESULTS / f"adaptive_residual_grid_{cfg['name']}_orders.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULTS / "adaptive_residual_grid_results.csv", index=False)
    manifest = {"model": MODEL, "fee_bp_one_way": FEE_BP, "funding_events": funding_events, "pair_budget": PAIR_BUDGET, "max_pairs": MAX_PAIRS, "gross_cap": GROSS_CAP, "entry_z": ENTRY_Z, "exit_z": EXIT_Z, "stop_z": STOP_Z, "rearm_z": REARM_Z, "cooldown_hours": COOLDOWN_HOURS, "max_hold_hours": MAX_HOLD_HOURS, "center_ratchet": CENTER_RATCHET, "formation_carry_forward": True, "configs": CONFIGS, "execution": "completed close signal, next 5m open market fill; no passive limit assumption", "same_timestamp_orders": "not netted; each leg charged 5 bp", "replay_check": "cash replay from order ledger plus funding; per-config error emitted in results", "selection": "all configs fixed before results"}
    (RESULTS / "adaptive_residual_grid_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__": main()
