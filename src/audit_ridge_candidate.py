"""Independent reconstruction of the missing Ridge candidate bundle.

The named zip is absent locally and from origin.  This module therefore runs
the written specification only; its assumptions are recorded in CONFIG and
the output manifest.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from research_v2 import END, INITIAL, MONTHS, RESULTS, START, SYMBOLS

CONFIG = {
    "fit_hours": 28 * 24,
    "signal_minutes": 5,
    "center_minutes": 180,
    "scale_hours": 28 * 24,
    "entry_sigma": 2.0,
    "entry_pct": 0.015,
    "repair_fraction": 0.50,
    "max_hold_hours": 4.0,
    "stop_pct": 0.03,
    "max_positions": 3,
    "allocation": 0.30,
    "max_new_budget": 0.90,
    "fill_delay_minutes": 1,
    "ridge_alpha": 100.0,
    "fee_modes": [0.0, 5.0],
}


@dataclass
class Position:
    target: str
    side: int
    entry_signal: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    notional: float
    center: float
    entry_residual: float
    entry_z: float
    entry_pct: float
    beta: np.ndarray
    entry_fee: float = 0.0
    funding_cashflow: float = 0.0


def repair_trigger(side: int, current_residual: float, entry_residual: float) -> bool:
    return side * current_residual >= side * entry_residual * (1.0 - CONFIG["repair_fraction"])


def funding_cashflow(side: int, notional: float, entry_price: float, mark_price: float, rate: float) -> float:
    return -side * notional / entry_price * mark_price * rate


def fit_ridge(x: np.ndarray, y: np.ndarray, scaled: bool, center_raw: bool = True) -> tuple[np.ndarray, float]:
    xm, ym = x.mean(0), y.mean()
    if scaled:
        xs = np.where(x.std(0, ddof=1) > 1e-12, x.std(0, ddof=1), 1.0)
        ys = max(float(y.std(ddof=1)), 1e-12)
        xn, yn = (x - xm) / xs, (y - ym) / ys
        bn = np.linalg.solve(xn.T @ xn + CONFIG["ridge_alpha"] * np.eye(x.shape[1]), xn.T @ yn)
        beta = ys * bn / xs
    elif center_raw:
        xc, yc = x - xm, y - ym
        beta = np.linalg.solve(xc.T @ xc + CONFIG["ridge_alpha"] * np.eye(x.shape[1]), xc.T @ yc)
    else:
        beta = np.linalg.solve(x.T @ x + CONFIG["ridge_alpha"] * np.eye(x.shape[1]), x.T @ y)
    return beta, float(ym - xm @ beta)


def load_minute() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(__file__).resolve().parents[1] / "data" / "klines"
    opens, closes = {}, {}
    for s in SYMBOLS:
        frames = [pd.read_parquet(root / f"symbol={s}" / f"month={m}.parquet", columns=["open_time_utc_ms", "open", "close"]) for m in MONTHS]
        d = pd.concat(frames, ignore_index=True)
        ix = pd.to_datetime(d.open_time_utc_ms, unit="ms", utc=True)
        opens[s] = pd.Series(d.open.to_numpy(float), index=ix)
        closes[s] = pd.Series(d.close.to_numpy(float), index=ix)
    ix = pd.date_range(START, END, freq="1min", inclusive="left", tz="UTC")
    return pd.DataFrame(opens).reindex(ix), pd.DataFrame(closes).reindex(ix)


def ridge_daily(hour_close: pd.DataFrame, scaled: bool, center_raw: bool = True) -> dict[pd.Timestamp, list[tuple[np.ndarray, float]]]:
    """Fit target-on-peers hourly returns using complete prior hours only."""
    ret = np.log(hour_close).diff(); out = {}
    for day in pd.date_range(START.normalize() + pd.Timedelta(days=28), END.normalize(), freq="D", inclusive="left", tz="UTC"):
        hist = ret.loc[ret.index < day].tail(CONFIG["fit_hours"])
        if len(hist) != CONFIG["fit_hours"] or hist.isna().any().any():
            continue
        h = hist.to_numpy(float); rows = []
        for i in range(len(SYMBOLS)):
            peers = [j for j in range(len(SYMBOLS)) if j != i]
            beta, alpha = fit_ridge(h[:, peers], h[:, i], scaled, center_raw)
            rows.append((beta, alpha))
        out[day] = rows
    return out


def make_features(close5: pd.DataFrame, fits: dict, leaky_center: bool = False) -> dict[str, pd.DataFrame]:
    lp = np.log(close5); out = {}
    cn = CONFIG["center_minutes"] // CONFIG["signal_minutes"]
    sn = CONFIG["scale_hours"] * 60 // CONFIG["signal_minutes"]
    for i, s in enumerate(SYMBOLS):
        peers = [j for j in range(len(SYMBOLS)) if j != i]
        spread = pd.Series(np.nan, index=close5.index, dtype=float)
        for day, rows in fits.items():
            mask = close5.index.normalize() == day
            if mask.any():
                x = lp.loc[mask, [SYMBOLS[j] for j in peers]].to_numpy(float)
                spread.loc[mask] = lp.loc[mask, s].to_numpy(float) - x @ rows[i][0]
        center = spread.rolling(cn, min_periods=cn).mean()
        if not leaky_center:
            center = center.shift(1)
        residual = spread - center
        scale = residual.rolling(sn, min_periods=sn).std(ddof=1)
        if not leaky_center:
            scale = scale.shift(1)
        out[s] = pd.DataFrame({"spread": spread, "center": center, "residual": residual, "scale": scale}, index=close5.index)
    return out


def funding_table() -> dict[pd.Timestamp, dict[str, tuple[float, float]]]:
    out = {}
    for s in SYMBOLS:
        p = RESULTS / "funding_api" / f"{s}.parquet"
        if not p.exists():
            continue
        for r in pd.read_parquet(p).itertuples(index=False):
            t = pd.to_datetime(int(r.funding_time_utc_ms), unit="ms", utc=True)
            out.setdefault(t, {})[s] = (float(r.funding_rate), float(r.mark_price))
    return out


def run(close5: pd.DataFrame, open1: pd.DataFrame, close1: pd.DataFrame, feat: dict[str, pd.DataFrame], fits: dict, cost_bp: float, with_funding: bool, cooldown: bool = True):
    times = close5.index; prices = close5.to_numpy(float); logs = np.log(prices); n = len(times)
    signal_ends = times + pd.Timedelta(minutes=CONFIG["signal_minutes"] - 1)
    exec_times = signal_ends + pd.Timedelta(minutes=CONFIG["fill_delay_minutes"])
    exec_px = open1.reindex(exec_times).to_numpy(float)
    exec_mark = close1.reindex(exec_times).to_numpy(float)
    valid_exec = exec_times < END
    fallback = np.broadcast_to(prices, exec_px.shape)
    exec_px = np.where(valid_exec[:, None] & ~np.isfinite(exec_px), fallback, exec_px)
    farr = {s: feat[s][["spread", "center", "residual", "scale"]].to_numpy(float) for s in SYMBOLS}
    cash = float(INITIAL); positions: list[Position] = []; trades = []; equity = []
    blocked: dict[str, int] = {}; funds = funding_table() if with_funding else {}
    fund_items = sorted(funds.items()); fund_idx = 0; funding_cash = 0.0; fee_rate = cost_bp / 10000.0
    for k, (t, row) in enumerate(zip(times, prices)):
        signal_end = signal_ends[k]
        while fund_idx < len(fund_items) and fund_items[fund_idx][0] <= signal_end:
            _, event = fund_items[fund_idx]
            for p in positions:
                if p.target in event and np.isfinite(event[p.target][1]):
                    rate, mark = event[p.target]; amount = funding_cashflow(p.side, p.notional, p.entry_price, mark, rate)
                    p.funding_cashflow += amount; cash += amount; funding_cash += amount
            fund_idx += 1
        for p in list(positions):
            j = SYMBOLS.index(p.target); peers = [q for q in range(len(SYMBOLS)) if q != j]
            cur = logs[k, j] - logs[k, peers] @ p.beta - p.center
            ret = row[j] / p.entry_price - 1.0
            # A repair moves the frozen residual toward zero.  The old draft
            # used <=, which exits while the deviation is growing and misses
            # a genuine half-repair (the retained pre_fix artifacts show it).
            repaired = repair_trigger(p.side, cur, p.entry_residual)
            timed = (signal_end - p.entry_time).total_seconds() / 3600.0 >= CONFIG["max_hold_hours"]
            stopped = -p.side * ret >= CONFIG["stop_pct"]
            if not (repaired or timed or stopped): continue
            px = exec_px[k, j]; et = exec_times[k]
            if not np.isfinite(px):
                px, et = prices[-1, j], times[-1]
            pnl = p.side * p.notional * (px / p.entry_price - 1.0)
            exit_fee = abs(p.notional * px / p.entry_price) * fee_rate; cash += pnl - exit_fee
            reason = "stop" if stopped else ("timeout" if timed else "repair50")
            trades.append({"target": p.target, "side": p.side, "signal_time": str(p.entry_signal), "entry_time": str(p.entry_time), "exit_time": str(et), "entry_price": p.entry_price, "exit_price": px, "notional": p.notional, "gross_price_pnl": pnl, "fee": p.entry_fee + exit_fee, "net_pnl": pnl - p.entry_fee - exit_fee, "funding_cashflow": p.funding_cashflow, "net_pnl_including_funding": pnl - p.entry_fee - exit_fee + p.funding_cashflow, "reason": reason, "entry_z": p.entry_z, "entry_pct": p.entry_pct})
            positions.remove(p)
            if cooldown: blocked[p.target] = int(np.sign(p.entry_residual))
        mark_eq = cash + sum(p.side * p.notional * (row[SYMBOLS.index(p.target)] / p.entry_price - 1.0) for p in positions)
        candidates = []
        for j, s in enumerate(SYMBOLS):
            if s in blocked or any(p.target == s for p in positions): continue
            residual, scale, center = farr[s][k, 2], farr[s][k, 3], farr[s][k, 1]
            if not np.isfinite([residual, scale, center]).all(): continue
            z = residual / scale; pct = abs(np.exp(residual) - 1.0)
            if abs(z) >= CONFIG["entry_sigma"] and pct >= CONFIG["entry_pct"]:
                candidates.append((abs(z), s, z, pct, center, residual))
        candidates.sort(reverse=True); budget = 0.0; current_eq = mark_eq
        for _, s, z, pct, center, residual in candidates:
            if len(positions) >= CONFIG["max_positions"] or budget >= current_eq * CONFIG["max_new_budget"]: break
            j = SYMBOLS.index(s); px = exec_px[k, j]; notional = current_eq * CONFIG["allocation"]
            if not np.isfinite(px) or budget + notional > current_eq * CONFIG["max_new_budget"]: continue
            side = -1 if z > 0 else 1; fee = notional * fee_rate; cash -= fee; budget += notional
            beta = fits[pd.Timestamp(t.normalize())][j][0].copy()
            positions.append(Position(s, side, signal_end, exec_times[k], px, notional, center, residual, z, pct, beta, fee)); blocked.pop(s, None)
        if cooldown:
            for s in list(blocked):
                r = farr[s][k, 2]
                if np.isfinite(r) and np.sign(r) != blocked[s]: del blocked[s]
        mark_row = exec_mark[k]
        if not np.isfinite(mark_row).all():
            mark_row = row
        marked_eq = cash + sum(p.side * p.notional * (mark_row[SYMBOLS.index(p.target)] / p.entry_price - 1.0) for p in positions)
        equity.append((exec_times[k] if exec_times[k] < END else times[-1], marked_eq))
    last, row = times[-1], prices[-1]
    for p in list(positions):
        px = row[SYMBOLS.index(p.target)]; pnl = p.side * p.notional * (px / p.entry_price - 1.0)
        exit_fee = abs(p.notional * px / p.entry_price) * fee_rate; cash += pnl - exit_fee
        trades.append({"target": p.target, "side": p.side, "signal_time": str(p.entry_signal), "entry_time": str(p.entry_time), "exit_time": str(last), "entry_price": p.entry_price, "exit_price": px, "notional": p.notional, "gross_price_pnl": pnl, "fee": p.entry_fee + exit_fee, "net_pnl": pnl - p.entry_fee - exit_fee, "funding_cashflow": p.funding_cashflow, "net_pnl_including_funding": pnl - p.entry_fee - exit_fee + p.funding_cashflow, "reason": "end_of_sample", "entry_z": p.entry_z, "entry_pct": p.entry_pct})
    equity.append((last, cash)); eq = pd.Series(dict(equity), dtype=float).sort_index(); tr = pd.DataFrame(trades); dd = eq / eq.cummax() - 1
    meta = {"cost_bp": cost_bp, "with_funding": with_funding, "final_equity": float(cash), "return": float(cash / INITIAL - 1), "max_drawdown": float(dd.min()), "trades": int(len(tr)), "fees": float(tr.fee.sum()) if len(tr) else 0.0, "gross_price_pnl": float(tr.gross_price_pnl.sum()) if len(tr) else 0.0, "funding_cash": funding_cash}
    return eq.to_frame("equity"), tr, meta


def main() -> None:
    opens, closes = load_minute(); close5 = closes.resample("5min", label="left", closed="left").last().dropna(how="any")
    hour = closes.resample("1h", label="left", closed="left").last().dropna(how="any"); runs = []
    for mode, scaled, center_raw in (("scaled", True, True), ("raw_centered", False, True), ("raw_uncentered", False, False)):
        fits = ridge_daily(hour, scaled, center_raw); feat = make_features(close5, fits, leaky_center=False)
        for fee in CONFIG["fee_modes"]:
            for with_funding in (False, True):
                eq, tr, m = run(close5, opens, closes, feat, fits, fee, with_funding)
                tag = f"{mode}_{fee:g}bp_{'funding' if with_funding else 'nofunding'}"
                eq.to_csv(RESULTS / f"ridge_candidate_{tag}_equity.csv"); tr.to_csv(RESULTS / f"ridge_candidate_{tag}_trades.csv", index=False)
                runs.append({"variant": tag, "scaled": scaled, "center_raw": center_raw, **m})
    (RESULTS / "ridge_candidate_audit.json").write_text(json.dumps({"config": CONFIG, "bundle_present": False, "repair_condition": "side*current_residual >= side*entry_residual*(1-repair_fraction)", "variants": runs}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(runs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
