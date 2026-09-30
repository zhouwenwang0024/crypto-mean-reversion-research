"""Causal backtest for per-target prices predicted from the other coins.

The predictor is an immutable CSV input.  A completed 5-minute close creates
the signal and the order is filled at the next bar open.  The residual scale is
estimated only from residuals observed before the signal unless the predictor
explicitly supplies a causal ``residual_mean``/``residual_scale`` pair.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from validate_5m_mean_reversion import RESULTS, SYMBOLS, load_funding, load_prices

FEE_BPS = (0.0, 2.0, 5.0, 10.0)
ENTRY_Z = 2.5
EXIT_Z = 0.5
REARM_Z = 1.0
ROLLING_DAYS = 28
MAX_HOLD_BARS = 14 * 288
TARGET_BUDGET = 0.30


def _utc(value: object) -> pd.Timestamp:
    out = pd.Timestamp(value)
    return out.tz_localize("UTC") if out.tzinfo is None else out.tz_convert("UTC")


def _column(frame: pd.DataFrame, names: tuple[str, ...], required: bool = True) -> str | None:
    lower = {str(c).strip().lower(): c for c in frame.columns}
    for name in names:
        if name in lower:
            return str(lower[name])
    if required:
        raise ValueError(f"prediction CSV needs one of {names}; columns={list(frame.columns)}")
    return None


def load_predictions(path: Path, index: pd.DatetimeIndex, close: np.ndarray, *, model: str | None = None) -> tuple[np.ndarray, pd.DataFrame, dict[tuple[int, int], dict[int, float]], np.ndarray, np.ndarray]:
    """Return predictions, quality rows, and frozen peer-basket weights."""
    raw = pd.read_csv(path)
    selected_schema = "model_end_time" in raw.columns and "coef_json" in raw.columns
    if selected_schema:
        models = sorted(raw["model"].dropna().astype(str).unique())
        if model is None and len(models) > 1:
            raise ValueError(f"selected model CSV contains {models}; pass model= explicitly")
        if model is not None:
            raw = raw[raw["model"].astype(str).eq(model)].copy()
        if raw.empty:
            raise ValueError(f"no rows for model {model}")
        pred = np.full((len(index), len(SYMBOLS)), np.nan)
        means = np.full_like(pred, np.nan); scales = np.full_like(pred, np.nan)
        hedge_map: dict[tuple[int, int], dict[int, float]] = {}
        sym_to_j = {s: j for j, s in enumerate(SYMBOLS)}
        for row in raw.to_dict("records"):
            target_name = str(row.get("target_symbol", row.get("target"))).upper().replace("USDT", "")
            if target_name not in sym_to_j: raise ValueError(f"unknown target {target_name}")
            target_j = sym_to_j[target_name]
            start = _utc(row["model_end_time"]); end = start.tz_convert(None).to_period("M").to_timestamp() + pd.offsets.MonthBegin(1); end = end.tz_localize("UTC")
            slots = np.flatnonzero((index >= start) & (index < end))
            if not len(slots): continue
            if np.isfinite(pred[slots, target_j]).any():
                raise ValueError(f"overlapping selected models for {target_name} at {start}")
            peers = json.loads(row["feature_symbols"]) if isinstance(row.get("feature_symbols"), str) else []
            if not peers and row.get("feature_indices"):
                peers = [SYMBOLS[int(i)] for i in json.loads(row["feature_indices"])]
            coef = np.asarray(json.loads(row["coef_json"]), float)
            if len(peers) != len(coef): raise ValueError("feature_symbols/coef_json length mismatch")
            peer_ids = [sym_to_j[str(x).upper().replace("USDT", "")] for x in peers]
            pred[slots, target_j] = float(row["intercept"]) + np.log(close[np.ix_(slots, peer_ids)]) @ coef
            means[slots, target_j] = float(row.get("cal_resid_mean", np.nan)); scales[slots, target_j] = float(row.get("cal_resid_scale", np.nan))
            payload = row.get("hedge_weights_json", "{}")
            weights_raw = json.loads(payload) if isinstance(payload, str) else (payload or {})
            hedge_map[(int(slots[0]), target_j)] = {sym_to_j[str(k).upper().replace("USDT", "")]: float(v) for k, v in weights_raw.items() if abs(float(v)) > 1e-12}
            for slot in slots[1:]: hedge_map[(int(slot), target_j)] = dict(hedge_map[(int(slots[0]), target_j)])
        quality = pd.DataFrame({"target": SYMBOLS, "prediction_rows": np.isfinite(pred).sum(0), "basket_rows": [sum(1 for (i, j) in hedge_map if j == k) for k in range(len(SYMBOLS))]})
        return pred, quality, hedge_map, means, scales

    time_col = _column(raw, ("timestamp", "time", "prediction_time", "bar_time"))
    target_col = _column(raw, ("target", "symbol", "coin"))
    log_col = _column(raw, ("predicted_log_price", "prediction_log_price", "pred_log_price"), False)
    price_col = _column(raw, ("predicted_price", "prediction_price", "predicted_close", "prediction"), False)
    if log_col is None and price_col is None:
        raise ValueError("prediction CSV needs predicted_log_price or predicted_price")
    hedge_col = _column(raw, ("hedge_weights_json", "hedge_weights", "basket_weights_json", "peer_weights_json"), False)
    frame = raw.copy()
    frame["_time"] = pd.to_datetime(frame[time_col], utc=True)
    frame["_target"] = frame[target_col].astype(str).str.upper().str.replace("USDT", "", regex=False)
    unknown = sorted(set(frame["_target"]) - set(SYMBOLS))
    if unknown:
        raise ValueError(f"unknown target symbols: {unknown}")
    if frame.duplicated(["_time", "_target"]).any():
        raise ValueError("duplicate target prediction timestamp")
    if "formation_time" in frame:
        form = pd.to_datetime(frame["formation_time"], utc=True)
        if (form > frame["_time"]).any():
            raise ValueError("prediction formation_time is after prediction_time")
    values = pd.to_numeric(frame[log_col], errors="coerce") if log_col else np.log(
        pd.to_numeric(frame[price_col], errors="coerce"))
    frame["_pred_log"] = values.to_numpy(float)
    aligned = np.full((len(index), len(SYMBOLS)), np.nan)
    means = np.full_like(aligned, np.nan); scales = np.full_like(aligned, np.nan)
    hedge_map: dict[tuple[int, int], dict[int, float]] = {}
    slots = pd.Index(index).get_indexer(frame["_time"])
    if (slots < 0).any():
        raise ValueError("prediction timestamps must be exact 5-minute archive timestamps")
    sym_to_j = {s: j for j, s in enumerate(SYMBOLS)}
    for target, value, slot in zip(frame["_target"].to_numpy(), frame["_pred_log"].to_numpy(float), slots):
        aligned[slot, sym_to_j[str(target)]] = float(value)
    mean_col = _column(raw, ("residual_mean", "cal_resid_mean"), False)
    scale_col = _column(raw, ("residual_scale", "cal_resid_scale"), False)
    if mean_col:
        for target, value, slot in zip(frame["_target"].to_numpy(), pd.to_numeric(frame[mean_col], errors="coerce"), slots):
            means[slot, sym_to_j[str(target)]] = float(value)
    if scale_col:
        for target, value, slot in zip(frame["_target"].to_numpy(), pd.to_numeric(frame[scale_col], errors="coerce"), slots):
            scales[slot, sym_to_j[str(target)]] = float(value)
    if hedge_col:
        for target, value, slot in zip(frame["_target"].to_numpy(), frame[hedge_col].to_numpy(), slots):
            if pd.isna(value) or str(value).strip() in ("", "{}", "nan"):
                continue
            try:
                payload = json.loads(value) if isinstance(value, str) else value
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid hedge weights at {target} {index[slot]}") from exc
            if not isinstance(payload, dict):
                raise ValueError("hedge weights must be a JSON object symbol->weight")
            target_j = sym_to_j[str(target)]; weights: dict[int, float] = {}
            for peer, weight in payload.items():
                peer_name = str(peer).upper().replace("USDT", "")
                if peer_name not in sym_to_j or peer_name == str(target):
                    raise ValueError(f"invalid peer in hedge weights: {peer}")
                w = float(weight)
                if not np.isfinite(w):
                    raise ValueError("non-finite hedge weight")
                if abs(w) > 20:
                    raise ValueError("unreasonably large hedge weight")
                if abs(w) > 1e-12:
                    weights[sym_to_j[peer_name]] = w
            hedge_map[(int(slot), target_j)] = weights

    quality: list[dict] = []
    for symbol, j in sym_to_j.items():
        mask = np.isfinite(aligned[:, j]) & np.isfinite(close[:, j])
        residual = np.full(len(index), np.nan)
        residual[mask] = np.log(close[mask, j]) - aligned[mask, j]
        valid = residual[np.isfinite(residual)]
        if len(valid) > 2:
            corr = float(np.corrcoef(np.log(close[mask, j]), aligned[mask, j])[0, 1])
            rmse = float(np.sqrt(np.mean(valid * valid)))
            direction = np.sign(valid[1:]) == np.sign(valid[:-1])
            hit = float(direction.mean()) if len(direction) else np.nan
        else:
            corr = rmse = hit = np.nan
        weight_counts = [len(hedge_map[(i, j)]) for i in range(len(index)) if (i, j) in hedge_map]
        quality.append({"target": symbol, "prediction_rows": int(mask.sum()), "basket_rows": len(weight_counts),
                        "residual_mean": float(np.mean(valid)) if len(valid) else np.nan,
                        "residual_std": float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan,
                        "residual_rmse": rmse, "price_prediction_corr": corr,
                        "residual_sign_persistence": hit})
    return aligned, pd.DataFrame(quality), hedge_map, means, scales


def _rolling_z(pred: np.ndarray, close: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Residual z score, with mean and scale from bars strictly before t."""
    n, m = pred.shape
    z = np.full((n, m), np.nan); residual = np.log(close) - pred
    for j in range(m):
        s = pd.Series(residual[:, j])
        mean = s.shift(1).rolling(window, min_periods=window).mean().to_numpy()
        scale = s.shift(1).rolling(window, min_periods=window).std(ddof=1).to_numpy()
        good = np.isfinite(residual[:, j]) & np.isfinite(mean) & (scale > 0)
        z[good, j] = (residual[good, j] - mean[good]) / scale[good]
    return z, residual


def _month_start(value: object) -> pd.Timestamp:
    t = _utc(value)
    return pd.Timestamp(t.year, t.month, 1, tz="UTC")


def build_frozen_inputs(selected: pd.DataFrame, model: str, index: pd.DatetimeIndex,
                        close: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[tuple[pd.Timestamp, int], dict[int, float]]]:
    """Expand each selected month model to 5m predictions without refitting."""
    n, m = close.shape
    z = np.full((n, m), np.nan); prediction = np.full((n, m), np.nan)
    hedge: dict[tuple[pd.Timestamp, int], dict[int, float]] = {}
    rows = selected[selected.model.eq(model)].copy()
    for raw in rows.to_dict("records"):
        target = int(raw["target"]); month = _month_start(raw["model_end_time"])
        next_month = month + pd.offsets.MonthBegin(1)
        left = int(index.searchsorted(month)); right = int(index.searchsorted(next_month))
        if right <= left: continue
        peers = [int(x) for x in json.loads(raw["feature_indices"])]
        coef = np.asarray(json.loads(raw["coef_json"]), float)
        if len(peers) != len(coef) or target in peers or not np.isfinite(coef).all():
            raise ValueError(f"invalid peer model for {raw.get('target_symbol')}")
        pred = float(raw["intercept"]) + np.log(close[left:right][:, peers]) @ coef
        scale = float(raw["cal_resid_scale"]); mean = float(raw["cal_resid_mean"])
        if not np.isfinite(pred).all() or not np.isfinite([scale, mean]).all() or scale <= 0:
            raise ValueError("non-finite frozen prediction parameters")
        prediction[left:right, target] = pred
        z[left:right, target] = (np.log(close[left:right, target]) - pred - mean) / scale
        payload = json.loads(raw.get("hedge_weights_json", "{}"))
        weights = {}
        for symbol, weight in payload.items():
            j = SYMBOLS.index(str(symbol).upper().replace("USDT", ""))
            if j != target and np.isfinite(float(weight)) and abs(float(weight)) > 1e-12:
                weights[j] = float(weight)
        if weights:
            hedge[(month, target)] = weights
    return z, prediction, hedge


def _frozen_z(pred: np.ndarray, close: np.ndarray, means: np.ndarray, scales: np.ndarray) -> np.ndarray:
    residual = np.log(close) - pred
    z = np.full_like(residual, np.nan)
    supplied = np.isfinite(means) & np.isfinite(scales) & (scales > 0)
    z[supplied] = (residual[supplied] - means[supplied]) / scales[supplied]
    if (~supplied).any():
        rolling, _ = _rolling_z(pred, close, ROLLING_DAYS * 288)
        z[~supplied] = rolling[~supplied]
    return z


def _drawdown(curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(np.r_[1.0, curve])[1:]
    return float(np.min(curve / peak - 1.0)) if len(curve) else 0.0


def run_target(index: pd.DatetimeIndex, op: np.ndarray, close: np.ndarray,
               rates: np.ndarray, marks: np.ndarray, z: np.ndarray, target: int,
               start: int, end: int, *, fee_bp: float = 5.0,
               use_funding: bool = True, entry_z: float = ENTRY_Z,
               exit_z: float = EXIT_Z, rearm_z: float = REARM_Z,
               hold_bars: int = MAX_HOLD_BARS, direction_mutant: bool = False,
               phase_bars: int = 0, observe_bars: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    fee_rate = fee_bp / 10_000.0; cash = 1.0; q = 0.0; entry_px = np.nan
    entry_t = None; entry_z_value = np.nan; entry_fee = 0.0; funding_total = 0.0
    blocked = False; pending_entry: tuple[int, int] | None = None; pending_exit: tuple[int, str] | None = None
    orders: list[dict] = []; trades: list[dict] = []; bars: list[dict] = []
    fees_total = gross_total = 0.0; symbol = SYMBOLS[target]

    def emit_order(t: int, delta: float, px: float, kind: str, pid: str, fee: float) -> None:
        orders.append({"time": str(index[t]), "kind": kind, "position_id": pid,
                       "target": symbol, "symbol_index": target, "quantity_change": delta,
                       "price": px, "fee": fee, "fee_bp": fee_bp})

    def close_position(t: int, px: float, reason: str) -> None:
        nonlocal cash, q, entry_px, entry_t, entry_fee, fees_total, gross_total
        if abs(q) <= 1e-14:
            return
        delta = -q; f = abs(delta * px) * fee_rate
        pnl = q * (px - entry_px); cash += pnl - f; fees_total += f; gross_total += pnl
        pid = f"{symbol}:{index[entry_t].isoformat()}"
        emit_order(t, delta, px, "forced_exit" if reason == "terminal" else "exit", pid, f)
        trades.append({"target": symbol, "position_id": pid, "entry_time": str(index[entry_t]),
                       "exit_time": str(index[t]), "entry_z": entry_z_value,
                       "exit_z": float(z[t, target]) if np.isfinite(z[t, target]) else np.nan,
                       "gross_pnl": pnl, "entry_fee": entry_fee, "exit_fee": f,
                       "fee": entry_fee + f, "funding": funding_for_lot,
                       "net_pnl": pnl + funding_for_lot - entry_fee - f,
                       "reason": reason, "hold_bars": t - entry_t})
        q = 0.0; entry_px = np.nan; entry_t = None; entry_fee = 0.0

    funding_for_lot = 0.0
    for t in range(start, end):
        px_open, px_close = float(op[t, target]), float(close[t, target])
        if use_funding and abs(q) > 1e-14 and np.isfinite(rates[t, target]) and np.isfinite(marks[t, target]):
            flow = -q * marks[t, target] * rates[t, target]
            cash += flow; funding_total += flow; funding_for_lot += flow
        if pending_exit is not None and pending_exit[0] + 1 == t and abs(q) > 1e-14:
            close_position(t, px_open, pending_exit[1]); pending_exit = None
        if pending_entry is not None and pending_entry[0] + 1 == t:
            signal_t, direction = pending_entry; pending_entry = None
            if abs(q) <= 1e-14 and np.isfinite(px_open) and px_open > 0:
                equity = cash
                budget = TARGET_BUDGET * equity
                if direction_mutant: direction = -direction
                q = direction * budget / px_open; f = abs(q * px_open) * fee_rate
                cash -= f; fees_total += f; entry_fee = f; entry_px = px_open; entry_t = t
                entry_z_value = float(z[signal_t, target]); funding_for_lot = 0.0
                emit_order(t, q, px_open, "entry", f"{symbol}:{index[t].isoformat()}", f)
        valid_signal = ((t - phase_bars) % observe_bars == 0 and np.isfinite(z[t, target]))
        if valid_signal:
            zt = float(z[t, target])
            if abs(q) > 1e-14:
                timed = t + 1 - int(entry_t) >= hold_bars
                if abs(zt) <= exit_z or timed:
                    pending_exit = (t, "mean" if abs(zt) <= exit_z else "timeout")
                    blocked = True
            elif blocked:
                if abs(zt) <= rearm_z: blocked = False
            elif abs(zt) >= entry_z:
                pending_entry = (t, -1 if zt > 0 else 1)
        marked = cash + (q * (px_close - entry_px) if abs(q) > 1e-14 else 0.0)
        bars.append({"time": index[t], "equity": marked, "cash": cash,
                     "funding": funding_total, "fees": fees_total,
                     "gross_exposure": abs(q * px_close), "net_exposure": q * px_close,
                     "open_position": int(abs(q) > 1e-14)})
    if abs(q) > 1e-14:
        close_position(end - 1, float(close[end - 1, target]), "terminal")
        bars[-1].update(equity=cash, cash=cash, gross_exposure=0.0, net_exposure=0.0, open_position=0)
    if abs(q) > 1e-12: raise AssertionError("non-flat target account")
    bar_df, trade_df, order_df = pd.DataFrame(bars), pd.DataFrame(trades), pd.DataFrame(orders)
    curve = bar_df.equity.to_numpy(float)
    summary = {"target": symbol, "target_index": target, "start": str(index[start]), "end": str(index[end - 1]),
               "return": float(curve[-1] - 1.0), "mdd": _drawdown(curve), "trades": len(trade_df),
               "gross_pnl": gross_total, "fees": fees_total, "funding": funding_total,
               "turnover": float((order_df.quantity_change.abs() * order_df.price).sum()) if len(order_df) else 0.0,
               "fee_bp_one_way": fee_bp, "funding_enabled": use_funding, "entry_z": entry_z,
               "exit_z": exit_z, "rearm_z": rearm_z, "hold_bars": hold_bars,
               "direction_mutant": direction_mutant,
               "reconciliation_error": float(curve[-1] - 1.0 - (trade_df.net_pnl.sum() if len(trade_df) else 0.0))}
    return bar_df, trade_df, order_df, summary


def replay_target(index: pd.DatetimeIndex, op: np.ndarray, close: np.ndarray, rates: np.ndarray,
                  marks: np.ndarray, bars: pd.DataFrame, orders: pd.DataFrame,
                  target: int, start: int, end: int, fee_bp: float, use_funding: bool) -> float:
    """Independent signed-inventory replay used as a ledger check."""
    grouped = {_utc(k): g.to_dict("records") for k, g in orders.groupby("time")} if len(orders) else {}
    qty = 0.0; entry = np.nan; cash = 1.0; fee_rate = fee_bp / 10_000.0
    for t in range(start, end):
        if use_funding and abs(qty) > 1e-14 and np.isfinite(rates[t, target]) and np.isfinite(marks[t, target]):
            cash += -qty * marks[t, target] * rates[t, target]
        for row in grouped.get(index[t], []):
            px = float(close[t, target] if row["kind"] == "forced_exit" else op[t, target])
            dq = float(row["quantity_change"])
            if abs(qty) > 1e-14 and qty * dq < 0: cash += qty * (px - entry)
            if abs(qty) <= 1e-14 and abs(dq) > 0: entry = px
            qty += dq; cash -= abs(dq * px) * fee_rate
    if abs(qty) > 1e-12: raise AssertionError("replay target not flat")
    return float(cash)


def run_target_basket(index: pd.DatetimeIndex, op: np.ndarray, close: np.ndarray,
                      rates: np.ndarray, marks: np.ndarray, z: np.ndarray,
                      hedge_map: dict[tuple[int, int], dict[int, float]], target: int,
                      start: int, end: int, *, fee_bp: float = 5.0,
                      use_funding: bool = True, entry_z: float = ENTRY_Z,
                      exit_z: float = EXIT_Z, rearm_z: float = REARM_Z,
                      hold_bars: int = MAX_HOLD_BARS,
                      direction_mutant: bool = False,
                      force_month_boundary: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Two-sided target versus frozen synthetic peer basket account."""
    fee_rate = fee_bp / 10_000.0; ncoin = len(SYMBOLS); symbol = SYMBOLS[target]
    cash = 1.0; q = np.zeros(ncoin); entry_px = np.full(ncoin, np.nan)
    entry_t: int | None = None; entry_z_value = np.nan; entry_fee = 0.0; funding_lot = 0.0
    pending_entry: tuple[int, int, dict[int, float]] | None = None; pending_exit: tuple[int, str] | None = None
    blocked = False; orders: list[dict] = []; trades: list[dict] = []; bars: list[dict] = []
    fee_total = funding_total = gross_total = 0.0

    def emit(t: int, delta: np.ndarray, px: np.ndarray, kind: str, pid: str) -> float:
        nonlocal fee_total
        f = float(np.abs(delta * px).sum() * fee_rate); fee_total += f
        for j in np.flatnonzero(np.abs(delta) > 1e-14):
            orders.append({"time": str(index[t]), "kind": kind, "position_id": pid,
                           "target": symbol, "leg": "target" if j == target else "hedge",
                           "symbol": SYMBOLS[j], "symbol_index": int(j),
                           "quantity_change": float(delta[j]), "price": float(px[j]), "fee": float(abs(delta[j] * px[j]) * fee_rate), "fee_bp": fee_bp})
        return f

    def close_position(t: int, px: np.ndarray, reason: str) -> None:
        nonlocal cash, q, entry_px, entry_t, entry_fee, gross_total, funding_lot
        if entry_t is None: return
        delta = -q; f = emit(t, delta, px, "forced_exit" if reason == "terminal" else "exit", f"{symbol}:{index[entry_t].isoformat()}")
        pnl_legs = q * (px - entry_px); pnl = float(pnl_legs.sum()); cash += pnl - f; gross_total += pnl
        leg_ids = np.flatnonzero(np.abs(q) > 1e-14)
        trades.append({"target": symbol, "position_id": f"{symbol}:{index[entry_t].isoformat()}",
                       "entry_time": str(index[entry_t]), "exit_time": str(index[t]),
                       "entry_z": entry_z_value, "exit_z": float(z[t, target]) if np.isfinite(z[t, target]) else np.nan,
                       "target_leg_pnl": float(pnl_legs[target]), "hedge_leg_pnl": float(pnl_legs.sum() - pnl_legs[target]),
                       "gross_pnl": pnl, "entry_fee": entry_fee, "exit_fee": f, "fee": entry_fee + f,
                       "funding": funding_lot, "net_pnl": pnl + funding_lot - entry_fee - f,
                       "reason": reason, "hold_bars": t - entry_t,
                       "hedge_symbols": ",".join(SYMBOLS[j] for j in leg_ids if j != target)})
        q = np.zeros(ncoin); entry_px[:] = np.nan; entry_t = None; entry_fee = 0.0; funding_lot = 0.0

    for t in range(start, end):
        month = pd.Timestamp(index[t].year, index[t].month, 1, tz="UTC")
        next_month = month + pd.offsets.MonthBegin(1)
        px_open, px_close = op[t], close[t]
        if use_funding and entry_t is not None:
            valid = np.isfinite(rates[t]) & np.isfinite(marks[t])
            flow = float((-q[valid] * marks[t, valid] * rates[t, valid]).sum())
            cash += flow; funding_total += flow; funding_lot += flow
        if pending_exit is not None and pending_exit[0] + 1 == t and entry_t is not None:
            close_position(t, px_open, pending_exit[1]); pending_exit = None
        if pending_entry is not None and pending_entry[0] + 1 == t:
            signal_t, direction, weights = pending_entry; pending_entry = None
            signal_month = pd.Timestamp(index[signal_t].year, index[signal_t].month, 1, tz="UTC")
            if (not force_month_boundary or signal_month == month) and entry_t is None and np.isfinite(px_open).all() and px_open[target] > 0:
                budget = TARGET_BUDGET * cash; denom = 1.0 + sum(abs(w) for w in weights.values())
                if direction_mutant: direction = -direction
                q[target] = direction * budget / denom / px_open[target]
                for j, weight in weights.items(): q[j] = direction * weight * budget / denom / px_open[j]
                entry_px = px_open.copy(); entry_t = t; entry_z_value = float(z[signal_t, target])
                entry_fee = emit(t, q, px_open, "entry", f"{symbol}:{index[t].isoformat()}"); cash -= entry_fee; funding_lot = 0.0
        z_valid = np.isfinite(z[t, target]); zt = float(z[t, target]) if z_valid else np.nan
        if entry_t is not None:
            timed = t + 1 - entry_t >= hold_bars
            boundary = force_month_boundary and (index[t] + pd.Timedelta(minutes=5) >= next_month)
            if boundary or timed or (z_valid and abs(zt) <= exit_z):
                reason = "mean" if z_valid and abs(zt) <= exit_z else ("month_boundary" if boundary else "timeout")
                pending_exit = (t, reason); blocked = True
        elif z_valid:
            if blocked:
                if abs(zt) <= rearm_z: blocked = False
            elif abs(zt) >= entry_z:
                weights = hedge_map.get((t, target))
                if weights: pending_entry = (t, -1 if zt > 0 else 1, dict(weights))
        marked = cash + (q * (px_close - entry_px)).sum() if entry_t is not None else cash
        bars.append({"time": index[t], "equity": float(marked), "cash": cash, "funding": funding_total,
                     "fees": fee_total, "target_exposure": float(q[target] * px_close[target]),
                     "hedge_exposure": float((q * px_close).sum() - q[target] * px_close[target]),
                     "gross_exposure": float(np.abs(q * px_close).sum()), "net_exposure": float((q * px_close).sum()),
                     "open_position": int(entry_t is not None)})
    if entry_t is not None:
        close_position(end - 1, close[end - 1], "terminal")
        bars[-1].update(equity=cash, cash=cash, target_exposure=0.0, hedge_exposure=0.0, gross_exposure=0.0, net_exposure=0.0, open_position=0)
    frame = pd.DataFrame(bars); trade_df = pd.DataFrame(trades); order_df = pd.DataFrame(orders)
    curve = frame.equity.to_numpy(float)
    summary = {"target": symbol, "target_index": target, "return": float(curve[-1] - 1), "mdd": _drawdown(curve), "trades": len(trade_df),
               "gross_pnl": gross_total, "fees": fee_total, "funding": funding_total,
               "turnover": float((order_df.quantity_change.abs() * order_df.price).sum()) if len(order_df) else 0.0,
               "fee_bp_one_way": fee_bp, "funding_enabled": use_funding, "basket": True, "direction_mutant": direction_mutant,
               "reconciliation_error": float(curve[-1] - 1 - (trade_df.net_pnl.sum() if len(trade_df) else 0.0)),
               "max_gross_exposure": float(frame.gross_exposure.max()), "max_abs_net_exposure": float(frame.net_exposure.abs().max()),
               "max_target_exposure": float(frame.target_exposure.abs().max()), "max_hedge_exposure": float(frame.hedge_exposure.abs().max())}
    return frame, trade_df, order_df, summary


def replay_basket(index: pd.DatetimeIndex, op: np.ndarray, close: np.ndarray, rates: np.ndarray,
                  marks: np.ndarray, orders: pd.DataFrame, target: int, start: int, end: int,
                  fee_bp: float, use_funding: bool) -> float:
    grouped = {_utc(k): g.to_dict("records") for k, g in orders.groupby("time")} if len(orders) else {}
    q = np.zeros(len(SYMBOLS)); entry = np.full(len(SYMBOLS), np.nan); cash = 1.0; rate = fee_bp / 10_000
    for t in range(start, end):
        if use_funding:
            valid = np.isfinite(rates[t]) & np.isfinite(marks[t]); cash += float((-q[valid] * marks[t, valid] * rates[t, valid]).sum())
        for row in grouped.get(index[t], []):
            j = int(row["symbol_index"]); px = float(close[t, j] if row["kind"] == "forced_exit" else op[t, j]); dq = float(row["quantity_change"])
            if abs(q[j]) > 1e-14 and q[j] * dq < 0: cash += q[j] * (px - entry[j])
            if abs(q[j]) <= 1e-14 and abs(dq) > 0: entry[j] = px
            q[j] += dq; cash -= abs(dq * px) * rate
            if abs(q[j]) <= 1e-14: entry[j] = np.nan
    if np.max(np.abs(q)) > 1e-12: raise AssertionError("basket replay not flat")
    return float(cash)


def replay_basket_cost(index: pd.DatetimeIndex, op: np.ndarray, close: np.ndarray,
                       rates: np.ndarray, marks: np.ndarray, orders: pd.DataFrame,
                       start: int, end: int, fee_bp: float, use_funding: bool) -> dict:
    """Reprice a fixed signal/order set at another fee and funding convention."""
    grouped = {_utc(k): g.to_dict("records") for k, g in orders.groupby("time")} if len(orders) else {}
    q = np.zeros(len(SYMBOLS)); entry = np.full(len(SYMBOLS), np.nan)
    cash = 1.0; fee_total = funding_total = 0.0; curve: list[float] = []
    rate = fee_bp / 10_000.0
    for t in range(start, end):
        if use_funding:
            valid = np.isfinite(rates[t]) & np.isfinite(marks[t])
            flow = float((-q[valid] * marks[t, valid] * rates[t, valid]).sum())
            cash += flow; funding_total += flow
        for row in grouped.get(index[t], []):
            j = int(row["symbol_index"])
            px = float(close[t, j] if row["kind"] == "forced_exit" else op[t, j])
            dq = float(row["quantity_change"])
            if abs(q[j]) > 1e-14 and q[j] * dq < 0:
                cash += q[j] * (px - entry[j]); entry[j] = np.nan
            if abs(q[j]) <= 1e-14 and abs(dq) > 0: entry[j] = px
            q[j] += dq
            fee = abs(dq * px) * rate; cash -= fee; fee_total += fee
            if abs(q[j]) <= 1e-14: entry[j] = np.nan
        curve.append(float(cash + np.nansum(q * (close[t] - entry))))
    if np.max(np.abs(q)) > 1e-12: raise AssertionError("cost replay not flat")
    curve_arr = np.asarray(curve, float)
    return {"return": float(curve_arr[-1] - 1.0), "mdd": _drawdown(curve_arr),
            "fees": fee_total, "funding": funding_total,
            "turnover": float((orders.quantity_change.abs() * orders.price).sum()) if len(orders) else 0.0,
            "trades": int(orders.loc[orders.kind.eq("entry"), "position_id"].nunique()) if len(orders) else 0}


def event_row(bar_df: pd.DataFrame, trade_df: pd.DataFrame, target: str, zone: str) -> dict:
    start = pd.Timestamp("2025-10-11", tz=zone).tz_convert("UTC"); end = start + pd.Timedelta(days=1)
    mask = (bar_df.time >= start) & (bar_df.time < end)
    changes = bar_df.equity.pct_change().fillna(bar_df.equity.iloc[0] - 1.0)
    idx = np.flatnonzero(mask.to_numpy()); before = float(bar_df.equity.iloc[idx[0] - 1]) if len(idx) and idx[0] else 1.0
    after = float(bar_df.equity.iloc[idx[-1]]) if len(idx) else before
    crossing = 0
    if len(trade_df):
        crossing = int(((pd.to_datetime(trade_df.entry_time, utc=True) < end) & (pd.to_datetime(trade_df.exit_time, utc=True) >= start)).sum())
    return {"target": target, "timezone": zone, "event_start_utc": str(start), "event_end_utc_exclusive": str(end),
            "event_bars": int(mask.sum()), "event_equity_change": after - before,
            "event_return_zeroed": float((1.0 + changes.where(~mask, 0.0)).prod() - 1.0), "crossing_trades": crossing}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--predictions", type=Path, default=RESULTS / "target_predictor_selected.csv")
    parser.add_argument("--model", default=None, help="model name for selected-model CSV (default: run every model)")
    parser.add_argument("--skip-cost", action="store_true", help="skip fixed-order fee sensitivity")
    parser.add_argument("--start", default="2024-06-01"); parser.add_argument("--end", default="2026-09-01")
    args = parser.parse_args()
    index, op, close, _ = load_prices(); rates, marks, _ = load_funding(index)
    start, end = int(index.searchsorted(_utc(args.start))), int(index.searchsorted(_utc(args.end)))
    preview = pd.read_csv(args.predictions, nrows=20)
    selected_schema = "model_end_time" in preview.columns and "coef_json" in preview.columns
    model_names = ([args.model] if args.model else sorted(pd.read_csv(args.predictions, usecols=["model"])["model"].dropna().astype(str).unique())) if selected_schema else [None]
    summaries: list[dict] = []; events: list[dict] = []; qualities: list[pd.DataFrame] = []; cost_rows: list[dict] = []; total_hedge_rows = 0
    checks: dict = {"targets": SYMBOLS, "selection_uses_return": False, "models": model_names}
    for model_name in model_names:
        pred, quality, hedge_map, means, scales = load_predictions(args.predictions, index, close, model=model_name)
        total_hedge_rows += len(hedge_map)
        z = _frozen_z(pred, close, means, scales); qualities.append(quality.assign(model=model_name or "direct"))
        active_targets = [j for j in range(len(SYMBOLS)) if np.isfinite(z[:, j]).any() and any((i, j) in hedge_map for i in range(len(index)))]
        base_orders: dict[int, pd.DataFrame] = {}
        for j in active_targets:
            target = SYMBOLS[j]
            bars, trades, orders, summary = run_target_basket(index, op, close, rates, marks, z, hedge_map, j, start, end)
            summary["model"] = model_name or "direct"
            replay = replay_basket(index, op, close, rates, marks, orders, j, start, end, 5.0, True)
            summary["independent_replay_error"] = replay - bars.equity.iloc[-1]
            if abs(summary["independent_replay_error"]) > 1e-9: raise AssertionError(summary)
            base_orders[j] = orders
            prefix = RESULTS / f"target_predictor_{model_name + '_' if model_name else ''}{target}"
            trades.to_csv(str(prefix) + "_trades.csv", index=False); orders.to_csv(str(prefix) + "_orders.csv", index=False)
            summaries.append(summary); events.extend(event_row(bars, trades, target, zone) | {"model": model_name or "direct"} for zone in ("UTC", "Asia/Shanghai"))
        if not args.skip_cost:
            for j in active_targets:
                for bp in FEE_BPS:
                    for fund in (True, False):
                        s = replay_basket_cost(index, op, close, rates, marks, base_orders[j], start, end, bp, fund)
                        s.update({"target": SYMBOLS[j], "target_index": j, "model": model_name or "direct",
                                  "fee_bp_one_way": bp, "funding_enabled": fund, "fixed_signal_orders": True})
                        cost_rows.append(s)
    suffix = f"_{model_names[0]}" if len(model_names) == 1 else ""
    pd.DataFrame(summaries).to_csv(RESULTS / f"target_predictor_basket_summary{suffix}.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(RESULTS / f"target_predictor_basket_cost_sensitivity{suffix}.csv", index=False)
    pd.DataFrame(events).to_csv(RESULTS / f"target_predictor_basket_event_attribution{suffix}.csv", index=False)
    pd.concat(qualities, ignore_index=True).to_csv(RESULTS / f"target_predictor_basket_quality{suffix}.csv", index=False)
    checks.update({"fee_sensitivity_rows": len(cost_rows), "quality_rows": int(sum(len(x) for x in qualities)), "basket_prediction_rows": total_hedge_rows,
                   "entry_z": ENTRY_Z, "exit_z": EXIT_Z, "rolling_days": ROLLING_DAYS,
                   "event_date_excluded_from_selection": True, "stat_arb_legs": "target_vs_frozen_peer_basket"})
    (RESULTS / f"target_predictor_basket_checks{suffix}.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
