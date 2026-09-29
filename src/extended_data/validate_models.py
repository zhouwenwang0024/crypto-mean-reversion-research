"""Frozen multi-model validation on the disjoint 2024-03--2026-02 sample.

The 1m archive is aggregated to a common 5m clock.  Six predeclared models
share the same causal 3h centre, seven-day scale, 28-day hourly fit, entry /
exit rules and cash ledger.  No model or threshold is selected from the final
six-month holdout.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "extended_1m"
RESULTS = ROOT / "results"
SYMBOLS = "BTC ETH BNB SOL XRP DOGE ADA TRX LINK SUI AVAX LTC BCH DOT HBAR XLM FIL UNI NEAR AAVE".split()
P = len(SYMBOLS)
BAR_MINUTES = 5
BARS_DAY = 24 * 60 // BAR_MINUTES
FIT_DAYS = 28
SCALE_DAYS = 7
CENTER_BARS = 36
ENTRY_Z = 2.0
ENTRY_GAP = 0.015
REARM_Z = 1.0
EXIT_GAP = float(np.log1p(0.0025))
MAX_HOLD = 240 // BAR_MINUTES
STOP = 0.03
PERIODS = {
    "development": ("2024-03-01", "2025-03-01"),
    "validation": ("2025-03-01", "2025-09-01"),
    "holdout": ("2025-09-01", "2026-03-01"),
}
MODELS = (
    "self_sma3", "peer_equal", "ridge_std_l100", "pca3_l100",
    "ridge_exp_h168", "ridge_robust",
)


def load_bars() -> tuple[np.ndarray, ...]:
    """Read 1m partitions and aggregate each symbol to an exact 5m grid."""
    out = []
    for symbol in SYMBOLS:
        parts = []
        for path in sorted((DATA / f"symbol={symbol}USDT").glob("month=*.parquet")):
            parts.append(pd.read_parquet(path))
        frame = pd.concat(parts, ignore_index=True)
        idx = pd.to_datetime(frame.open_time_utc_ms, unit="ms", utc=True)
        frame = frame.set_index(idx)
        agg = frame.resample("5min", label="left", closed="left").agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), quote_volume=("quote_volume", "sum"),
        )
        out.append(agg)
    index = out[0].index
    if any(not x.index.equals(index) for x in out[1:]):
        raise ValueError("symbols do not share one 5m grid")
    op = np.column_stack([x.open.to_numpy(float) for x in out])
    hi = np.column_stack([x.high.to_numpy(float) for x in out])
    lo = np.column_stack([x.low.to_numpy(float) for x in out])
    cl = np.column_stack([x.close.to_numpy(float) for x in out])
    vol = np.column_stack([x.quote_volume.to_numpy(float) for x in out])
    if not np.isfinite(cl).all() or (cl <= 0).any() or (vol < 0).any():
        raise ValueError("invalid aggregated prices or volume")
    return index, op, hi, lo, cl, vol


def load_funding(index: pd.DatetimeIndex) -> tuple[np.ndarray, int]:
    cash_per_unit = np.zeros((len(index), P)); events = 0
    for j, symbol in enumerate(SYMBOLS):
        frame = pd.read_parquet(ROOT / "data" / "extended_funding" / f"symbol={symbol}USDT.parquet")
        for row in frame.itertuples(index=False):
            raw_ms = int(row.funding_time_utc_ms); start_ms = int(index[0].value // 10**6)
            k = int(round((raw_ms - start_ms) / 300_000))
            if k < 0 or k >= len(index) or abs(start_ms + k * 300_000 - raw_ms) > 1_000:
                raise ValueError(f"funding event not on 5m grid: {symbol} {raw_ms}")
            cash_per_unit[k, j] = -float(row.funding_rate) * float(row.mark_price); events += 1
    if not np.isfinite(cash_per_unit).all() or events != P * 2277:
        raise ValueError(f"funding coverage failure: {events}")
    return cash_per_unit, events


def fit_weights(ret: np.ndarray, model: str) -> np.ndarray:
    """Fit one daily cross-currency hedge using only the supplied history."""
    n, p = ret.shape
    if model == "self_sma3":
        return np.eye(p)
    if model == "peer_equal":
        w = np.full((p, p), -1.0 / (p - 1)); np.fill_diagonal(w, 1.0); return w
    sd = ret.std(0, ddof=1); mu = ret.mean(0); z = (ret - mu) / sd
    if model == "ridge_robust":
        z = np.clip(z, -4.0, 4.0)
    if model == "ridge_exp_h168":
        wt = 2.0 ** (-np.arange(n - 1, -1, -1) / 168.0); wt /= wt.sum()
        mu = wt @ ret; sd = np.sqrt(np.sum(wt[:, None] * (ret - mu) ** 2, axis=0) /
                                     (1.0 - np.sum(wt * wt)))
        z = (ret - mu) / sd
        cov = (z * wt[:, None]).T @ z / (1.0 - np.sum(wt * wt))
    else:
        cov = z.T @ z / (n - 1)
    if model == "pca3_l100":
        w = np.eye(p); lam = 0.1
        for j in range(p):
            ids = np.delete(np.arange(p), j); _, sv, vh = np.linalg.svd(z[:, ids], full_matrices=False)
            v = vh[:3].T; f = z[:, ids] @ v
            b = np.linalg.solve(f.T @ f + lam * (n - 1) * np.eye(3), f.T @ z[:, j])
            w[j, ids] = -(sd[j] / sd[ids]) * (v @ b)
        return w
    lam = 0.1
    k = np.linalg.solve(cov + lam * np.eye(p), np.eye(p))
    return k / np.diag(k)[:, None] * sd[:, None] / sd[None, :]


def feature_set(log_close: np.ndarray, model: str, fit_days: int = FIT_DAYS,
                center_bars: int = CENTER_BARS) -> dict[str, np.ndarray]:
    n, p = log_close.shape; day_count = n // BARS_DAY
    prefix = np.vstack([np.zeros((1, p)), np.cumsum(log_close, axis=0)])
    center = np.full_like(log_close, np.nan)
    for k in range(center_bars, n):
        center[k] = (prefix[k] - prefix[k - center_bars]) / center_bars
    delta = log_close - center
    dev = np.full((n, p), np.nan); scale = np.full_like(dev, np.nan)
    weights = np.full((day_count, p, p), np.nan); max_peer = np.full(day_count, np.nan)
    first_day = max(fit_days + 1, 29)
    for d in range(first_day, day_count):
        ds = d * BARS_DAY
        hs = ds - (fit_days * 24 + 1) * 12
        hourly = log_close[ds - 1 - np.arange(fit_days * 24, -1, -1, dtype=np.int64) * 12]
        w = fit_weights(np.diff(hourly, axis=0), model)
        weights[d] = w; max_peer[d] = np.nanmax(np.abs(w - np.eye(p)))
        current = delta[ds - 1:ds + BARS_DAY - 1] @ w.T
        past = delta[ds - SCALE_DAYS * BARS_DAY:ds] @ w.T
        dev[ds:ds + BARS_DAY] = current
        scale[ds:ds + BARS_DAY] = np.std(past, axis=0, ddof=1)
    return {"dev": dev, "scale": scale, "weights": weights, "max_peer": max_peer}


def _position_vector(target: int, direction: float, budget: float, price: np.ndarray,
                     w: np.ndarray, hedge: str) -> np.ndarray:
    coeff = np.zeros(P); coeff[target] = 1.0
    if hedge in ("half", "full", "pair"):
        peers = np.delete(np.arange(P), target)
        if hedge == "pair":
            peers = np.array([peers[np.argmax(np.abs(w[target, peers]))]])
        factor = 0.5 if hedge == "half" else 1.0
        coeff[peers] = factor * w[target, peers]
    weights = coeff / np.sum(np.abs(coeff))
    return direction * budget * weights / price


def simulate(op: np.ndarray, cl: np.ndarray, vol: np.ndarray, feat: dict[str, np.ndarray],
             start: int, end: int, cost_bp: float, delay_bars: int = 1,
             hedge: str = "target", signal_stride: int = 1, stable_filter: bool = False,
             exit_gap: float = EXIT_GAP, funding: np.ndarray | None = None,
             reverse_signal: bool = False, signal_phase: int = 0) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cost = cost_bp / 10000.0; cash = 1.0; turnover = 0.0; funding_cash = 0.0; funding_events = 0
    armed = np.ones(P, bool); positions = {}
    pending: list[tuple[int, str, object]] = []; equity = []; exposures = []; trades = []
    for t in range(start, end):
        price = op[t]; known = cl[t - 1]
        due, pending = [x for x in pending if x[0] == t], [x for x in pending if x[0] != t]
        if funding is not None and np.any(funding[t]):
            for pos in positions.values():
                funding_cash += float(np.sum(pos["q"] * funding[t])); funding_events += int(np.count_nonzero(funding[t]))
            # Apply the event cash flow once; the accumulator above is only for reporting.
            flow = float(sum(np.sum(pos["q"] * funding[t]) for pos in positions.values())); cash += flow
        # Net all fills at one timestamp before charging fees.  This is the
        # actual signed order delta when hedge legs trade the same symbol.
        actions = []
        for _, kind, data in due:
            if kind == "entry":
                q, target, signal, budget, sizing_price = data
                if not np.isclose(np.sum(np.abs(q * sizing_price)), budget, rtol=1e-8, atol=1e-10):
                    raise AssertionError("entry notional is not the requested budget")
                actions.append((kind, (q, target, signal, budget)))
            else:
                target = data; actions.append((kind, (target, positions.pop(target))))
        if actions:
            delta = {}
            notionals = []
            for kind, data in actions:
                q = data[0] if kind == "entry" else -data[1]["q"]
                for j, x in enumerate(q): delta[j] = delta.get(j, 0.0) + float(x)
                notionals.append(float(np.sum(np.abs(q * price))))
            turnover += float(sum(abs(x * price[j]) for j, x in delta.items()))
            fee_total = cost * float(sum(abs(x * price[j]) for j, x in delta.items()))
            denom = sum(notionals)
            for (kind, data), gross in zip(actions, notionals):
                fee = fee_total * gross / denom if denom else 0.0
                if kind == "entry":
                    q, target, signal, budget = data
                    cash -= fee
                    positions[target] = {"q": q, "entry_px": price.copy(), "entry": t, "signal": signal,
                                         "fee": fee, "direction": float(np.sign(q[target])), "gross_entry": budget}
                else:
                    target, pos = data; pnl = float(np.sum(pos["q"] * (price - pos["entry_px"])))
                    cash += pnl - fee
                    trades.append({"target": SYMBOLS[target], "entry_bar": pos["entry"], "exit_bar": t,
                                   "hold_bars": t - pos["entry"], "pnl": pnl, "fee": pos["fee"] + fee,
                                   "gross_entry": float(np.sum(np.abs(pos["q"] * pos["entry_px"])))})
        mark = cash + sum(float(np.sum(v["q"] * (price - v["entry_px"]))) for v in positions.values())
        gross = sum(float(np.sum(np.abs(v["q"] * known))) for v in positions.values())
        net = sum(float(np.sum(v["q"] * known)) for v in positions.values())
        equity.append(mark); exposures.append((gross, net))
        if mark <= 0: break
        # All exits use the completed close at t-1 and execute after the chosen delay.
        for target, pos in list(positions.items()):
            loss = float(np.sum(pos["q"] * (known - pos["entry_px"]))) / pos["gross_entry"]
            support = -pos["direction"] * feat["dev"][t, target]
            reason = loss <= -STOP or support <= exit_gap or t + delay_bars - pos["entry"] >= MAX_HOLD
            if reason and not any(x[1] == "exit" and x[2] == target for x in pending):
                if t + delay_bars < end: pending.append((t + delay_bars, "exit", target))
        if (t - start - signal_phase) % signal_stride or t <= start or t + delay_bars >= end: continue
        dv, sc = feat["dev"][t], feat["scale"][t]
        if not np.isfinite(dv).all() or not np.isfinite(sc).all(): continue
        z = dv / sc
        for j in range(P):
            if j not in positions and abs(z[j]) < REARM_Z: armed[j] = True
        pending_targets = {x[2][1] for x in pending if x[1] == "entry"}
        used = sum(float(np.sum(np.abs(v["q"] * known))) for v in positions.values())
        used += sum(float(x[2][3]) for x in pending if x[1] == "entry") if pending else 0.0
        for target in np.argsort(-np.abs(z)):
            gap = np.expm1(dv[target])
            if (target in positions or target in pending_targets or not armed[target] or
                    abs(z[target]) < ENTRY_Z or abs(gap) < ENTRY_GAP or abs(gap) > 1 or
                    vol[t - 1, target] <= 0 or (stable_filter and feat["max_peer"][t // BARS_DAY] > 0.5)):
                continue
            if len(positions) + len(pending_targets) >= 3: break
            budget = min(0.3 * mark, 0.9 * mark - used)
            if budget <= 1e-12: break
            direction = np.sign(z[target]) if reverse_signal else -np.sign(z[target]); q = _position_vector(target, direction, budget, known,
                                                                     feat["weights"][t // BARS_DAY], hedge)
            pending.append((t + delay_bars, "entry", (q, int(target), t, budget, known.copy())))
            pending_targets.add(int(target)); used += budget; armed[target] = False
    if positions:
        price = cl[end - 1]; actions = list(positions.items()); delta = {}; notionals = []
        for _, pos in actions:
            q = -pos["q"]
            for j, x in enumerate(q): delta[j] = delta.get(j, 0.0) + float(x)
            notionals.append(float(np.sum(np.abs(q * price))))
        turnover += float(sum(abs(x * price[j]) for j, x in delta.items()))
        fee_total = cost * float(sum(abs(x * price[j]) for j, x in delta.items())); denom = sum(notionals)
        for (target, pos), gross in zip(actions, notionals):
            pnl = float(np.sum(pos["q"] * (price - pos["entry_px"])))
            fee = fee_total * gross / denom if denom else 0.0; cash += pnl - fee
            trades.append({"target": SYMBOLS[target], "entry_bar": pos["entry"], "exit_bar": end,
                           "hold_bars": end - pos["entry"], "pnl": pnl, "fee": pos["fee"] + fee,
                           "gross_entry": float(np.sum(np.abs(pos["q"] * pos["entry_px"])))})
        if equity:
            equity[-1] = cash
            exposures[-1] = (0.0, 0.0)
    eq = np.asarray(equity); ex = np.asarray(exposures)
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    tr = pd.DataFrame(trades)
    step_year = 365.25 * BARS_DAY
    ret = np.diff(eq) / np.maximum(eq[:-1], 1e-12) if len(eq) > 1 else np.array([])
    summary = {"return": float((cash - 1.0)), "mdd": float(np.min(eq / peak - 1.0)) if len(eq) else np.nan,
               "annualized_return": float(eq[-1] ** (step_year / len(eq)) - 1.0) if len(eq) and eq[-1] > 0 else np.nan,
               "annualized_vol": float(np.std(ret, ddof=1) * np.sqrt(step_year)) if len(ret) > 1 else np.nan,
               "sharpe": float(np.mean(ret) / np.std(ret, ddof=1) * np.sqrt(step_year)) if len(ret) > 1 and np.std(ret, ddof=1) > 0 else np.nan,
               "trades": int(len(tr)), "fees": float(tr.fee.sum()) if len(tr) else 0.0,
               "price_pnl": float(tr.pnl.sum()) if len(tr) else 0.0,
               "funding_cash": float(funding_cash), "funding_events": int(funding_events),
               "turnover": float(turnover),
               "mean_hold_bars": float(tr.hold_bars.mean()) if len(tr) else 0.0,
               "win_rate": float((tr.pnl > 0).mean()) if len(tr) else 0.0,
               "mean_gross_fraction": float(np.mean(ex[:, 0] / np.maximum(eq, 1e-12))) if len(eq) else 0.0,
               "mean_net_fraction": float(np.mean(ex[:, 1] / np.maximum(eq, 1e-12))) if len(eq) else 0.0,
               "max_gross_fraction": float(np.max(ex[:, 0] / np.maximum(eq, 1e-12))) if len(eq) else 0.0,
               "max_abs_net_fraction": float(np.max(np.abs(ex[:, 1] / np.maximum(eq, 1e-12)))) if len(eq) else 0.0,
               "reconciliation_error": float(cash - 1.0 - (tr.pnl.sum() - tr.fee.sum() + funding_cash)) if len(tr) else float(cash - 1.0 - funding_cash)}
    return summary, tr, pd.DataFrame({"equity": eq, "gross": ex[:, 0], "net": ex[:, 1]})


def bar_index(index: pd.DatetimeIndex, stamp: str) -> int:
    x = pd.Timestamp(stamp, tz="UTC")
    return int(index.searchsorted(x))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.rglob("*.parquet")):
        h.update(str(p.relative_to(path)).encode())
        with p.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def main() -> None:
    index, op, _, _, cl, vol = load_bars(); lc = np.log(cl); funding, funding_rows = load_funding(index)
    results = []; stability = []
    feature_cache = {}
    for model in MODELS:
        feat = feature_set(lc, model); feature_cache[model] = feat
        valid_w = feat["weights"][np.isfinite(feat["weights"]).all((1, 2))]
        stability.append({"model": model, "daily_fits": len(valid_w),
                          "mean_weight_l1_change": float(np.mean(np.abs(np.diff(valid_w, axis=0)))) if len(valid_w) > 1 else 0.0,
                          "median_max_peer": float(np.nanmedian(feat["max_peer"]))})
        for period, (a, b) in PERIODS.items():
            start, end = bar_index(index, a), bar_index(index, b)
            for bp in (0.0, 2.0, 5.0, 10.0):
                summary, trades, equity = simulate(op, cl, vol, feat, start, end, bp, delay_bars=1)
                summary.update({"model": model, "period": period, "cost_bp": bp, "delay_minutes": 5,
                                "bar_interval": "5m", "hedge": "target", "start": a, "end": b})
                results.append(summary)
                if bp == 5.0:
                    trades.to_csv(RESULTS / f"extended_{model}_{period}_5bp_trades.csv", index=False)
                    equity.to_csv(RESULTS / f"extended_{model}_{period}_5bp_equity.csv", index=False)
    # Frozen mechanism/risk checks on the final holdout, all at 5bp.
    ridge = feature_cache["ridge_std_l100"]; start, end = bar_index(index, PERIODS["holdout"][0]), bar_index(index, PERIODS["holdout"][1])
    for hedge in ("target", "half", "full", "pair"):
        s, _, _ = simulate(op, cl, vol, ridge, start, end, 5.0, delay_bars=1, hedge=hedge)
        s.update({"model": "ridge_std_l100", "period": "holdout", "cost_bp": 5.0,
                  "delay_minutes": 5, "bar_interval": "5m", "hedge": hedge,
                  "start": PERIODS["holdout"][0], "end": PERIODS["holdout"][1]})
        results.append(s)
    stable, _, _ = simulate(op, cl, vol, ridge, start, end, 5.0, delay_bars=1,
                            hedge="target", stable_filter=True)
    stable.update({"model": "ridge_std_l100_stable_filter", "period": "holdout", "cost_bp": 5.0,
                   "delay_minutes": 5, "bar_interval": "5m", "hedge": "target",
                   "start": PERIODS["holdout"][0], "end": PERIODS["holdout"][1]})
    results.append(stable)
    for days in (21, 35):
        alt = feature_set(lc, "ridge_std_l100", fit_days=days)
        s, _, _ = simulate(op, cl, vol, alt, start, end, 5.0, delay_bars=1)
        s.update({"model": f"ridge_std_w{days}", "period": "holdout", "cost_bp": 5.0,
                  "delay_minutes": 5, "bar_interval": "5m", "hedge": "target",
                  "start": PERIODS["holdout"][0], "end": PERIODS["holdout"][1]})
        results.append(s)
    alt_center = feature_set(lc, "ridge_std_l100", center_bars=72)
    s, _, _ = simulate(op, cl, vol, alt_center, start, end, 5.0, delay_bars=1)
    s.update({"model": "ridge_center6h", "period": "holdout", "cost_bp": 5.0,
              "delay_minutes": 5, "bar_interval": "5m", "hedge": "target",
              "start": PERIODS["holdout"][0], "end": PERIODS["holdout"][1]})
    results.append(s)
    s, _, _ = simulate(op, cl, vol, ridge, start, end, 5.0, delay_bars=1, signal_stride=3)
    s.update({"model": "ridge_signal15m", "period": "holdout", "cost_bp": 5.0,
              "delay_minutes": 5, "bar_interval": "5m", "hedge": "target",
              "start": PERIODS["holdout"][0], "end": PERIODS["holdout"][1]})
    results.append(s)
    for phase in (0, 1, 2):
        s, _, _ = simulate(op, cl, vol, ridge, start, end, 5.0, delay_bars=1,
                           signal_stride=3, signal_phase=phase)
        s.update({"model": f"ridge_signal15m_phase{phase}", "period": "holdout", "cost_bp": 5.0,
                  "delay_minutes": 5, "observation_phase_min": phase * 5,
                  "bar_interval": "5m", "hedge": "target", "start": PERIODS["holdout"][0], "end": PERIODS["holdout"][1]})
        results.append(s)
    # Execution sensitivity is a predeclared diagnostic, not a selection loop.
    for delay in (1, 2, 3):
        s, _, _ = simulate(op, cl, vol, ridge, start, end, 5.0, delay_bars=delay, hedge="target")
        s.update({"model": "ridge_std_l100", "period": "holdout", "cost_bp": 5.0,
                  "delay_minutes": delay * 5, "bar_interval": "5m", "hedge": "target",
                  "start": PERIODS["holdout"][0], "end": PERIODS["holdout"][1]})
        results.append(s)
    for model in MODELS:
        s, _, _ = simulate(op, cl, vol, feature_cache[model], start, end, 5.0, delay_bars=1, funding=funding)
        s.update({"model": model, "period": "holdout", "cost_bp": 5.0, "delay_minutes": 5,
                  "bar_interval": "5m", "hedge": "target", "funding": True,
                  "funding_rows": funding_rows, "start": PERIODS["holdout"][0], "end": PERIODS["holdout"][1]})
        results.append(s)
    for period, (a, b) in PERIODS.items():
        ps, pe = bar_index(index, a), bar_index(index, b)
        for bp in (0.0, 2.0, 5.0, 10.0):
            s, _, _ = simulate(op, cl, vol, ridge, ps, pe, bp, delay_bars=1, funding=funding)
            s.update({"model": "ridge_std_l100", "period": period, "cost_bp": bp, "delay_minutes": 5,
                      "bar_interval": "5m", "hedge": "target", "funding": True,
                      "funding_rows": funding_rows, "start": a, "end": b})
            results.append(s)
    out = pd.DataFrame(results); out.to_csv(RESULTS / "extended_model_results.csv", index=False)
    pd.DataFrame(stability).to_csv(RESULTS / "extended_model_stability.csv", index=False)
    manifest = {"source": "data/extended_1m aggregated to 5m", "data_digest": digest(DATA),
                "download_manifest": "data/extended_1m/download_manifest.json", "warmup": "2024-02",
                "new_sample": "2024-03-01/2026-03-01", "periods": PERIODS,
                "models": list(MODELS), "cost_bp": [0, 2, 5, 10], "delay_minutes": [5, 10, 15],
                "center_minutes": 180, "scale_days": 7, "fit_days": 28, "entry_z": 2,
                "entry_gap": 0.015, "exit_gap": 0.0025, "max_hold_minutes": 240,
                "stop_fraction": 0.03, "funding": "Binance fapi fundingRate plus markPrice; missing is an error; rows per symbol=2277",
                "selection": "all models frozen before holdout; no holdout tuning"}
    (RESULTS / "extended_validation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out.to_string(index=False)); print(pd.DataFrame(stability).to_string(index=False))


if __name__ == "__main__": main()
