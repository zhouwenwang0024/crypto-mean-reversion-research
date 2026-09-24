"""Causal multi-asset mean-reversion research.

The old study is kept for provenance.  This module is the corrected experiment:
features use only completed bars and prior windows, orders are queued until the
delayed one-minute fill, and a futures-style cash/equity ledger reconciles every
fill.  It intentionally uses a small predeclared grid and selects parameters on
May--June before reporting July--August once.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
START = pd.Timestamp("2026-03-01", tz="UTC")
END = pd.Timestamp("2026-09-01", tz="UTC")
MONTHS = [f"2026-{m:02d}" for m in range(3, 9)]
KEEP = ["open_time_utc_ms", "open", "close", "quote_volume"]
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "TRXUSDT", "LINKUSDT", "SUIUSDT", "AVAXUSDT", "LTCUSDT",
    "BCHUSDT", "DOTUSDT", "HBARUSDT", "XLMUSDT", "FILUSDT", "UNIUSDT",
    "NEARUSDT", "AAVEUSDT",
]
TRAIN_END = pd.Timestamp("2026-05-01", tz="UTC")
VALID_END = pd.Timestamp("2026-07-01", tz="UTC")
INITIAL = 100_000.0


@dataclass
class Feature:
    model: str
    frequency: int
    close: pd.DataFrame
    level: pd.DataFrame
    hedge_by_day: dict[pd.Timestamp, np.ndarray]
    neutralize: bool = True


def load_prices() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read only the final 20 symbols and build exact 1m/5m price tables."""
    opens, closes, volumes = {}, {}, {}
    for s in SYMBOLS:
        parts = []
        for m in MONTHS:
            p = DATA / "klines" / f"symbol={s}" / f"month={m}.parquet"
            if not p.exists():
                raise FileNotFoundError(p)
            parts.append(pd.read_parquet(p, columns=KEEP))
        d = pd.concat(parts, ignore_index=True)
        ix = pd.to_datetime(d.open_time_utc_ms, unit="ms", utc=True)
        x = d.set_index(ix)
        opens[s] = x.open.astype(float)
        closes[s] = x.close.astype(float)
        volumes[s] = x.quote_volume.astype(float)
    minute = pd.date_range(START, END, freq="1min", inclusive="left", tz="UTC")
    return (
        pd.DataFrame(opens).reindex(minute),
        pd.DataFrame(closes).reindex(minute),
        pd.DataFrame(volumes).reindex(minute),
    )


def load_funding_events() -> list[tuple[pd.Timestamp, np.ndarray, np.ndarray]]:
    """Recover historical funding mark prices from Binance's public API."""
    out_dir = RESULTS / "funding_api"; out_dir.mkdir(exist_ok=True)
    start_ms, end_ms = int(START.timestamp() * 1000), int(END.timestamp() * 1000) - 1
    status = []
    for s in SYMBOLS:
        path = out_dir / f"{s}.parquet"
        try:
            if path.exists():
                d = pd.read_parquet(path)
            else:
                rows, cursor = [], start_ms
                while cursor <= end_ms:
                    r = requests.get("https://fapi.binance.com/fapi/v1/fundingRate", params={"symbol": s, "startTime": cursor, "endTime": end_ms, "limit": 1000}, timeout=30)
                    r.raise_for_status(); batch = r.json()
                    if not batch: break
                    rows.extend(batch); last = max(int(x["fundingTime"]) for x in batch)
                    if last < cursor: break
                    cursor = last + 1
                    if len(batch) < 1000: break
                d = pd.DataFrame(rows)[["fundingTime", "fundingRate", "markPrice"]].rename(columns={"fundingTime": "funding_time_utc_ms", "fundingRate": "funding_rate", "markPrice": "mark_price"})
                d["symbol"] = s; d = d.drop_duplicates("funding_time_utc_ms").sort_values("funding_time_utc_ms")
                d.to_parquet(path, index=False, compression="zstd")
            d["funding_time_utc_ms"] = d.funding_time_utc_ms.astype("int64"); d["funding_rate"] = d.funding_rate.astype(float); d["mark_price"] = d.mark_price.astype(float)
            status.append({"symbol": s, "status": "ok", "rows": int(len(d)), "path": str(path.relative_to(ROOT))})
        except Exception as e:
            status.append({"symbol": s, "status": "error", "error": repr(e)})
    (RESULTS / "funding_api_status.json").write_text(json.dumps({"retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat(), "results": status}, ensure_ascii=False, indent=2), encoding="utf-8")
    frames = []
    for s in SYMBOLS:
        p = out_dir / f"{s}.parquet"
        if p.exists():
            d = pd.read_parquet(p); d["symbol"] = s; frames.append(d)
    if not frames: return []
    all_d = pd.concat(frames, ignore_index=True)
    events = []
    for t, g in all_d.groupby("funding_time_utc_ms"):
        rate = np.zeros(len(SYMBOLS)); mark = np.full(len(SYMBOLS), np.nan)
        for _, x in g.iterrows():
            i = SYMBOLS.index(x.symbol); rate[i] = float(x.funding_rate); mark[i] = float(x.mark_price)
        events.append((pd.to_datetime(int(t), unit="ms", utc=True), rate, mark))
    return sorted(events, key=lambda x: x[0])


def make_bars(close1: pd.DataFrame, volume1: pd.DataFrame, frequency: int) -> pd.DataFrame:
    """Return left-labelled completed bars; zero-volume bars cannot signal."""
    rule = f"{frequency}min"
    c = close1.resample(rule, label="left", closed="left").last()
    v = volume1.resample(rule, label="left", closed="left").sum(min_count=frequency)
    c = c.where(v > 0).dropna(how="any")
    return c[(c.index >= START) & (c.index < END)]


def rolling_z(level: pd.DataFrame, frequency: int, center_hours: int, scale_days: int) -> pd.DataFrame:
    center_n = max(2, round(center_hours * 60 / frequency))
    scale_n = max(center_n + 1, round(scale_days * 24 * 60 / frequency))
    center = level.rolling(center_n, min_periods=max(2, center_n // 2)).mean().shift(1)
    scale = level.rolling(scale_n, min_periods=max(10, scale_n // 2)).std().shift(1)
    return (level - center) / scale.replace(0, np.nan)


def peer_feature(close: pd.DataFrame, frequency: int) -> Feature:
    lp = np.log(close)
    a = lp.to_numpy()
    peer = (a.sum(axis=1, keepdims=True) - a) / (len(SYMBOLS) - 1)
    level = pd.DataFrame(a - peer, index=close.index, columns=SYMBOLS)
    w = np.full((len(SYMBOLS), len(SYMBOLS)), -0.5 / (len(SYMBOLS) - 1))
    np.fill_diagonal(w, 0.5)
    days = {pd.Timestamp(d): w for d in close.index.normalize().unique()}
    return Feature("PEER", frequency, close, level, days, True)


def self_feature(close: pd.DataFrame, frequency: int) -> Feature:
    level = pd.DataFrame(np.log(close), index=close.index, columns=SYMBOLS)
    days = {pd.Timestamp(d): np.eye(len(SYMBOLS)) for d in close.index.normalize().unique()}
    return Feature("B0", frequency, close, level, days, False)


def residual_feature(close: pd.DataFrame, frequency: int, model: str) -> Feature:
    """Daily-frozen return-factor residuals.  Every fit excludes the current day."""
    ret = np.log(close).diff()
    n, p = ret.shape
    bars_day = 24 * 60 // frequency
    levels = np.full((n, p), np.nan)
    hedge_by_day: dict[pd.Timestamp, np.ndarray] = {}
    last = np.zeros(p)
    started = False
    for day, positions in ret.groupby(ret.index.normalize()).groups.items():
        day = pd.Timestamp(day)
        pos_idx = ret.index.get_indexer(positions)
        hist = ret.loc[ret.index < day].tail(28 * bars_day).dropna()
        if len(hist) < 28 * bars_day:
            continue
        h = hist.to_numpy()
        mu = h.mean(axis=0)
        sd = h.std(axis=0, ddof=1).clip(min=1e-8)
        if model.startswith("PCA"):
            k = int(model[3:])
            cur = ret.iloc[pos_idx].to_numpy()
            residual = np.full((len(pos_idx), p), np.nan); hedge = np.zeros((p, p))
            # Fit each target against factors extracted from the other 19 assets.
            # This prevents the target from helping define its own fair-value factor.
            for i in range(p):
                peers = [j for j in range(p) if j != i]
                zp = (h[:, peers] - mu[peers]) / sd[peers]
                _, _, vh = np.linalg.svd(zp - zp.mean(axis=0), full_matrices=False)
                v = vh[:k].T
                factors = zp @ v
                target_z = (h[:, i] - mu[i]) / sd[i]
                beta_f = np.linalg.solve(factors.T @ factors + 1e-6 * np.eye(k), factors.T @ target_z)
                cp = (cur[:, peers] - mu[peers]) / sd[peers]
                ct = (cur[:, i] - mu[i]) / sd[i]
                residual[:, i] = (ct - (cp @ v) @ beta_f) * sd[i]
                hedge[i, i] = 1.0
                hedge[i, peers] = -(sd[i] / sd[peers]) * (v @ beta_f)
        elif model == "RIDGE":
            hedge = np.zeros((p, p)); residual = np.full((len(positions), p), np.nan)
            for i in range(p):
                peers = [j for j in range(p) if j != i]
                x = h[:, peers]; y = h[:, i]
                xm, xs = x.mean(0), x.std(0, ddof=1).clip(min=1e-8)
                ym, ys = y.mean(), max(float(y.std(ddof=1)), 1e-8)
                xn, yn = (x - xm) / xs, (y - ym) / ys
                beta_n = np.linalg.solve(xn.T @ xn + 100.0 * np.eye(p - 1), xn.T @ yn)
                beta = ys * beta_n / xs
                alpha = ym - xm @ beta
                hedge[i, i] = 1.0
                hedge[i, peers] = -beta
                cc = ret.iloc[pos_idx, peers].to_numpy()
                residual[:, i] = ret.iloc[pos_idx, i].to_numpy() - (alpha + cc @ beta)
        else:
            raise ValueError(model)
        hedge_by_day[day] = hedge
        if not started:
            started = True
            last[:] = 0.0
        for row, pos in enumerate(pos_idx):
            if np.isfinite(residual[row]).all():
                last += residual[row]
                levels[pos] = last
    return Feature(model, frequency, close, pd.DataFrame(levels, index=close.index, columns=SYMBOLS), hedge_by_day)


def feature_grid(close1: pd.DataFrame, volume1: pd.DataFrame) -> dict[tuple[str, int], Feature]:
    out = {}
    for frequency in (5, 15, 60):
        c = make_bars(close1, volume1, frequency)
        out[("B0", frequency)] = self_feature(c, frequency)
        out[("PEER", frequency)] = peer_feature(c, frequency)
        out[("PCA1", frequency)] = residual_feature(c, frequency, "PCA1")
        out[("PCA3", frequency)] = residual_feature(c, frequency, "PCA3")
        out[("PCA5", frequency)] = residual_feature(c, frequency, "PCA5")
    return out


def signed_weights(z: float, hedge: np.ndarray, neutralize: bool = True) -> np.ndarray:
    """Positive residual means short the target residual and long its hedge."""
    w = -np.sign(z) * np.asarray(hedge, dtype=float)
    w[~np.isfinite(w)] = 0.0
    total = np.abs(w).sum()
    if total <= 0:
        return np.zeros_like(w)
    w /= total
    if neutralize:
        # Project the residual trade onto a dollar-neutral portfolio before sizing.
        w -= w.mean()
    # The account-level 20% per-coin exposure cap below handles extreme rows;
    # leaving this projection un-clipped preserves exact dollar neutrality.
    return w / (np.abs(w).sum() or 1.0)


def net_turnover(delta: np.ndarray, price: np.ndarray) -> float:
    delta, price = np.asarray(delta, float), np.asarray(price, float)
    if delta.shape != price.shape or not np.isfinite(price[delta != 0]).all():
        raise ValueError("invalid fill price")
    return float(np.abs(delta * price).sum())


def _equity(cash: float, positions: list[dict], mark: np.ndarray) -> float:
    return float(cash + sum(np.dot(p["qty"], mark - p["entry_px"]) for p in positions))


def backtest(feature: Feature, z: pd.DataFrame, open1: pd.DataFrame, close1: pd.DataFrame,
             entry: float, exit: float, hold_hours: int, cost_bp: float,
             max_positions: int = 3, allocation: float = 0.10,
             flat_boundaries: tuple[pd.Timestamp, ...] = (TRAIN_END, VALID_END),
             funding_events: list[tuple[pd.Timestamp, np.ndarray, np.ndarray]] | None = None,
             entry_mask: pd.DataFrame | None = None,
             exit_rule: str = "band",
             fee_mode: str = "net") -> tuple[pd.Series, pd.DataFrame, dict]:
    """Run a delayed-fill futures ledger; quantities are sized from known signal close."""
    close = feature.close
    frequency = feature.frequency
    signal_ends = close.index + pd.Timedelta(minutes=frequency - 1)
    pending: dict[pd.Timestamp, list[dict]] = {}
    positions: list[dict] = []
    trades, equity_rows, ledger = [], [], []
    cash, next_id = INITIAL, 1
    flattened: set[pd.Timestamp] = set()
    funding_events = funding_events or []
    funding_idx = 0
    funding_cash = 0.0
    funding_rows = []

    def execute(at: pd.Timestamp, actions: list[dict], forced: bool = False) -> None:
        nonlocal cash
        if not actions:
            return
        px = (close1.loc[at] if forced else open1.loc[at]).to_numpy(float)
        touched = np.zeros(len(SYMBOLS), dtype=bool)
        for a in actions:
            touched |= np.abs(a["qty"]) > 0
        if not np.isfinite(px[touched]).all():
            ledger.append({"time": str(at), "status": "cancelled_missing_price", "actions": len(actions)})
            return
        delta = np.zeros(len(SYMBOLS)); gross_actions = []
        for a in actions:
            signed = a["qty"] if a["kind"] == "entry" else -a["qty"]
            delta += signed
            gross_actions.append(float(np.abs(signed * px).sum()))
        net_fee = net_turnover(delta, px) * cost_bp / 10000.0
        gross_fee = sum(gross_actions) * cost_bp / 10000.0
        if fee_mode == "net":
            fee = net_fee
        elif fee_mode == "gross":
            fee = gross_fee
        else:
            raise ValueError(f"unknown fee_mode={fee_mode}")
        gross_sum = sum(gross_actions) or 1.0
        cash -= fee
        for a, gross_action in zip(actions, gross_actions):
            p = a["position"]
            alloc_fee = fee * gross_action / gross_sum
            if a["kind"] == "entry":
                p["entry_px"] = px.copy(); p["entry_time"] = at; p["entry_fee"] = alloc_fee
                p["actual_gross"] = gross_action
                p["entry_turnover"] = gross_action
                positions.append(p)
            else:
                pnl = float(np.dot(p["qty"], px - p["entry_px"]))
                cash += pnl
                if p in positions:
                    positions.remove(p)
                trades.append({
                    "id": p["id"], "model": feature.model, "frequency_min": frequency,
                    "target": p["target"], "signal_time": str(p["signal_time"]),
                    "entry_time": str(p["entry_time"]), "exit_time": str(at),
                    "gross_target": p["gross"], "actual_gross": p.get("actual_gross", p["gross"]),
                    "entry_turnover": p["entry_turnover"],
                    "exit_turnover": gross_action, "price_pnl": pnl,
                    "fee": p.get("entry_fee", 0.0) + alloc_fee,
                    "funding_cashflow": p.get("funding_cashflow", 0.0),
                    "net_pnl": pnl - p.get("entry_fee", 0.0) - alloc_fee,
                    "net_pnl_including_funding": pnl - p.get("entry_fee", 0.0) - alloc_fee + p.get("funding_cashflow", 0.0),
                    "hold_minutes": (at - p["entry_time"]).total_seconds() / 60.0,
                    "z_entry": p["z_entry"], "reason": a.get("reason", "exit"),
                })
        ledger.append({"time": str(at), "status": "filled", "turnover": net_turnover(delta, px),
                       "gross_turnover": sum(gross_actions), "fee": fee,
                       "net_fee": net_fee, "gross_fee": gross_fee, "cash": cash})

    def apply_funding(up_to: pd.Timestamp) -> None:
        nonlocal funding_idx, cash, funding_cash
        while funding_idx < len(funding_events) and funding_events[funding_idx][0] <= up_to:
            at, rate, mark = funding_events[funding_idx]; amount = 0.0
            for p in positions:
                valid = np.isfinite(mark) & np.isfinite(rate) & (p["qty"] != 0)
                p_amount = float(-np.dot(p["qty"][valid] * mark[valid], rate[valid]))
                p["funding_cashflow"] = p.get("funding_cashflow", 0.0) + p_amount
                amount += p_amount
            cash += amount; funding_cash += amount
            funding_rows.append({"time": str(at), "cashflow": amount})
            funding_idx += 1

    for start, row in close.iterrows():
        now = start + pd.Timedelta(minutes=frequency - 1)
        for boundary in flat_boundaries:
            if boundary <= now and boundary not in flattened:
                pending.clear()
                cut = boundary - pd.Timedelta(minutes=1)
                apply_funding(cut)
                execute(cut, [{"kind": "exit", "qty": p["qty"], "position": p, "reason": "period_boundary"} for p in list(positions)], forced=True)
                # Record the forced close at its actual cutoff so the prior
                # period includes its PnL and fee; repeat the same cash at the
                # boundary as the next period's starting equity.
                equity_rows.append((cut, cash))
                equity_rows.append((boundary, cash))
                flattened.add(boundary)
        due = [t for t in pending if t <= now]
        for t in sorted(due):
            # Settlement events at the same timestamp are processed before fills.
            apply_funding(t)
            execute(t, pending.pop(t))
        apply_funding(now)
        mark = row.to_numpy(float)
        if not np.isfinite(mark).all():
            continue
        equity_rows.append((now, _equity(cash, positions, mark)))
        zrow = z.loc[start].to_numpy(float)
        for p in list(positions):
            j = SYMBOLS.index(p["target"])
            age = (now - p["entry_time"]).total_seconds() / 3600.0
            pnl = float(np.dot(p["qty"], mark - p["entry_px"]))
            stop_trigger = pnl / max(p.get("actual_gross", p["gross"]), 1.0) <= -0.03
            timeout_trigger = age >= hold_hours
            if exit_rule == "band":
                mean_trigger = np.isfinite(zrow[j]) and (abs(zrow[j]) <= exit)
            elif exit_rule == "zero_cross":
                mean_trigger = np.isfinite(zrow[j]) and p["side"] * zrow[j] <= 0
            elif exit_rule == "time_only":
                mean_trigger = False
            else:
                raise ValueError(f"unknown exit_rule={exit_rule}")
            trigger = mean_trigger or timeout_trigger or stop_trigger
            if trigger and not p.get("pending_exit"):
                p["pending_exit"] = True
                reason = "stop" if stop_trigger else ("timeout" if timeout_trigger else ("zero_cross" if exit_rule == "zero_cross" else "mean"))
                pending.setdefault(now + pd.Timedelta(minutes=2), []).append({"kind": "exit", "qty": p["qty"], "position": p, "reason": reason})
        reserved = {p["target"] for p in positions} | {a["position"]["target"] for v in pending.values() for a in v if a["kind"] == "entry"}
        for j in np.argsort(-np.nan_to_num(np.abs(zrow), nan=-np.inf)):
            if len(positions) + sum(a["kind"] == "entry" for v in pending.values() for a in v) >= max_positions:
                break
            if j >= len(SYMBOLS) or SYMBOLS[j] in reserved or not np.isfinite(zrow[j]):
                continue
            if abs(float(zrow[j])) < entry:
                continue
            if entry_mask is not None and not bool(entry_mask.loc[start, SYMBOLS[j]]):
                continue
            hedge = feature.hedge_by_day.get(pd.Timestamp(start.normalize()))
            if hedge is None:
                continue
            px_signal = mark
            eq = _equity(cash, positions, mark)
            gross = max(0.0, eq * allocation)
            w = signed_weights(float(zrow[j]), hedge[j],
                               neutralize=feature.neutralize and feature.model not in {"B0", "AR1"})
            qty = gross * w / px_signal
            if not np.isfinite(qty).all() or np.abs(qty).sum() == 0:
                continue
            exposure = sum((p["qty"] * px_signal for p in positions), np.zeros(len(SYMBOLS)))
            exposure += sum((a["qty"] * px_signal for v in pending.values() for a in v if a["kind"] == "entry"), np.zeros(len(SYMBOLS)))
            contribution = qty * px_signal
            alpha = 1.0
            limit = eq * 0.20
            for i in range(len(SYMBOLS)):
                if abs(exposure[i] + contribution[i]) > limit and abs(contribution[i]) > 0:
                    room = max(0.0, limit - abs(exposure[i]))
                    alpha = min(alpha, room / abs(contribution[i]))
            qty *= max(0.0, min(1.0, alpha))
            if np.abs(qty).sum() == 0:
                continue
            actual_gross = float(np.abs(qty * px_signal).sum())
            p = {"id": next_id, "target": SYMBOLS[j], "qty": qty, "entry_px": None, "entry_time": None,
                 "signal_time": now, "gross": gross, "actual_gross": actual_gross,
                 "entry_turnover": actual_gross,
                 "z_entry": float(zrow[j]), "side": float(np.sign(zrow[j]))}
            next_id += 1; reserved.add(SYMBOLS[j])
            pending.setdefault(now + pd.Timedelta(minutes=2), []).append({"kind": "entry", "qty": qty, "position": p})
        
    # Do not fill an order beyond the sample.  Close all filled lots at the final minute close.
    pending.clear()
    last_time = close1.index[-1]
    apply_funding(last_time)
    execute(last_time, [{"kind": "exit", "qty": p["qty"], "position": p, "reason": "end_of_sample"} for p in list(positions)], forced=True)
    eq = pd.Series(dict(equity_rows)).sort_index()
    if len(eq):
        eq.loc[last_time] = cash
    meta = {"cash_final": cash, "ledger": ledger, "funding_rows": funding_rows, "funding_cash": funding_cash, "cancelled_pending": True, "flat_boundaries": [str(x) for x in flat_boundaries]}
    return eq, pd.DataFrame(trades), meta


def metrics(eq: pd.Series, trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, funding_rows: list[dict] | None = None) -> dict:
    x = eq[(eq.index >= start) & (eq.index < end)]
    if x.empty:
        return {"return": np.nan, "max_drawdown": np.nan, "trades": 0}
    previous = eq[eq.index < start]
    base = float(previous.iloc[-1]) if len(previous) else INITIAL
    daily_end = x.resample("1D").last().dropna()
    prior_day = daily_end.shift(1)
    prior_day.iloc[0] = base
    daily = daily_end / prior_day - 1.0
    peak = x.cummax().clip(lower=base); dd = x / peak - 1
    t = trades[(pd.to_datetime(trades.exit_time, utc=True) >= start) & (pd.to_datetime(trades.exit_time, utc=True) < end)] if len(trades) else trades
    turnover = float(t.entry_turnover.add(t.exit_turnover).sum()) if len(t) else 0.0
    gross = float(t.price_pnl.sum()) if len(t) else 0.0
    funding = sum(float(x["cashflow"]) for x in (funding_rows or []) if start <= pd.Timestamp(x["time"]) < end)
    return {
        "return": float(x.iloc[-1] / base - 1), "start_equity": base, "end_equity": float(x.iloc[-1]),
        "max_drawdown": float(dd.min()), "sharpe_daily": float(daily.mean() / daily.std(ddof=1) * math.sqrt(365)) if len(daily) > 1 and daily.std(ddof=1) > 0 else np.nan,
        "trades": int(len(t)), "win_rate": float((t.net_pnl > 0).mean()) if len(t) else np.nan,
        "profit_factor": float(t.loc[t.net_pnl > 0, "net_pnl"].sum() / -t.loc[t.net_pnl < 0, "net_pnl"].sum()) if len(t) and (t.net_pnl < 0).any() else np.nan,
        "net_pnl": float(t.net_pnl.sum()) if len(t) else 0.0, "gross_price_pnl": gross, "funding_cashflow": funding,
        "net_pnl_including_funding": (float(t.net_pnl.sum()) if len(t) else 0.0) + funding,
        "fees": float(t.fee.sum()) if len(t) else 0.0, "turnover": turnover,
        "break_even_bp": gross / turnover * 10000 if turnover else np.nan,
        "median_hold_min": float(t.hold_minutes.median()) if len(t) else np.nan,
    }


def run() -> None:
    RESULTS.mkdir(exist_ok=True)
    open1, close1, volume1 = load_prices()
    funding_events = load_funding_events()
    features = feature_grid(close1, volume1)
    # Frozen before seeing May--August returns: 27 baseline configurations
    # (B0/B1/B2 × 5/15/60m × 1/4/12h).  Two predeclared PCA factor controls
    # bring the total to 29; center, scale, entry=2 and exit=.5 stay fixed.
    grid = [(m, f, 4, 7, 2.0, 0.5, h) for m in ("B0", "PEER", "PCA3") for f in (5, 15, 60) for h in (1, 4, 12)]
    grid += [("PCA1", 15, 4, 7, 2.0, 0.5, 4), ("PCA5", 15, 4, 7, 2.0, 0.5, 4)]
    rows, selected = [], []
    for n, (model, freq, center, scale, entry, exit_, hold) in enumerate(grid, 1):
        ft = features[(model, freq)]
        z = rolling_z(ft.level, freq, center, scale)
        eq, tr, meta = backtest(ft, z, open1, close1, entry, exit_, hold, 2.0, funding_events=funding_events)
        v = metrics(eq, tr, TRAIN_END, VALID_END, meta["funding_rows"]); q = metrics(eq, tr, VALID_END, END, meta["funding_rows"]); allm = metrics(eq, tr, START, END, meta["funding_rows"])
        row = {"model": model, "frequency_min": freq, "center_hours": center, "scale_days": scale, "entry_sigma": entry, "exit_sigma": exit_, "hold_hours": hold, **{f"valid_{k}": val for k, val in v.items()}, **{f"test_{k}": val for k, val in q.items()}, **{f"all_{k}": val for k, val in allm.items()}}
        rows.append(row)
        if n % 50 == 0: print(f"grid {n}/{len(grid)}", flush=True)
    table = pd.DataFrame(rows)
    table.sort_values(["valid_return", "valid_sharpe_daily"], ascending=False).to_csv(RESULTS / "v2_grid_2bp.csv", index=False)
    eligible = table[(table.valid_trades >= 30) & np.isfinite(table.valid_return)]
    primary = eligible.sort_values(["valid_return", "valid_sharpe_daily"], ascending=False).head(1) if len(eligible) else table.head(1)
    # Controls are fixed before looking at holdout returns: the equal-peer B1
    # and PCA3 B2 at the common 15m/4h baseline.
    controls = table[(table.model.isin(["PEER", "PCA3"])) & (table.frequency_min == 15) & (table.hold_hours == 4)]
    chosen = pd.concat([primary, controls]).drop_duplicates(subset=["model", "frequency_min", "center_hours", "scale_days", "entry_sigma", "exit_sigma", "hold_hours"])
    detail = []
    selected_configs = []
    for rank, (_, r) in enumerate(chosen.iterrows(), 1):
        ft = features[(r.model, int(r.frequency_min))]; z = rolling_z(ft.level, int(r.frequency_min), int(r.center_hours), int(r.scale_days))
        selected_configs.append({k: (int(v) if isinstance(v, (np.integer, int)) else float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items() if k in {"model", "frequency_min", "center_hours", "scale_days", "entry_sigma", "exit_sigma", "hold_hours"}})
        for bp in (0.0, 1.0, 2.0, 3.0):
            eq, tr, meta = backtest(ft, z, open1, close1, float(r.entry_sigma), float(r.exit_sigma), int(r.hold_hours), bp, funding_events=funding_events)
            if bp == 2.0:
                tr.to_csv(RESULTS / f"v2_trades_rank{rank}_2bp.csv", index=False)
                eq.rename("equity").to_frame().to_csv(RESULTS / f"v2_equity_rank{rank}_2bp.csv", index_label="time")
                (RESULTS / f"v2_ledger_rank{rank}_2bp.json").write_text(json.dumps(meta["ledger"], ensure_ascii=False, indent=2), encoding="utf-8")
            for name, a, b in (("validation", TRAIN_END, VALID_END), ("holdout", VALID_END, END)):
                detail.append({"selection_rank": rank, "model": r.model, "frequency_min": int(r.frequency_min), "center_hours": int(r.center_hours), "scale_days": int(r.scale_days), "entry_sigma": float(r.entry_sigma), "exit_sigma": float(r.exit_sigma), "hold_hours": int(r.hold_hours), "cost_bp": bp, "period": name, **metrics(eq, tr, a, b, meta["funding_rows"])})
    pd.DataFrame(detail).to_csv(RESULTS / "v2_selected_costs.csv", index=False)
    (RESULTS / "v2_selected_configs.json").write_text(json.dumps(selected_configs, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"symbols": SYMBOLS, "grid_rows": len(table), "selected_rows": len(detail), "selection": "rank 1 is the single highest May-June return at 2bp with >=30 validation trades; PEER/B1 and PCA3/B2 15m/4h are fixed controls", "train": str(START.date()) + "/" + str(TRAIN_END.date()), "validation": str(TRAIN_END.date()) + "/" + str(VALID_END.date()), "holdout": str(VALID_END.date()) + "/" + str(END.date()), "funding": "included from /fapi/v1/fundingRate markPrice and fundingRate; settlement cashflow is reported separately", "feature_models": ["B0", "PEER", "PCA1", "PCA3", "PCA5"], "grid_models": ["B0", "PEER", "PCA3", "PCA1", "PCA5"], "frequencies_min": [5, 15, 60], "grid_rows_predeclared": 29, "costs_bp": [0, 1, 2, 3], "all_in_cost_definition": "one-way bps charged on net aggregate order turnover; no separate slippage"}
    (RESULTS / "v2_run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
