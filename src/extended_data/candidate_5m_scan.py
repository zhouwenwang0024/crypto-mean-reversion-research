"""Causal 5-minute mean-reversion candidate screen.

This is a compact screening harness.  Every signal uses close[t-1] or earlier;
orders fill at open[t] and close at a later open.  It deliberately evaluates a
small fixed family across calendar splits, rather than selecting a single
profitable slice.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIRECT = ROOT / "data" / "extended_5m"
OLD = ROOT / "data" / "klines"
RESULTS = ROOT / "results"
SYMBOLS = "BTC ETH BNB SOL XRP DOGE ADA TRX LINK SUI AVAX LTC BCH DOT HBAR XLM FIL UNI NEAR AAVE".split()
STEP = pd.Timedelta(minutes=5)


def load() -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    frames = []
    months = pd.period_range("2024-02", "2026-02", freq="M").astype(str).tolist()
    tail = pd.period_range("2026-03", "2026-08", freq="M").astype(str).tolist()
    for s in SYMBOLS:
        rows = [pd.read_parquet(DIRECT / f"symbol={s}USDT" / f"month={m}.parquet") for m in months]
        for m in tail:
            p = OLD / f"symbol={s}USDT" / f"month={m}.parquet"
            x = pd.read_parquet(p)
            x.index = pd.to_datetime(x.open_time_utc_ms, unit="ms", utc=True)
            x = x.resample("5min", label="left", closed="left").agg(
                open=("open", "first"), close=("close", "last"),
                quote_volume=("quote_volume", "sum"),
            ).reset_index(names="timestamp")
            x["open_time_utc_ms"] = x.pop("timestamp").astype("int64") // 10**6
            rows.append(x[["open_time_utc_ms", "open", "close", "quote_volume"]])
        f = pd.concat(rows, ignore_index=True).sort_values("open_time_utc_ms")
        frames.append(f)
    idx = pd.DatetimeIndex(pd.to_datetime(frames[0].open_time_utc_ms, unit="ms", utc=True))
    op = np.column_stack([f.open.to_numpy(float) for f in frames])
    cl = np.column_stack([f.close.to_numpy(float) for f in frames])
    vol = np.column_stack([f.quote_volume.to_numpy(float) for f in frames])
    if any(not pd.DatetimeIndex(pd.to_datetime(f.open_time_utc_ms, unit="ms", utc=True)).equals(idx) for f in frames[1:]):
        raise ValueError("misaligned symbols")
    return idx, op, cl, vol


def summary(rets: np.ndarray, times: pd.DatetimeIndex, name: str, fee_bp: float = 5.0) -> dict:
    if not len(rets): return {"model": name, "trades": 0}
    formal = times >= pd.Timestamp("2024-03-01", tz="UTC")
    rets, times = rets[formal], times[formal]
    if not len(rets): return {"model": name, "trades": 0}
    net = rets - 2.0 * fee_bp / 10000.0
    eq = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    out = {"model": name, "trades": int(len(net)), "mean_gross_bp": float(rets.mean() * 1e4),
           "mean_net_bp": float(net.mean() * 1e4), "return": float(eq[-1] - 1.0),
           "mdd": float(np.min(eq / peak - 1.0)), "win_rate": float((net > 0).mean())}
    for p, a, b in (("development", "2024-03-01", "2025-03-01"),
                    ("validation", "2025-03-01", "2025-09-01"),
                    ("holdout", "2025-09-01", "2026-09-01"),
                    ("tail", "2026-03-01", "2026-09-01")):
        m = (times >= pd.Timestamp(a, tz="UTC")) & (times < pd.Timestamp(b, tz="UTC"))
        vals = net[m]
        out[p + "_trades"] = int(len(vals)); out[p + "_return"] = float(np.prod(1.0 + vals) - 1.0) if len(vals) else np.nan
        out[p + "_mean_net_bp"] = float(vals.mean() * 1e4) if len(vals) else np.nan
    return out


def cross_sectional(op: np.ndarray, cl: np.ndarray, idx: pd.DatetimeIndex, lookback_h: int,
                    hold_h: int, k: int, step_h: int, mode: str, min_spread: float = 0.0) -> tuple[np.ndarray, pd.DatetimeIndex]:
    n = len(cl); lb = lookback_h * 12; hold = hold_h * 12; stride = step_h * 12
    logp = np.log(cl); out = []; tt = []
    for t in range(max(lb + 1, 12), n - hold, stride):
        # Signal at the close immediately before entry; open[t] is the next fill.
        r = logp[t - 1] - logp[t - 1 - lb]
        if mode == "z":
            r = (r - np.median(r)) / (1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12)
        order = np.argsort(r); lo, hi = order[:k], order[-k:]
        if np.mean(r[hi]) - np.mean(r[lo]) < min_spread: continue
        fwd = np.log(op[t + hold]) - np.log(op[t])
        out.append(0.5 * (np.exp(fwd[lo]).mean() - np.exp(fwd[hi]).mean()))
        tt.append(idx[t])
    return np.asarray(out), pd.DatetimeIndex(tt)


def shock_reversal(op: np.ndarray, cl: np.ndarray, idx: pd.DatetimeIndex, lookback_b: int,
                   hold_b: int, k: int, min_spread: float = 0.0) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Short-horizon cross-sectional reversal after a signed shock."""
    lp = np.log(cl); out = []; tt = []
    for t in range(lookback_b + 1, len(cl) - hold_b, hold_b):
        r = lp[t - 1] - lp[t - 1 - lookback_b]; order = np.argsort(r); lo, hi = order[:k], order[-k:]
        if np.mean(r[hi]) - np.mean(r[lo]) < min_spread: continue
        fwd = np.log(op[t + hold_b]) - np.log(op[t]); out.append(0.5 * (np.exp(fwd[lo]).mean() - np.exp(fwd[hi]).mean())); tt.append(idx[t])
    return np.asarray(out), pd.DatetimeIndex(tt)


def single_reversal(op: np.ndarray, cl: np.ndarray, idx: pd.DatetimeIndex, lookback_h: int,
                    hold_h: int, j: int) -> tuple[np.ndarray, pd.DatetimeIndex]:
    lp = np.log(cl[:, j]); lb = lookback_h * 12; hold = hold_h * 12; out = []; tt = []
    for t in range(lb + 1, len(cl) - hold, hold):
        sig = lp[t - 1] - lp[t - 1 - lb]; fwd = np.log(op[t + hold, j]) - np.log(op[t, j]); out.append(-np.sign(sig) * (np.exp(fwd) - 1.0)); tt.append(idx[t])
    return np.asarray(out), pd.DatetimeIndex(tt)


def factor_residual(op: np.ndarray, cl: np.ndarray, idx: pd.DatetimeIndex, lookback_h: int,
                    hold_h: int, k: int, formation_h: int = 24 * 30) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Cross-sectional reversal of returns unexplained by a trailing BTC beta."""
    lp = np.log(cl); out = []; tt = []; lb = lookback_h * 12; hold = hold_h * 12; fw = formation_h * 12
    btc = np.diff(lp[:, 0]); stride = hold
    for t in range(max(lb + 1, fw + 2), len(cl) - hold, stride):
        x = btc[t - fw:t - 1]; xc = x - x.mean(); den = np.sum(xc * xc)
        if den <= 1e-14: continue
        betas = np.sum((np.diff(lp[t - fw:t], axis=0) - np.diff(lp[t - fw:t], axis=0).mean(0)) * xc[:, None], axis=0) / den
        r = (lp[t - 1] - lp[t - 1 - lb]) - betas * (lp[t - 1, 0] - lp[t - 1 - lb, 0])
        order = np.argsort(r); lo, hi = order[:k], order[-k:]; fwd = np.log(op[t + hold]) - np.log(op[t])
        out.append(.5 * (np.exp(fwd[lo]).mean() - np.exp(fwd[hi]).mean())); tt.append(idx[t])
    return np.asarray(out), pd.DatetimeIndex(tt)


def residual_rank(op: np.ndarray, cl: np.ndarray, idx: pd.DatetimeIndex, lookback_h: int,
                  hold_h: int, k: int, step_h: int) -> tuple[np.ndarray, pd.DatetimeIndex]:
    n = len(cl); lb = lookback_h * 12; hold = hold_h * 12; stride = step_h * 12
    lp = np.log(cl); out = []; tt = []
    # Residual against the contemporaneous cross-sectional median level.  The
    # median is taken only at the prior close, and each coin is ranked by its
    # residual change over the lookback.
    for t in range(lb + 1, n - hold, stride):
        past = lp[t - 1 - lb:t]
        med = np.median(past, axis=1)
        r = (lp[t - 1] - med[-1]) - (past[0] - med[0])
        order = np.argsort(r); lo, hi = order[:k], order[-k:]
        fwd = np.log(op[t + hold]) - np.log(op[t])
        out.append(0.5 * (np.exp(fwd[lo]).mean() - np.exp(fwd[hi]).mean())); tt.append(idx[t])
    return np.asarray(out), pd.DatetimeIndex(tt)


def pair_static(op: np.ndarray, cl: np.ndarray, idx: pd.DatetimeIndex) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Fixed formation top-five pairs, hourly z-score and causal exits.

    Pair identities and hedge ratios are frozen from the first 90 days.  This
    makes the result useful as a conservative control: no later pair choice is
    allowed to leak the evaluation path.
    """
    h = cl[::12]; hi = idx[::12]; form = hi < pd.Timestamp("2024-05-01", tz="UTC")
    lp = np.log(h); norm = lp - lp[form][0]; pairs = []
    for a in range(lp.shape[1]):
        for b in range(a + 1, lp.shape[1]):
            d = float(np.sum((norm[form, a] - norm[form, b]) ** 2)); pairs.append((d, a, b))
    pairs = sorted(pairs)[:5]
    events = []
    for _, a, b in pairs:
        f = lp[form]; x, y = f[:, b], f[:, a]
        beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)); alpha = float(y.mean() - beta * x.mean())
        s = lp[:, a] - beta * lp[:, b] - alpha; w = 7 * 24
        mu = pd.Series(s).rolling(w, min_periods=w).mean().shift(1).to_numpy()
        sd = pd.Series(s).rolling(w, min_periods=w).std(ddof=1).shift(1).to_numpy(); z = (s - mu) / sd
        pos = 0; ent = 0; ep = None; ez = np.nan
        for j in range(np.where(~form)[0][0], len(h) - 1):
            if pos == 0 and np.isfinite(z[j]) and abs(z[j]) >= 2.0:
                pos = -1 if z[j] > 0 else 1; ent = j + 1; ep = np.array([op[(j + 1) * 12, a], op[(j + 1) * 12, b]]); ez = z[j]
            elif pos and (abs(z[j]) <= 0.5 or j + 1 - ent >= 48):
                xp = np.array([op[(j + 1) * 12, a], op[(j + 1) * 12, b]])
                rr = pos * 0.5 * ((xp[0] / ep[0] - 1.0) - beta * (xp[1] / ep[1] - 1.0))
                events.append((hi[j + 1], rr)); pos = 0
        if pos:
            j = len(h) - 1; xp = np.array([cl[-1, a], cl[-1, b]])
            rr = pos * 0.5 * ((xp[0] / ep[0] - 1.0) - beta * (xp[1] / ep[1] - 1.0)); events.append((hi[j], rr))
    if not events: return np.array([]), pd.DatetimeIndex([])
    events.sort(key=lambda x: x[0]); times = pd.DatetimeIndex([x[0] for x in events]); vals = np.array([x[1] for x in events]) / max(1, len(pairs))
    return vals, times


def pair_dynamic(op: np.ndarray, cl: np.ndarray, idx: pd.DatetimeIndex, entry_z: float = 2.0,
                 exit_z: float = .5, max_hold: int = 48, score_mode: str = "diff") -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Monthly rolling formation: choose five short-half-life pairs causally."""
    lp = np.log(cl[::12]); oo = op[::12]; hh = idx[::12]; out = []
    months = pd.PeriodIndex(hh, freq="M").unique()
    for m in months:
        cur = np.where((hh.year == m.year) & (hh.month == m.month))[0]
        if not len(cur): continue
        lo = cur[0]; hist = np.arange(max(0, lo - 90 * 24), lo)
        if len(hist) < 60 * 24: continue
        cand = []
        for a in range(lp.shape[1]):
            for b in range(a + 1, lp.shape[1]):
                x, y = lp[hist, b], lp[hist, a]; beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
                s = y - beta * x; d = np.diff(s)
                rho = float(np.corrcoef(d[:-1], d[1:])[0, 1]) if len(d) > 2 else 1.0
                if score_mode == "level": rho = float(np.corrcoef(s[:-1], s[1:])[0, 1]) if len(s) > 2 else 1.0
                # Fast, stable spread changes are selected from history only.
                cand.append((rho, float(np.std(d)), a, b, beta))
        pairs = sorted(cand, key=lambda x: (x[0], x[1]))[:5]
        for _, _, a, b, beta in pairs:
            s = lp[:, a] - beta * lp[:, b]; w = 7 * 24
            mu = pd.Series(s).rolling(w, min_periods=w).mean().shift(1).to_numpy(); sd = pd.Series(s).rolling(w, min_periods=w).std(ddof=1).shift(1).to_numpy(); z = (s - mu) / sd
            pos = 0; ent = 0; ep = None
            for j in range(max(lo, w + 1), min(len(hh) - 1, cur[-1] + 1)):
                if pos == 0 and np.isfinite(z[j]) and abs(z[j]) >= entry_z:
                    pos = -1 if z[j] > 0 else 1; ent = j + 1; ep = np.array([oo[j + 1, a], oo[j + 1, b]])
                elif pos and (abs(z[j]) <= exit_z or j + 1 - ent >= max_hold):
                    xp = np.array([oo[j + 1, a], oo[j + 1, b]])
                    out.append((hh[j + 1], pos * .5 * ((xp[0] / ep[0] - 1) - beta * (xp[1] / ep[1] - 1)) / 5.0)); pos = 0
            if pos:
                xp = np.array([cl[cur[-1] * 12, a], cl[cur[-1] * 12, b]])
                out.append((hh[cur[-1]], pos * .5 * ((xp[0] / ep[0] - 1) - beta * (xp[1] / ep[1] - 1)) / 5.0))
    if not out: return np.array([]), pd.DatetimeIndex([])
    out.sort(key=lambda x: x[0]); return np.array([x[1] for x in out]), pd.DatetimeIndex([x[0] for x in out])


def run() -> None:
    idx, op, cl, vol = load(); rows = []
    specs = [
        ("cs_raw_1h_4h_k3", 1, 4, 3, 4, "raw"),
        ("cs_raw_4h_4h_k3", 4, 4, 3, 4, "raw"),
        ("cs_raw_24h_12h_k3", 24, 12, 3, 12, "raw"),
        ("cs_z_4h_4h_k3", 4, 4, 3, 4, "z"),
        ("cs_z_24h_12h_k3", 24, 12, 3, 12, "z"),
    ]
    for name, lb, hold, k, stride, mode in specs:
        r, t = cross_sectional(op, cl, idx, lb, hold, k, stride, mode); rows.append(summary(r, t, name))
    for lb, hold, stride, spread in ((4, 4, 4, .02), (4, 4, 4, .04), (24, 12, 12, .02),
                                     (24, 12, 12, .04), (24, 24, 24, .04)):
        r, t = cross_sectional(op, cl, idx, lb, hold, 3, stride, "raw", spread)
        rows.append(summary(r, t, f"cs_tail_{lb}h_{hold}h_spread{int(spread*100)}pct"))
    r, t = cross_sectional(op, cl, idx, 336, 672, 2, 672, "raw")
    rows.append(summary(0.9 * r, t, "cs_reversal_336h_672h_k2_budget90"))
    for lb, hold, k, spread in ((1, 3, 3, .002), (1, 6, 3, .002), (1, 12, 3, .002),
                                (3, 6, 3, .004), (3, 12, 3, .004)):
        r, t = shock_reversal(op, cl, idx, lb, hold, k, spread)
        rows.append(summary(r, t, f"shock_rev_{lb}x{hold}_k{k}_spread{int(spread*1e4)}bp"))
    for j in range(len(SYMBOLS)):
        r, t = single_reversal(op, cl, idx, 24, 48, j)
        rows.append(summary(r, t, f"single_rev_24h_48h_{SYMBOLS[j]}"))
    for lb, hold, k in ((4, 4, 3), (24, 12, 3), (24, 24, 2), (72, 24, 2)):
        r, t = factor_residual(op, cl, idx, lb, hold, k); rows.append(summary(r, t, f"btc_residual_{lb}h_{hold}h_k{k}"))
    for lb, hold, k in ((1, 4, 3), (4, 4, 3), (24, 12, 3), (4, 12, 5)):
        r, t = residual_rank(op, cl, idx, lb, hold, k, hold); rows.append(summary(r, t, f"resid_median_{lb}h_{hold}h_k{k}"))
    r, t = pair_static(op, cl, idx); rows.append(summary(r, t, "pair_static_top5"))
    r, t = pair_dynamic(op, cl, idx); rows.append(summary(r, t, "pair_dynamic_monthly_top5", fee_bp=1.0))
    for ez, xz, mh in ((2.5, .5, 48), (3.0, .5, 48), (2.0, .25, 72), (2.5, .25, 72)):
        r, t = pair_dynamic(op, cl, idx, ez, xz, mh)
        rows.append(summary(r, t, f"pair_dynamic_e{ez}_x{xz}_h{mh}", fee_bp=1.0))
    for ez, xz, mh in ((2.0, .25, 72), (2.5, .25, 72), (2.0, .5, 48)):
        r, t = pair_dynamic(op, cl, idx, ez, xz, mh, "level")
        rows.append(summary(r, t, f"pair_dynamic_level_e{ez}_x{xz}_h{mh}", fee_bp=1.0))
    out = pd.DataFrame(rows); out.to_csv(RESULTS / "candidate_5m_scan.csv", index=False)
    (RESULTS / "candidate_5m_scan_manifest.json").write_text(json.dumps({"cost_bp_one_way": 5.0, "models": [x[0] for x in specs], "signal": "close[t-1]", "entry": "open[t]", "tail_is_reused": True}, indent=2), encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__": run()
