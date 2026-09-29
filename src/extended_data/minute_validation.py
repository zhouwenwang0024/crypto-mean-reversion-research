"""Causal minute-clock validation on the disjoint two-year sample.

This runner keeps the handoff strategy's one-minute centre, daily fit and
five-minute observation clock.  It uses the audited ``engine.simulate6`` so
fills, net order fees, dynamic exits and terminal liquidation share one ledger.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rolling_ridge_audit.models import fit_pca, fit_ridge  # noqa: E402
from rolling_ridge_audit.engine import simulate6  # noqa: E402

DATA = ROOT / "data" / "extended_1m"
RESULTS = ROOT / "results"
SYMBOLS = "BTC ETH BNB SOL XRP DOGE ADA TRX LINK SUI AVAX LTC BCH DOT HBAR XLM FIL UNI NEAR AAVE".split()
P = len(SYMBOLS)
PERIODS = {"development": ("2024-03-01", "2025-03-01"),
           "validation": ("2025-03-01", "2025-09-01"),
           "holdout": ("2025-09-01", "2026-03-01")}
MODELS = ("ridge_sma3", "pca2_sma3", "pca3_sma3", "pca5_sma3", "ridge_recent_hourly", "ridge_huber")


def load_data() -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    frames = []
    for symbol in SYMBOLS:
        chunks = [pd.read_parquet(p, columns=["open_time_utc_ms", "open", "close", "quote_volume"])
                  for p in sorted((DATA / f"symbol={symbol}USDT").glob("month=*.parquet"))]
        frame = pd.concat(chunks, ignore_index=True)
        frames.append(frame)
    expected = frames[0].open_time_utc_ms.to_numpy(np.int64)
    if not np.array_equal(expected, expected[0] + 60_000 * np.arange(len(expected), dtype=np.int64)):
        raise ValueError("minute grid has gaps")
    for frame in frames[1:]:
        if not np.array_equal(frame.open_time_utc_ms.to_numpy(np.int64), expected):
            raise ValueError("symbols do not share one minute grid")
    op = np.column_stack([f.open.to_numpy(float) for f in frames])
    cl = np.column_stack([f.close.to_numpy(float) for f in frames])
    vol = np.column_stack([f.quote_volume.to_numpy(float) for f in frames])
    if not np.isfinite(cl).all() or (cl <= 0).any() or (vol < 0).any():
        raise ValueError("invalid minute prices or volume")
    return pd.to_datetime(expected, unit="ms", utc=True), op, cl, vol


def fit_history(lc: np.ndarray, model: str, d: int) -> np.ndarray:
    ds = d * 1440
    points = ds - 1 - np.arange(28 * 24, -1, -1, dtype=np.int64) * 60
    returns = np.diff(lc[points], axis=0)
    if model.startswith("pca"):
        return fit_pca(returns, int(model[3]))[0]
    method = {"ridge_recent_hourly": "weighted", "ridge_huber": "huber"}.get(model, "ridge")
    return fit_ridge(returns, method=method, penalty=0.1)[0]


def features(lc: np.ndarray, model: str) -> dict[str, np.ndarray]:
    n = len(lc); days = (n + 1439) // 1440
    pref = np.vstack([np.zeros((1, P)), np.cumsum(lc, axis=0)])
    centre = np.full_like(lc, np.nan)
    centre[180:] = (pref[180:n] - pref[:n - 180]) / 180.0
    delta = lc - centre
    dev = np.full((n + 1, P), np.nan); projected = np.full_like(dev, np.nan)
    scale = np.full_like(dev, np.nan); W = np.full((days + 1, P, P), np.nan)
    for d in range(29, days):
        ds = d * 1440; w = fit_history(lc, model, d); W[d] = w
        hi = min(ds + 1440, n + 1); src = np.arange(ds, hi) - 1
        projected[ds:hi] = centre[src] @ w.T; dev[ds:hi] = delta[src] @ w.T
        past = delta[ds - 7 * 1440:ds] @ w.T; scale[ds:hi] = np.std(past, axis=0, ddof=1)
    return {"dev": dev, "center": projected, "scale": scale, "W": W}


def load_funding(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, int]:
    cash = np.zeros((len(index) + 1, P)); present = np.zeros_like(cash, dtype=bool); rows = 0
    start_ms = int(index[0].value // 10**6)
    for j, symbol in enumerate(SYMBOLS):
        frame = pd.read_parquet(ROOT / "data" / "extended_funding" / f"symbol={symbol}USDT.parquet")
        for row in frame.itertuples(index=False):
            k = int(round((int(row.funding_time_utc_ms) - start_ms) / 60_000))
            if k < 0 or k >= len(cash) or abs(start_ms + k * 60_000 - int(row.funding_time_utc_ms)) > 1_000:
                raise ValueError("funding timestamp is off the minute grid")
            cash[k, j] = -float(row.funding_rate) * float(row.mark_price); present[k, j] = True; rows += 1
    if rows != P * 2277 or not np.isfinite(cash).all(): raise ValueError(f"funding coverage failure: {rows}")
    return cash, present, rows


def digest() -> str:
    h = hashlib.sha256()
    for path in sorted(DATA.rglob("*.parquet")):
        h.update(path.relative_to(DATA).as_posix().encode())
        with path.open("rb") as stream:
            chunk_hash = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""): chunk_hash.update(chunk)
        h.update(chunk_hash.digest())
    return h.hexdigest()


def run(op, cl, vol, feat, index, start: int, end: int, cost_bp: float, fund=None,
        include_funding=False, structure: int = 0, entry_delay: int = 1, signal_stride: int = 1):
    n = len(cl); ends = np.arange(15, n, 5 * signal_stride, dtype=np.int64)
    prefix = np.vstack([np.zeros((1, P)), np.cumsum(vol, axis=0)])
    five_volume = prefix[ends] - prefix[ends - 5]
    valid = np.isfinite(five_volume).all(1) & (five_volume > 0).all(1)
    day = ends // 1440
    zero_fund = np.zeros_like(cl) if fund is None else fund
    # Numba's dispatcher is strict about keyword arguments on this compiled
    # function, so keep the audited call signature positional after ``end``.
    return simulate6(op, cl, np.log(cl), zero_fund, ends, day,
                     np.ascontiguousarray(feat["W"]), np.ascontiguousarray(feat["dev"][ends]),
                     np.ascontiguousarray(feat["center"][ends]), np.ascontiguousarray(feat["scale"][ends]),
                     valid, start, end, 2.0, 0.015, structure, 8, 0.5, 240, 25.0, cost_bp, include_funding,
                     None, True, 0.03, 0, None, 0.0, 0.0, 0, 50.0, 1.0, None, 60, entry_delay,
                     feat["dev"], feat["scale"], None, 25.0)


def summarize(a, index: pd.DatetimeIndex, start: int, end: int, model: str, period: str, bp: float, funding: bool):
    eq, turnover, gross, net, trades, _ = a; peak = np.maximum.accumulate(eq); ret = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    fee = float(trades[:, 8].sum()); funding_cash = float(trades[:, 9].sum())
    return {"model": model, "period": period, "cost_bp": bp, "funding": funding,
            "return": float(eq[-1] - 1), "mdd": float(np.min(eq / peak - 1)), "trades": int(len(trades)),
            "fees": fee, "price_pnl": float(trades[:, 7].sum()), "funding_cash": funding_cash,
            "turnover": float(turnover.sum()), "mean_gross_fraction": float(gross.mean()),
            "max_gross_fraction": float(gross.max()), "mean_net_fraction": float(net.mean()),
            "max_abs_net_fraction": float(np.max(np.abs(net))),
            "annualized_vol": float(np.std(ret, ddof=1) * np.sqrt(365.25 * 1440)) if len(ret) > 1 else np.nan,
            "sharpe": float(np.mean(ret) / np.std(ret, ddof=1) * np.sqrt(365.25 * 1440)) if len(ret) > 1 and np.std(ret, ddof=1) > 0 else np.nan,
            "reconciliation_error": float(eq[-1] - 1 - (trades[:, 7].sum() - fee + funding_cash)),
            "start": str(index[start]), "end": str(index[end - 1])}


def main() -> None:
    index, op, cl, vol = load_data(); lc = np.log(cl); funding, present, funding_rows = load_funding(index)
    results = []; stability = []; features_cache = {}
    for model in MODELS:
        feat = features(lc, model); features_cache[model] = feat
        valid_w = feat["W"][np.isfinite(feat["W"]).all((1, 2))]
        stability.append({"model": model, "daily_fits": len(valid_w),
                          "mean_weight_l1_change": float(np.mean(np.abs(np.diff(valid_w, axis=0)))) if len(valid_w) > 1 else 0.0,
                          "median_max_peer": float(np.nanmedian(np.nanmax(np.abs(valid_w - np.eye(P)), axis=(1, 2))))})
        for period, (a, b) in PERIODS.items():
            start, end = int(index.searchsorted(pd.Timestamp(a, tz="UTC"))), int(index.searchsorted(pd.Timestamp(b, tz="UTC")))
            for bp in (0.0, 2.0, 5.0, 10.0):
                account = run(op, cl, vol, feat, index, start, end, bp)
                row = summarize(account, index, start, end, model, period, bp, False)
                if model == "ridge_sma3" and period == "holdout" and bp == 5.0:
                    names = ["entry_bar", "exit_bar", "target", "other", "z", "gap", "gross_entry", "pnl", "fee", "funding", "reason", "min_pnl_frac", "max_pnl_frac", "entry_z", "signal_bar", "entry_center", "turnover", "signed_entry"]
                    pd.DataFrame(account[4], columns=names + [f"q_{s}" for s in SYMBOLS] + [f"w_{s}" for s in SYMBOLS]).to_csv(RESULTS / "extended_minute_ridge_holdout_5bp_trades.csv", index=False)
                results.append(row)
    # Funding is reported separately at 5 bp on the frozen holdout.
    start, end = [int(index.searchsorted(pd.Timestamp(x, tz="UTC"))) for x in PERIODS["holdout"]]
    for model, feat in features_cache.items():
        row = summarize(run(op, cl, vol, feat, index, start, end, 5.0, funding, True), index, start, end, model, "holdout", 5.0, True)
        row["funding_rows"] = funding_rows; row["funding_timestamps"] = int(present.sum()); results.append(row)
    out = pd.DataFrame(results); out.to_csv(RESULTS / "extended_minute_model_results.csv", index=False)
    pd.DataFrame(stability).to_csv(RESULTS / "extended_minute_model_stability.csv", index=False)
    manifest = {"source": "data/extended_1m, native 1m execution", "data_digest": digest(),
                "new_sample": "2024-03-01/2026-03-01", "warmup": "2024-01/2024-02",
                "periods": PERIODS, "models": list(MODELS), "cost_bp": [0, 2, 5, 10],
                "observation_minutes": 5, "centre_minutes": 180, "fit_days": 28,
                "scale_days": 7, "entry_z": 2.0, "entry_gap": 0.015, "exit_gap": 0.0025,
                "max_hold_minutes": 240, "stop_fraction": 0.03,
                "funding_rows": funding_rows, "selection": "frozen before holdout"}
    (RESULTS / "extended_minute_validation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__": main()
