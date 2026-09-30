"""Causal short-horizon reversal tests motivated by recent crypto research.

The rule is frozen before reading the result tail: a completed bar's return
sets the opposite position at the next bar open, held for exactly one bar.
The script tests horizons and three predeclared position rules, includes
funding at settlement timestamps, and reports calendar splits and costs.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "data" / "combined_5m_2024-02_to_2026-08.zip"
RESULTS = ROOT / "results"
SYMBOLS = "BTC ETH BNB SOL XRP DOGE ADA TRX LINK SUI AVAX LTC BCH DOT HBAR XLM FIL UNI NEAR AAVE".split()
P = len(SYMBOLS)
PERIODS = {
    "development": ("2024-03-01", "2025-03-01"),
    "validation": ("2025-03-01", "2025-09-01"),
    "historical_holdout": ("2025-09-01", "2026-03-01"),
    "extension": ("2026-03-01", "2026-09-01"),
}


def load_5m() -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    months = pd.period_range("2024-02", "2026-08", freq="M").astype(str)
    frames = []
    with zipfile.ZipFile(ARCHIVE) as z:
        for symbol in SYMBOLS:
            parts = [pd.read_parquet(BytesIO(z.read(f"symbol={symbol}USDT/month={m}.parquet")),
                                     columns=["open_time_utc_ms", "open", "close", "quote_volume"])
                     for m in months]
            frame = pd.concat(parts, ignore_index=True).sort_values("open_time_utc_ms")
            frames.append(frame)
    stamp = frames[0].open_time_utc_ms.to_numpy(np.int64)
    if not np.array_equal(stamp, stamp[0] + 300_000 * np.arange(len(stamp), dtype=np.int64)):
        raise ValueError("5m grid has gaps")
    for frame in frames[1:]:
        if not np.array_equal(frame.open_time_utc_ms.to_numpy(np.int64), stamp):
            raise ValueError("symbols are misaligned")
    return (pd.to_datetime(stamp, unit="ms", utc=True),
            np.column_stack([f.open.to_numpy(float) for f in frames]),
            np.column_stack([f.close.to_numpy(float) for f in frames]),
            np.column_stack([f.quote_volume.to_numpy(float) for f in frames]))


def load_funding(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    rate = np.full((len(index), P), np.nan); mark = np.full_like(rate, np.nan)
    index_ms = index.view("i8") // 1_000_000
    for j, symbol in enumerate(SYMBOLS):
        paths = [ROOT / "data" / "extended_funding" / f"symbol={symbol}USDT.parquet",
                 ROOT / "results" / "funding_api" / f"{symbol}USDT.parquet"]
        frames = [pd.read_parquet(p, columns=["funding_time_utc_ms", "funding_rate", "mark_price"])
                  for p in paths if p.exists()]
        if not frames:
            raise FileNotFoundError(f"funding missing: {symbol}")
        frame = pd.concat(frames, ignore_index=True).drop_duplicates("funding_time_utc_ms")
        for row in frame.itertuples(index=False):
            raw = int(row.funding_time_utc_ms)
            right = int(np.searchsorted(index_ms, raw)); candidates = [max(0, right - 1), min(len(index_ms) - 1, right)]
            i = min(candidates, key=lambda k: abs(int(index_ms[k]) - raw))
            if abs(int(index_ms[i]) - raw) > 2_000:
                raise ValueError(f"funding event off 5m grid: {symbol} {row.funding_time_utc_ms}")
            rate[i, j] = float(row.funding_rate); mark[i, j] = float(row.mark_price)
    if not np.isfinite(rate[np.isfinite(rate)]).all() or not np.isfinite(mark[np.isfinite(mark)]).all():
        raise ValueError("invalid funding values")
    return rate, mark


def weights(prev: np.ndarray, rule: str, valid: np.ndarray) -> np.ndarray:
    w = np.zeros(P)
    if rule == "directional_sign":
        active = np.flatnonzero((prev != 0) & valid)
        if len(active): w[active] = -np.sign(prev[active]) * (0.90 / len(active))
    elif rule == "directional_top30":
        active = np.flatnonzero(valid); n = min(max(1, P * 3 // 10), len(active))
        if n == 0: return w
        active = active[np.argsort(np.abs(prev[active]))[-n:]]
        w[active] = -np.sign(prev[active]) * (0.90 / n)
    elif rule == "cross_sectional_5x5":
        active = np.flatnonzero(valid); k = min(5, len(active) // 2)
        if k == 0: return w
        order = active[np.argsort(prev[active])]
        w[order[:k]] = 0.90 / (2 * k); w[order[-k:]] = -0.90 / (2 * k)
    else:
        raise ValueError(rule)
    return w


def run(index: pd.DatetimeIndex, op: np.ndarray, cl: np.ndarray, vol: np.ndarray,
        funding_rate: np.ndarray, funding_mark: np.ndarray,
        minutes: int, rule: str, start: int, end: int, cost_bp: float) -> tuple[dict, pd.DataFrame]:
    step = minutes // 5; oo = op[::step]; cc = cl[step - 1::step]; vv = vol[::step]; rr = funding_rate[::step]; mm = funding_mark[::step]
    ii = index[::step]; a = int(np.searchsorted(ii, index[start])); b = int(np.searchsorted(ii, index[end - 1], side="right"))
    equity = 1.0; records = []
    for t in range(max(1, a), min(b - 1, len(ii) - 1)):
        prev = cc[t - 1] / oo[t - 1] - 1.0
        valid = (vv[t - 1] > 0) & (vv[t] > 0)
        w = weights(prev, rule, valid)
        future = oo[t + 1] / oo[t] - 1.0
        gross = float(w @ future)
        fee = float(np.abs(w).sum()) * 2.0 * cost_bp / 10_000.0
        fund = 0.0
        valid = np.isfinite(rr[t + 1]) & np.isfinite(mm[t + 1])
        if valid.any():
            fund = float(np.sum(-w[valid] * rr[t + 1, valid] * mm[t + 1, valid] / oo[t, valid]))
        net = gross - fee + fund
        records.append({"time": ii[t], "gross_bp": gross * 1e4, "fee_bp": fee * 1e4,
                        "funding_bp": fund * 1e4, "net_bp": net * 1e4,
                        "active": int(np.count_nonzero(w)), "turnover_fraction": float(2 * np.abs(w).sum()),
                        "prev_abs_bp": float(np.mean(np.abs(prev)) * 1e4)})
        equity *= 1.0 + net
    out = pd.DataFrame(records)
    curve = np.r_[1.0, (1.0 + out.net_bp.to_numpy(float) / 1e4).cumprod()] if len(out) else np.array([1.0])
    peak = np.maximum.accumulate(curve)
    summary = {"horizon_min": minutes, "rule": rule, "cost_bp_one_way": cost_bp,
               "start": str(index[start]), "end": str(index[end - 1]), "bars": len(out),
               "return": float(curve[-1] - 1), "mdd": float(np.min(curve / peak - 1)),
               "mean_gross_bp": float(out.gross_bp.mean()) if len(out) else np.nan,
               "mean_net_bp": float(out.net_bp.mean()) if len(out) else np.nan,
               "win_rate": float((out.net_bp > 0).mean()) if len(out) else np.nan,
               "funding_bp": float(out.funding_bp.sum()) if len(out) else 0.0,
               "turnover_fraction": float(out.turnover_fraction.mean()) if len(out) else 0.0}
    return summary, out


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def main() -> None:
    index, op, cl, vol = load_5m(); rate, mark = load_funding(index); rows = []
    for minutes in (5, 15, 30, 60):
        for rule in ("directional_sign", "directional_top30", "cross_sectional_5x5"):
            for period, (lo, hi) in PERIODS.items():
                start = int(index.searchsorted(pd.Timestamp(lo, tz="UTC"))); end = int(index.searchsorted(pd.Timestamp(hi, tz="UTC")))
                for bp in (0.0, 2.0, 5.0, 10.0):
                    s, bars = run(index, op, cl, vol, rate, mark, minutes, rule, start, end, bp)
                    s["period"] = period; rows.append(s)
                    if bp == 5.0 and period == "historical_holdout":
                        bars.to_csv(RESULTS / f"short_reversal_{minutes}m_{rule}_historical_holdout_5bp.csv", index=False)
    out = pd.DataFrame(rows); out.to_csv(RESULTS / "short_horizon_reversal_results.csv", index=False)
    checks = {"archive_sha256": digest(ARCHIVE), "grid_rows_5m": len(index),
              "zero_quote_volume_5m_cells": int(np.count_nonzero(vol == 0)),
              "funding_events": int(np.isfinite(rate).sum()),
              "horizons_min": [5, 15, 30, 60], "rules": ["directional_sign", "directional_top30", "cross_sectional_5x5"],
              "signal": "previous completed bar open-to-close return", "entry": "next bar open",
              "exit": "following bar open", "costs_one_way_bp": [0, 2, 5, 10],
              "zero_volume_policy": "exclude assets when signal or entry bar quote_volume is zero",
              "selection": "all horizons, rules, periods and costs fixed before reading results"}
    (RESULTS / "short_horizon_reversal_manifest.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(out[out.cost_bp_one_way == 5.0].to_string(index=False))


if __name__ == "__main__":
    main()
