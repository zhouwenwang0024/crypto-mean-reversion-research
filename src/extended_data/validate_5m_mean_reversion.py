"""Causal low-turnover cross-sectional mean-reversion on the complete 5m lake.

Signals are frozen at a completed bar, filled on the next bar open, and
charged against the signed net order.  The primary rule is a 14-day
cross-sectional reversal held for 28 days; neighboring predeclared rules are
reported for robustness rather than selected from the final tail.
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
STEP_MS = 300_000
FEE_BP = 5.0
GROSS_FRACTION = 0.90
PERIODS = {
    "development": ("2024-03-01", "2025-03-01"),
    "validation": ("2025-03-01", "2025-09-01"),
    "historical_holdout": ("2025-09-01", "2026-03-01"),
    "extension": ("2026-03-01", "2026-09-01"),
}
RULES = {
    "xs_14d_28d_k2": (14, 28, 2),
    "xs_7d_14d_k2": (7, 14, 2),
    "xs_14d_14d_k2": (14, 14, 2),
    "xs_28d_28d_k2": (28, 28, 2),
    "xs_14d_42d_k2": (14, 42, 2),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prices() -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    months = pd.period_range("2024-02", "2026-08", freq="M").astype(str)
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for symbol in SYMBOLS:
            parts = []
            for month in months:
                name = f"symbol={symbol}USDT/month={month}.parquet"
                part = pd.read_parquet(BytesIO(archive.read(name)), columns=[
                    "open_time_utc_ms", "open", "close", "quote_volume"
                ])
                expected = pd.Period(month).days_in_month * 288
                if len(part) != expected:
                    raise ValueError(f"row count mismatch: {name}")
                parts.append(part)
            frame = pd.concat(parts, ignore_index=True).sort_values("open_time_utc_ms")
            frames.append(frame)
    idx = pd.to_datetime(frames[0].open_time_utc_ms.to_numpy(), unit="ms", utc=True)
    expected_rows = sum(pd.Period(month).days_in_month * 288 for month in months)
    if len(idx) != expected_rows or not np.all(np.diff(idx.view("i8")) == STEP_MS * 1_000_000):
        raise ValueError("invalid 5-minute grid")
    for frame in frames[1:]:
        other = pd.to_datetime(frame.open_time_utc_ms.to_numpy(), unit="ms", utc=True)
        if not idx.equals(other):
            raise ValueError("symbol grids are not identical")
    op = np.column_stack([x.open.to_numpy(float) for x in frames])
    cl = np.column_stack([x.close.to_numpy(float) for x in frames])
    vol = np.column_stack([x.quote_volume.to_numpy(float) for x in frames])
    if not np.isfinite(op).all() or not np.isfinite(cl).all() or (op <= 0).any() or (cl <= 0).any() or (vol < 0).any():
        raise ValueError("invalid prices or volumes")
    return idx, op, cl, vol


def load_funding(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, int]:
    """Combine the existing 2024-02/2026-02 and 2026-03/08 funding files.

    Missing tail funding is an error; it is never silently treated as zero.
    """
    rate = np.full((len(index), len(SYMBOLS)), np.nan)
    mark = np.full_like(rate, np.nan)
    events = 0
    index_ns = index.view("i8")
    slot_sets = []
    for j, symbol in enumerate(SYMBOLS):
        paths = [ROOT / "data" / "extended_funding" / f"symbol={symbol}USDT.parquet",
                 ROOT / "results" / "funding_api" / f"{symbol}USDT.parquet"]
        parts = [pd.read_parquet(p) for p in paths if p.exists()]
        if not parts:
            raise FileNotFoundError(f"funding missing for {symbol}")
        frame = pd.concat(parts, ignore_index=True).drop_duplicates("funding_time_utc_ms")
        slots = []
        for row in frame.itertuples(index=False):
            raw = int(row.funding_time_utc_ms)
            target = raw * 1_000_000
            right = int(np.searchsorted(index_ns, target))
            candidates = [max(0, right - 1), min(len(index_ns) - 1, right)]
            k = min(candidates, key=lambda x: abs(int(index_ns[x]) - target))
            if abs(int(index[k].value // 1_000_000) - raw) > 2_000:
                raise ValueError(f"funding event not on 5m grid: {symbol} {raw}")
            if np.isfinite(rate[k, j]):
                raise ValueError(f"duplicate funding slot: {symbol} {raw}")
            rate[k, j] = float(row.funding_rate)
            mark[k, j] = float(row.mark_price)
            slots.append(k)
            events += 1
        slot_sets.append(set(slots))
    if not slot_sets or any(x != slot_sets[0] for x in slot_sets[1:]):
        raise ValueError("funding symbols do not share the same event grid")
    slots = np.array(sorted(slot_sets[0]), dtype=int)
    expected = np.arange(slots[0], slots[-1] + 1, 96, dtype=int)
    if not np.array_equal(slots, expected):
        raise ValueError("funding event grid has gaps")
    if not np.isfinite(rate[~np.isnan(rate)]).all() or not np.isfinite(mark[~np.isnan(mark)]).all():
        raise ValueError("invalid funding values")
    return rate, mark, events


def signal_legs(cl: np.ndarray, t: int, lookback_days: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    bars = lookback_days * 288
    r = np.log(cl[t - 1]) - np.log(cl[t - 1 - bars])
    order = np.argsort(r)
    return order[:k], order[-k:]


def run_rule(index: pd.DatetimeIndex, op: np.ndarray, cl: np.ndarray,
             funding_rate: np.ndarray, funding_mark: np.ndarray,
             rule: tuple[int, int, int], start: int, end: int,
             cost_bp: float = FEE_BP, funding: bool = True,
             phase_shift_bars: int = 0, delay_bars: int = 0) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    lookback_days, hold_days, k = rule
    lookback = lookback_days * 288; hold = hold_days * 288
    # Anchor the rebalance clock to the archive warm-up. Filtering this
    # global schedule by each reporting interval avoids phase selection at a
    # period boundary.
    signal_times = [t for t in range(lookback + 1 + phase_shift_bars, len(cl) - hold - delay_bars, hold)
                    if start <= t < end - hold - delay_bars]
    entry_set = {t + delay_bars: t for t in signal_times}
    exit_set = {t + delay_bars + hold: t for t in signal_times}
    cash = 1.0; funding_cash = 0.0; q: np.ndarray | None = None; entry_px = None; entry_time = None
    rows = []; orders = []; equity = []; fee_rate = cost_bp / 10_000.0

    def fee_for(delta: np.ndarray, px: np.ndarray) -> float:
        return float(np.abs(delta * px).sum() * fee_rate)

    for t in range(start, end):
        px_open = op[t]
        # Funding is settled before any fill at this timestamp.
        if funding and np.isfinite(funding_rate[t]).any() and q is not None:
            valid = np.isfinite(funding_rate[t]) & np.isfinite(funding_mark[t])
            flow = float(np.sum(-q[valid] * funding_mark[t, valid] * funding_rate[t, valid]))
            cash += flow; funding_cash += flow
        if t in exit_set and q is not None:
            delta = -q; fee = fee_for(delta, px_open)
            pnl = float(np.sum(q * (px_open - entry_px)))
            cash += pnl - fee
            orders.append({"time": str(index[t]), "type": "exit", "turnover": float(np.abs(delta * px_open).sum()), "fee": fee})
            rows.append({"entry_time": str(index[entry_time]), "exit_time": str(index[t]),
                         "gross_pnl": pnl, "fee": float(fee + entry_fee),
                         "net_pnl": float(pnl - fee - entry_fee), "hold_days": hold_days,
                         "long": ",".join(SYMBOLS[x] for x in long_ids),
                         "short": ",".join(SYMBOLS[x] for x in short_ids)})
            q = None; entry_px = None; entry_time = None
        if t in entry_set:
            if q is not None:
                raise AssertionError("overlapping positions")
            signal_t = entry_set[t]
            long_ids, short_ids = signal_legs(cl, signal_t, lookback_days, k)
            mark_px = cl[t - 1]
            eq = cash
            budget = GROSS_FRACTION * eq
            q = np.zeros(len(SYMBOLS))
            q[long_ids] = budget / 2 / k / mark_px[long_ids]
            q[short_ids] = -budget / 2 / k / mark_px[short_ids]
            entry_px = px_open.copy(); entry_time = t
            entry_fee = fee_for(q, px_open)
            cash -= entry_fee
            orders.append({"time": str(index[t]), "type": "entry", "turnover": float(np.abs(q * px_open).sum()), "fee": entry_fee,
                           "long": ",".join(SYMBOLS[x] for x in long_ids), "short": ",".join(SYMBOLS[x] for x in short_ids)})
        marked = cash if q is None else cash + float(np.sum(q * (cl[t] - entry_px)))
        equity.append({"time": index[t], "equity": marked, "cash": cash,
                       "gross_exposure": 0.0 if q is None else float(np.abs(q * cl[t]).sum()),
                       "net_exposure": 0.0 if q is None else float(np.sum(q * cl[t]))})
    if q is not None:
        t = end - 1; delta = -q; fee = fee_for(delta, cl[t]); pnl = float(np.sum(q * (cl[t] - entry_px))); cash += pnl - fee
        orders.append({"time": str(index[t]), "type": "forced_exit", "turnover": float(np.abs(delta * cl[t]).sum()), "fee": fee})
        rows.append({"entry_time": str(index[entry_time]), "exit_time": str(index[t]), "gross_pnl": pnl,
                     "fee": float(fee + entry_fee), "net_pnl": float(pnl - fee - entry_fee), "hold_days": hold_days,
                     "long": ",".join(SYMBOLS[x] for x in long_ids), "short": ",".join(SYMBOLS[x] for x in short_ids)})
        equity[-1]["equity"] = cash; equity[-1]["cash"] = cash; equity[-1]["gross_exposure"] = 0.0; equity[-1]["net_exposure"] = 0.0
    eq = pd.DataFrame(equity).set_index("time")
    tr = pd.DataFrame(rows); od = pd.DataFrame(orders)
    curve = eq.equity.to_numpy(float); peak = np.maximum.accumulate(np.r_[1.0, curve])[1:]
    net = tr.net_pnl.to_numpy(float) if len(tr) else np.array([])
    out = {"return": float(cash - 1.0), "mdd": float(np.min(curve / peak - 1.0)), "trades": int(len(tr)),
           "fees": float(tr.fee.sum()) if len(tr) else 0.0, "gross_pnl": float(tr.gross_pnl.sum()) if len(tr) else 0.0,
           "funding": float(funding_cash) if funding else 0.0,
           "mean_trade_net_bp": float(net.mean() * 1e4) if len(net) else np.nan,
           "win_rate": float((net > 0).mean()) if len(net) else np.nan,
           "turnover": float(od.turnover.sum()) if len(od) else 0.0,
           "reconciliation_error": float(cash - 1.0 - ((tr.gross_pnl.sum() - tr.fee.sum()) if len(tr) else 0.0) - funding_cash)}
    return out, tr, eq.reset_index()


def main() -> None:
    index, op, cl, _ = load_prices(); funding_rate, funding_mark, funding_events = load_funding(index)
    rows = []; primary_trades = None; primary_equity = None
    for name, rule in RULES.items():
        for period, (a, b) in PERIODS.items():
            start, end = int(index.searchsorted(pd.Timestamp(a, tz="UTC"))), int(index.searchsorted(pd.Timestamp(b, tz="UTC")))
            summary, trades, equity = run_rule(index, op, cl, funding_rate, funding_mark, rule, start, end, funding=True)
            summary.update({"rule": name, "period": period, "start": a, "end": b, "cost_bp_one_way": FEE_BP, "funding_events": funding_events})
            rows.append(summary)
            if name == "xs_14d_28d_k2":
                trades.to_csv(RESULTS / f"5m_{name}_{period}_5bp_trades.csv", index=False)
                equity.to_csv(RESULTS / f"5m_{name}_{period}_5bp_equity.csv", index=False)
        start = int(index.searchsorted(pd.Timestamp("2024-03-01", tz="UTC")))
        end = int(index.searchsorted(pd.Timestamp("2026-09-01", tz="UTC")))
        summary, trades, equity = run_rule(index, op, cl, funding_rate, funding_mark, rule, start, end, funding=True)
        summary.update({"rule": name, "period": "full_sample", "start": "2024-03-01", "end": "2026-09-01",
                        "cost_bp_one_way": FEE_BP, "funding_events": funding_events})
        rows.append(summary)
        if name == "xs_14d_28d_k2":
            trades.to_csv(RESULTS / "5m_xs_14d_28d_k2_full_sample_5bp_trades.csv", index=False)
            equity.to_csv(RESULTS / "5m_xs_14d_28d_k2_full_sample_5bp_equity.csv", index=False)
    out = pd.DataFrame(rows); out.to_csv(RESULTS / "5m_mean_reversion_results.csv", index=False)
    full_start = int(index.searchsorted(pd.Timestamp("2024-03-01", tz="UTC")))
    full_end = int(index.searchsorted(pd.Timestamp("2026-09-01", tz="UTC")))
    sensitivity = []
    for label, phase, delay in (("phase_0", 0, 0), ("phase_1bar", 1, 0),
                                ("phase_1h", 12, 0), ("phase_1d", 288, 0),
                                ("delay_1bar", 0, 1), ("delay_2bar", 0, 2)):
        s, _, _ = run_rule(index, op, cl, funding_rate, funding_mark, RULES["xs_14d_28d_k2"],
                            full_start, full_end, funding=True,
                            phase_shift_bars=phase, delay_bars=delay)
        s.update({"variant": label, "phase_shift_bars": phase, "delay_bars": delay})
        sensitivity.append(s)
    pd.DataFrame(sensitivity).to_csv(RESULTS / "5m_mean_reversion_sensitivity.csv", index=False)
    costs = []
    for bp in (0.0, 2.0, 5.0, 10.0):
        s, _, _ = run_rule(index, op, cl, funding_rate, funding_mark, RULES["xs_14d_28d_k2"],
                            full_start, full_end, cost_bp=bp, funding=True)
        s.update({"cost_bp_one_way": bp})
        costs.append(s)
    pd.DataFrame(costs).to_csv(RESULTS / "5m_mean_reversion_costs.csv", index=False)
    probe_t = full_start + 1_000
    baseline = signal_legs(cl, probe_t, 14, 2)
    perturbed = cl.copy(); perturbed[probe_t:] *= 1.5
    future_probe = signal_legs(perturbed, probe_t, 14, 2)
    checks = {
        "archive_grid_rows": int(len(index)),
        "funding_events": int(funding_events),
        "future_suffix_prefix_unchanged": bool(all(np.array_equal(a, b) for a, b in zip(baseline, future_probe))),
        "max_reconciliation_error": float(np.max(np.abs(out.reconciliation_error))),
        "cost_return_nonincreasing": bool(np.all(np.diff([x["return"] for x in costs]) <= 1e-12)),
    }
    (RESULTS / "5m_mean_reversion_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    manifest = {"archive": str(ARCHIVE.relative_to(ROOT)).replace("\\", "/"), "archive_sha256": _sha256(ARCHIVE),
                "bar_interval": "5m", "signal": "close[t-1] trailing cross-sectional log return", "entry": "next bar open",
                "exit": "next bar open after fixed hold; terminal close forced", "cost_bp_one_way": FEE_BP,
                "gross_fraction": GROSS_FRACTION, "funding": "extended_funding + funding_api; missing is an error",
                "funding_events": funding_events, "periods": PERIODS, "rules": RULES,
                "selection": "primary rule fixed as 14d lookback/28d hold/k2; neighboring rules are robustness checks",
                "clock_sensitivity": ["phase 0/1 bar/1 hour/1 day", "entry delay 1/2 bars"],
                "cost_sensitivity": [0, 2, 5, 10]}
    (RESULTS / "5m_mean_reversion_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
