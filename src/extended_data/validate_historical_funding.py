"""Validate the funding relative-value filters on the 2022-02--2024-01 sample.

This is deliberately a separate loader: the primary 20-symbol 2024--2026
archive is unchanged.  The historical universe is discovered from the common
5m and funding files (SUI, and any contract without a complete event grid, is
excluded).  Build
``data/historical_5m_2022-02_to_2024-01.zip`` from the downloaded parquet
partitions before running this module; the parquet directory itself is also
accepted for convenience.
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from funding_entry_exit_experiments import run_policy
    import funding_entry_exit_experiments as experiment
except ModuleNotFoundError:  # package import from repository root
    from .funding_entry_exit_experiments import run_policy
    from . import funding_entry_exit_experiments as experiment

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ARCHIVE_CANDIDATES = (
    ROOT / "data" / "historical_5m_2022-02_to_2024-01.zip",
    ROOT / "data" / "historical_5m_2022-02_to_2024-02.zip",
    ROOT / "data" / "historical_5m",
)
FUNDING_DIR = ROOT / "data" / "historical_funding"
SYMBOLS = "BTC ETH BNB SOL XRP DOGE ADA TRX LINK AVAX LTC BCH DOT HBAR XLM FIL UNI NEAR AAVE".split()
MONTHS = pd.period_range("2022-02", "2024-01", freq="M").astype(str).tolist()
START = pd.Timestamp("2022-02-01", tz="UTC")
END = pd.Timestamp("2024-02-01", tz="UTC")
PERIODS = {
    "historical_2022_2023": ("2022-02-01", "2023-02-01"),
    "historical_2023_2024": ("2023-02-01", "2024-02-01"),
}
STEP_MS = 300_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source() -> Path:
    for path in ARCHIVE_CANDIDATES:
        if path.is_file() or (path.is_dir() and any(path.glob("symbol=*/month=*.parquet"))):
            return path
    discovered = sorted((ROOT / "data").glob("historical_5m_*.zip"))
    if discovered:
        return discovered[0]
    names = ", ".join(str(path.relative_to(ROOT)) for path in ARCHIVE_CANDIDATES)
    raise FileNotFoundError(f"historical 5m source not found; expected one of: {names}")


def _read_part(source: Path, symbol: str, month: str) -> pd.DataFrame:
    name = f"symbol={symbol}USDT/month={month}.parquet"
    if source.is_dir():
        frame = pd.read_parquet(source / name)
    else:
        with zipfile.ZipFile(source) as archive:
            frame = pd.read_parquet(BytesIO(archive.read(name)))
    frame = frame[["open_time_utc_ms", "open", "close"]].copy()
    frame["open_time_utc_ms"] = pd.to_numeric(frame.open_time_utc_ms, errors="raise").astype("int64")
    frame[["open", "close"]] = frame[["open", "close"]].apply(pd.to_numeric, errors="raise")
    expected = pd.Period(month).days_in_month * 288
    first = int(pd.Timestamp(month + "-01", tz="UTC").timestamp() * 1000)
    stamps = frame.open_time_utc_ms.to_numpy()
    if len(frame) != expected or not np.array_equal(stamps, first + STEP_MS * np.arange(expected, dtype="int64")):
        raise ValueError(f"invalid 5m grid: {name}")
    if not np.isfinite(frame[["open", "close"]]).all().all() or (frame[["open", "close"]] <= 0).any().any():
        raise ValueError(f"invalid prices: {name}")
    return frame


def _available_kline(source: Path) -> tuple[set[str], set[str]]:
    if source.is_dir():
        names = [p.relative_to(source).as_posix() for p in source.glob("symbol=*/month=*.parquet")]
    else:
        with zipfile.ZipFile(source) as archive:
            names = archive.namelist()
    symbols: set[str] = set(); months: set[str] = set()
    for name in names:
        parts = name.split("/")
        if len(parts) == 2 and parts[0].startswith("symbol=") and parts[1].startswith("month="):
            symbols.add(parts[0].split("=", 1)[1].removesuffix("USDT"))
            months.add(parts[1].split("=", 1)[1].removesuffix(".parquet"))
    return symbols, months


def load_prices(source: Path) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    frames: list[pd.DataFrame] = []
    for symbol in SYMBOLS:
        parts = [_read_part(source, symbol, month) for month in MONTHS]
        frames.append(pd.concat(parts, ignore_index=True))
    index = pd.to_datetime(frames[0].open_time_utc_ms.to_numpy(), unit="ms", utc=True)
    for frame in frames[1:]:
        other = pd.to_datetime(frame.open_time_utc_ms.to_numpy(), unit="ms", utc=True)
        if not index.equals(other):
            raise ValueError("historical symbols do not share an identical 5m grid")
    return index, np.column_stack([f.open.to_numpy(float) for f in frames]), np.column_stack(
        [f.close.to_numpy(float) for f in frames]
    )


def _funding_path(symbol: str) -> Path:
    candidates = (FUNDING_DIR / f"symbol={symbol}USDT.parquet", FUNDING_DIR / f"{symbol}USDT.parquet")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"funding parquet missing for {symbol}USDT under {FUNDING_DIR}")


def load_funding(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    rate = np.full((len(index), len(SYMBOLS)), np.nan)
    mark = np.full_like(rate, np.nan)
    index_ns = index.view("i8")
    slot_sets: list[set[int]] = []
    start_ms = int(START.timestamp() * 1000); end_ms = int(END.timestamp() * 1000)
    for j, symbol in enumerate(SYMBOLS):
        frame = pd.read_parquet(_funding_path(symbol))
        frame = frame.rename(columns={"fundingTime": "funding_time_utc_ms", "fundingRate": "funding_rate", "markPrice": "mark_price"})
        frame = frame[["funding_time_utc_ms", "funding_rate", "mark_price"]].copy()
        for column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        frame = frame[(frame.funding_time_utc_ms >= start_ms) & (frame.funding_time_utc_ms < end_ms)]
        slots: set[int] = set()
        for row in frame.itertuples(index=False):
            target = int(row.funding_time_utc_ms) * 1_000_000
            right = int(np.searchsorted(index_ns, target)); candidates = [max(0, right - 1), min(len(index_ns) - 1, right)]
            slot = min(candidates, key=lambda x: abs(int(index_ns[x]) - target))
            if abs(int(index[slot].value // 1_000_000) - int(row.funding_time_utc_ms)) > 2_000:
                raise ValueError(f"funding event is not on the 5m grid: {symbol}")
            if slot in slots:
                raise ValueError(f"duplicate funding event: {symbol} {row.funding_time_utc_ms}")
            slots.add(slot); rate[slot, j] = float(row.funding_rate); mark[slot, j] = float(row.mark_price)
        if not slots:
            raise ValueError(f"no funding rows in historical range: {symbol}")
        slot_sets.append(slots)
    if any(slots != slot_sets[0] for slots in slot_sets[1:]):
        raise ValueError("historical funding symbols do not share an identical event grid")
    slots = np.array(sorted(slot_sets[0]), dtype=int)
    if len(slots) < 2 or not np.array_equal(slots, np.arange(slots[0], slots[-1] + 96, 96, dtype=int)):
        raise ValueError("historical funding event grid has gaps")
    return rate, mark


def _run(name: str, policy: dict, index: pd.DatetimeIndex, op: np.ndarray,
         close: np.ndarray, rates: np.ndarray, marks: np.ndarray) -> dict:
    row, bar, orders, trades = run_policy(index, op, close, rates, marks, 0, len(index), policy)
    prefix = RESULTS / f"historical_funding_{name}"
    bar.to_csv(prefix.with_name(prefix.name + "_equity.csv"), index=False)
    orders.to_csv(prefix.with_name(prefix.name + "_orders.csv"), index=False)
    trades.to_csv(prefix.with_name(prefix.name + "_trades.csv"), index=False)
    if abs(float(row["replay_error"])) > 1e-8:
        raise AssertionError(f"replay error: {row}")
    return row


def main() -> None:
    global SYMBOLS, MONTHS, START, END, PERIODS
    source = _source()
    available_symbols, available_months = _available_kline(source)
    funding_symbols = {
        path.name.split("=", 1)[-1].removesuffix("USDT.parquet")
        for path in FUNDING_DIR.glob("symbol=*.parquet")
    }
    common = [symbol for symbol in SYMBOLS if symbol in available_symbols and symbol in funding_symbols]
    if len(common) < 2:
        raise ValueError(f"fewer than two common historical symbols: {common}")
    available_months = sorted(available_months & set(MONTHS))
    if not available_months:
        raise ValueError("historical source has no months in the requested 2022-02--2024-01 window")
    # Support either the original 2022-02--2024-01 download or a later
    # two-year window produced by the downloader, without changing the main
    # sample.  Every symbol is still required to contain every selected month.
    SYMBOLS = common
    MONTHS = available_months
    START = pd.Timestamp(MONTHS[0] + "-01", tz="UTC")
    END = (pd.Period(MONTHS[-1], freq="M") + 1).start_time.tz_localize("UTC")
    split = max(1, len(MONTHS) // 2)
    split_time = pd.Timestamp(MONTHS[split] + "-01", tz="UTC") if split < len(MONTHS) else END
    PERIODS = {"historical_first_half": (START.strftime("%Y-%m-%d"), split_time.strftime("%Y-%m-%d")),
               "historical_second_half": (split_time.strftime("%Y-%m-%d"), END.strftime("%Y-%m-%d"))}
    index, op, close = load_prices(source)
    rates, marks = load_funding(index)
    # run_policy references these module globals; patching is scoped to this
    # standalone historical audit and leaves the primary experiment untouched.
    experiment.SYMBOLS = SYMBOLS.copy()
    experiment.PERIODS = PERIODS
    policies = [
        {"name": "baseline", "persistence": 1, "spread_threshold": 0.0},
        {"name": "cost1_persist3", "persistence": 3, "spread_threshold": experiment.COST_COVER_THRESHOLD},
    ]
    rows = [_run(p["name"], p, index, op, close, rates, marks) for p in policies]
    summary = pd.DataFrame(rows)
    out = RESULTS / "historical_funding_2022_2024.csv"
    summary.to_csv(out, index=False)
    manifest = {
        "sample_start": START.isoformat(), "sample_end_exclusive": END.isoformat(),
        "source": str(source.relative_to(ROOT)), "source_sha256": _sha256(source) if source.is_file() else None,
        "funding_dir": str(FUNDING_DIR.relative_to(ROOT)), "symbols": [s + "USDT" for s in SYMBOLS],
        "excluded": {symbol + "USDT": "missing complete common kline/funding files"
                     for symbol in (set(SYMBOLS) ^ set("BTC ETH BNB SOL XRP DOGE ADA TRX LINK AVAX LTC BCH DOT HBAR XLM FIL UNI NEAR AAVE".split()))},
        "fee_bp_one_way": 5.0, "policies": policies, "rows": rows,
    }
    (RESULTS / "historical_funding_2022_2024_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    lines = ["# 历史资金费率扩展验证", "", f"固定单边手续费 5bp；样本使用 {len(SYMBOLS)} 个共同合约，按文件可用性排除缺失合约。", "",
             "|策略|CAGR|MDD|交易数|资金费收益|手续费|", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"|{row['policy']}|{row['cagr']:.2%}|{row['mdd']:.2%}|{int(row['trades'])}|{row['funding']:.2%}|{row['fees']:.2%}|")
    lines += ["", "该脚本只用于历史扩展审计；它不改变主 20 符号回测的输入或结果。", ""]
    (ROOT / "REPORT_HISTORICAL_FUNDING_2022_2024_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary[["policy", "cagr", "mdd", "trades", "funding", "fees"]].to_string(index=False))


if __name__ == "__main__":
    main()
