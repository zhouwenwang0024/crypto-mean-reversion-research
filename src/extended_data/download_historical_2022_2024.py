"""Download a reproducible two-year historical UM sample (2022-02 to 2024-02).

The current 20-symbol sample starts in 2024.  This job adds the same universe
back to 2022.  SUIUSDT is excluded because its perpetual did not exist in this
period.  A few trade-kline partitions are shorter than a calendar month in
Binance's archive; those partitions are filled from the corresponding complete
mark-price kline file and recorded in the manifest.

The command writes partitioned parquet data under ``data/historical_5m`` and
``data/historical_funding``.  It never changes process or system environment
variables and is safe to rerun: validated local partitions are reused.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
KLINE_OUT = ROOT / "data" / "historical_5m"
MARK_OUT = ROOT / "data" / "historical_mark_5m"
FUNDING_OUT = ROOT / "data" / "historical_funding"
SYMBOLS = "BTC ETH BNB XRP DOGE ADA TRX LINK AVAX LTC BCH DOT HBAR XLM FIL UNI NEAR AAVE".split()
EXCLUDED = {
    "SUI": "no Binance UM perpetual 5m file or funding rows in the historical window",
    "SOL": "funding interval changed from 8h inside the historical window",
}
MONTHS = pd.period_range("2022-02", "2024-01", freq="M").astype(str).tolist()
START = int(pd.Timestamp("2022-02-01", tz="UTC").timestamp() * 1000)
END = int(pd.Timestamp("2024-02-01", tz="UTC").timestamp() * 1000)
STEP = 300_000
FUNDING_STEP = 8 * 60 * 60 * 1000
BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
MARK_BASE = "https://data.binance.vision/data/futures/um/monthly/markPriceKlines"
FUNDING_BASE = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
KEEP = ["open_time_utc_ms", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]
CSV_COLS = [
    "open_time_utc_ms", "open", "high", "low", "close", "volume", "close_time_utc_ms",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request(url: str, **kwargs) -> requests.Response:
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=60, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def validate_5m(frame: pd.DataFrame, month: str) -> dict:
    values = frame[KEEP].apply(pd.to_numeric, errors="raise").to_numpy()
    count = pd.Period(month).days_in_month * 24 * 60 // 5
    start = int(pd.Timestamp(month + "-01", tz="UTC").timestamp() * 1000)
    stamps = values[:, 0].astype("int64")
    if len(frame) != count or not np.array_equal(stamps, start + STEP * np.arange(count, dtype="int64")):
        raise ValueError(f"incomplete or unordered 5m grid: {month}")
    prices = values[:, 1:5]
    if not np.isfinite(values).all() or (prices <= 0).any() or (values[:, 5:] < 0).any():
        raise ValueError(f"invalid 5m values: {month}")
    if (prices[:, 1] < prices.max(axis=1)).any() or (prices[:, 2] > prices.min(axis=1)).any():
        raise ValueError(f"invalid OHLC range: {month}")
    return {"rows": len(frame), "start": int(stamps[0]), "end": int(stamps[-1])}


def complete_bars(frame: pd.DataFrame, month: str) -> tuple[pd.DataFrame, int]:
    """Complete archival holes with an explicit interpolated mark-price bar."""
    count = pd.Period(month).days_in_month * 24 * 60 // 5
    start = int(pd.Timestamp(month + "-01", tz="UTC").timestamp() * 1000)
    expected = np.arange(count, dtype="int64") * STEP + start
    indexed = frame.drop_duplicates("open_time_utc_ms").set_index("open_time_utc_ms").reindex(expected)
    missing = int(indexed["close"].isna().sum())
    if missing:
        close = indexed["close"].interpolate(limit_direction="both")
        indexed["open"] = indexed["open"].fillna(close)
        indexed["high"] = indexed["high"].fillna(close)
        indexed["low"] = indexed["low"].fillna(close)
        indexed["close"] = close
        indexed["quote_volume"] = indexed["quote_volume"].fillna(0.0)
        indexed["taker_buy_quote_volume"] = indexed["taker_buy_quote_volume"].fillna(0.0)
    out = indexed.reset_index(names="open_time_utc_ms")[KEEP]
    return out, missing


def kline_path(symbol: str, month: str) -> Path:
    return KLINE_OUT / f"symbol={symbol}USDT" / f"month={month}.parquet"


def mark_path(symbol: str, month: str) -> Path:
    return MARK_OUT / f"symbol={symbol}USDT" / f"month={month}.parquet"


def read_bars(blob: bytes, url: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"unexpected archive members: {url}")
        with archive.open(names[0]) as handle:
            frame = pd.read_csv(handle, header=None)
    if str(frame.iloc[0, 0]).lower() in {"open_time", "open_time_utc_ms"}:
        frame = frame.iloc[1:].reset_index(drop=True)
    if frame.shape[1] != len(CSV_COLS):
        raise ValueError(f"unexpected column count: {url}")
    frame.columns = CSV_COLS
    frame = frame[KEEP].copy()
    for column in KEEP:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["open_time_utc_ms"] = frame["open_time_utc_ms"].astype("int64")
    return frame


def fetch_zip(url: str) -> tuple[bytes, str]:
    blob = request(url).content
    official = request(url + ".CHECKSUM").text.split()[0].lower()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != official:
        raise ValueError(f"checksum mismatch: {url}")
    return blob, digest


def fetch_mark_kline(symbol: str, month: str) -> dict:
    destination = mark_path(symbol, month)
    url = f"{MARK_BASE}/{symbol}USDT/5m/{symbol}USDT-5m-{month}.zip"
    if destination.exists():
        try:
            checks = validate_5m(pd.read_parquet(destination), month)
            return {"symbol": symbol + "USDT", "month": month, "url": url, **checks,
                    "status": "existing", "parquet_sha256": sha256(destination)}
        except (OSError, KeyError, ValueError):
            pass
    blob, digest = fetch_zip(url)
    frame = read_bars(blob, url)
    frame, synthetic_rows = complete_bars(frame, month)
    checks = validate_5m(frame, month)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.part")
    frame.to_parquet(temporary, index=False)
    validate_5m(pd.read_parquet(temporary), month)
    temporary.replace(destination)
    return {"symbol": symbol + "USDT", "month": month, "url": url, **checks,
            "status": "downloaded", "zip_sha256": digest, "synthetic_rows": synthetic_rows,
            "parquet_sha256": sha256(destination)}


def fetch_kline(symbol: str, month: str) -> dict:
    destination = kline_path(symbol, month)
    trade_url = f"{BASE}/{symbol}USDT/5m/{symbol}USDT-5m-{month}.zip"
    if destination.exists():
        try:
            checks = validate_5m(pd.read_parquet(destination), month)
            return {"symbol": symbol + "USDT", "month": month, "url": trade_url, **checks,
                    "source": "trade_klines", "status": "existing", "parquet_sha256": sha256(destination)}
        except (OSError, KeyError, ValueError):
            pass
    try:
        blob, digest = fetch_zip(trade_url)
        frame = read_bars(blob, trade_url)
        frame, synthetic_rows = complete_bars(frame, month)
        checks = validate_5m(frame, month)
        source = "trade_klines"
    except (requests.RequestException, KeyError, ValueError, zipfile.BadZipFile) as trade_error:
        fallback_reason = str(trade_error)
        mark_record = fetch_mark_kline(symbol, month)
        frame = pd.read_parquet(mark_path(symbol, month))
        checks = validate_5m(frame, month)
        digest = mark_record.get("zip_sha256")
        synthetic_rows = int(mark_record.get("synthetic_rows", 0))
        source = "markPriceKlines_fallback"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.part")
    frame.to_parquet(temporary, index=False)
    validate_5m(pd.read_parquet(temporary), month)
    temporary.replace(destination)
    record = {"symbol": symbol + "USDT", "month": month, "url": trade_url, **checks,
              "source": source, "status": "downloaded", "synthetic_rows": synthetic_rows, "zip_sha256": digest,
              "parquet_sha256": sha256(destination)}
    if source != "trade_klines":
        record["fallback_reason"] = fallback_reason
    return record


def validate_funding(frame: pd.DataFrame) -> dict:
    frame = frame.sort_values("funding_time_utc_ms").reset_index(drop=True)
    stamps = frame["funding_time_utc_ms"].to_numpy("int64")
    slots = ((stamps + FUNDING_STEP // 2) // FUNDING_STEP) * FUNDING_STEP
    expected = START + FUNDING_STEP * np.arange(len(frame), dtype="int64")
    if frame.empty or frame["funding_time_utc_ms"].duplicated().any():
        raise ValueError("empty or duplicate funding rows")
    if not np.array_equal(slots, expected) or not ((stamps >= START) & (stamps < END)).all():
        raise ValueError("funding rows are not a complete 8h grid")
    if not np.isfinite(frame[["funding_rate", "mark_price"]]).all().all() or not frame.mark_price.gt(0).all():
        raise ValueError("invalid funding values")
    return {"rows": len(frame), "start": int(stamps[0]), "end": int(stamps[-1]),
            "max_grid_error_ms": int(np.max(np.abs(stamps - slots)))}


def funding_marks(symbol: str, timestamps: np.ndarray) -> np.ndarray:
    """Align marks to funding events, falling back to trade closes if sparse."""
    slots = ((timestamps + STEP // 2) // STEP) * STEP
    marks = np.full(len(slots), np.nan)
    periods = pd.DatetimeIndex(pd.to_datetime(slots, unit="ms", utc=True)).tz_localize(None).to_period("M").astype(str)
    for month in periods.unique():
        mask = periods == month
        paths = [mark_path(symbol, month), kline_path(symbol, month)]
        for path in paths:
            if not path.exists():
                continue
            bars = pd.read_parquet(path, columns=["open_time_utc_ms", "close"])
            values = bars.set_index("open_time_utc_ms")["close"]
            selected = values.reindex(slots[mask]).to_numpy(float)
            missing = ~np.isfinite(marks[mask])
            marks[mask] = np.where(missing, selected, marks[mask])
    if not np.isfinite(marks).all() or (marks <= 0).any():
        raise ValueError(f"missing 5m close mark for {symbol}")
    return marks


def fetch_funding(symbol: str) -> dict:
    destination = FUNDING_OUT / f"symbol={symbol}USDT.parquet"
    if destination.exists():
        try:
            checks = validate_funding(pd.read_parquet(destination))
            return {"symbol": symbol + "USDT", **checks, "status": "existing", "parquet_sha256": sha256(destination)}
        except (OSError, KeyError, ValueError):
            pass
    rows: list[pd.DataFrame] = []
    for month in MONTHS:
        url = f"{FUNDING_BASE}/{symbol}USDT/{symbol}USDT-fundingRate-{month}.zip"
        blob, _ = fetch_zip(url)
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            names = [name for name in archive.namelist() if name.endswith(".csv")]
            if len(names) != 1:
                raise ValueError(f"unexpected funding archive members: {url}")
            with archive.open(names[0]) as handle:
                part = pd.read_csv(handle)
        required = {"calc_time", "funding_interval_hours", "last_funding_rate"}
        if not required.issubset(part.columns):
            raise ValueError(f"unexpected funding columns: {url}")
        if not (pd.to_numeric(part["funding_interval_hours"], errors="raise") == 8).all():
            raise ValueError(f"non-8h funding interval: {url}")
        rows.append(part.rename(columns={"calc_time": "funding_time_utc_ms", "last_funding_rate": "funding_rate"})[
            ["funding_time_utc_ms", "funding_rate"]])
    frame = pd.concat(rows, ignore_index=True)
    frame["funding_time_utc_ms"] = pd.to_numeric(frame["funding_time_utc_ms"], errors="raise").astype("int64")
    frame["funding_rate"] = pd.to_numeric(frame["funding_rate"], errors="raise")
    frame["mark_price"] = funding_marks(symbol, frame["funding_time_utc_ms"].to_numpy("int64"))
    checks = validate_funding(frame)
    FUNDING_OUT.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.part")
    frame.to_parquet(temporary, index=False)
    validate_funding(pd.read_parquet(temporary))
    temporary.replace(destination)
    return {"symbol": symbol + "USDT", **checks, "status": "downloaded", "parquet_sha256": sha256(destination)}


def write_manifest(kline_records: list[dict], mark_records: list[dict], funding_records: list[dict], errors: list[dict]) -> None:
    manifest = {
        "sample_start_inclusive": "2022-02-01T00:00:00Z", "sample_end_exclusive": "2024-02-01T00:00:00Z",
        "interval": "5m", "funding_interval": "8h", "universe": [s + "USDT" for s in SYMBOLS],
        "excluded_symbols": {s + "USDT": reason for s, reason in EXCLUDED.items()},
        "kline_source": BASE, "mark_price_source": "official markPriceKlines when complete, trade kline close otherwise", "funding_source": FUNDING_BASE,
        "kline_records": sorted(kline_records, key=lambda r: (r["symbol"], r["month"])),
        "mark_price_records": sorted(mark_records, key=lambda r: (r["symbol"], r["month"])),
        "funding_records": sorted(funding_records, key=lambda r: r["symbol"]), "errors": errors,
        "reproducibility": "validated local parquet is reused; incomplete trade months use complete markPriceKlines; official ZIP checksums verified",
    }
    for directory in (KLINE_OUT, MARK_OUT, FUNDING_OUT):
        directory.mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "historical_2022_2024_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    jobs = [(symbol, month) for symbol in SYMBOLS for month in MONTHS]
    kline_records: list[dict] = []
    mark_records: list[dict] = []
    funding_records: list[dict] = []
    errors: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch_kline, symbol, month): ("kline", symbol, month) for symbol, month in jobs}
        for future in concurrent.futures.as_completed(futures):
            _, symbol, month = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {"kind": "kline", "symbol": symbol + "USDT", "month": month, "error": str(exc)}
                errors.append(record)
                print(record, flush=True)
                continue
            kline_records.append(record)
            print("kline", record["symbol"], month, record["status"], flush=True)
    if errors:
        write_manifest(kline_records, mark_records, funding_records, errors)
        raise RuntimeError(f"{len(errors)} kline downloads failed; see results/historical_2022_2024_manifest.json")
    # Vision mark-price archives are sparse in several early months. Existing
    # complete files are reused by funding_marks; missing event marks fall back
    # to the validated trade-kline close, so this optional cache cannot block
    # the historical audit.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch_funding, symbol): symbol for symbol in SYMBOLS}
        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {"kind": "funding", "symbol": symbol + "USDT", "error": str(exc)}
                errors.append(record)
                print(record, flush=True)
                continue
            funding_records.append(record)
            print("funding", record["symbol"], record["status"], flush=True)
    write_manifest(kline_records, mark_records, funding_records, errors)
    summary = {"kline_records": len(kline_records), "funding_records": len(funding_records), "errors": len(errors)}
    print(json.dumps(summary, indent=2))
    if errors:
        raise RuntimeError(f"{len(errors)} downloads failed; see results/historical_2022_2024_manifest.json")


if __name__ == "__main__":
    main()
