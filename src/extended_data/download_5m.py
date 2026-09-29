"""Download a disjoint two-year Binance UM kline sample.

The existing audit data is 1m data for 2026-03 through 2026-08.  This
downloader requests only 2024-01 through 2026-02 (January and February are warmup),
skips validated local targets, and writes parquet partitions for the same
20-symbol universe.  ``INTERVAL`` and ``STEP_MS`` select 1m or 5m output.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "extended_5m"
INTERVAL = "5m"
STEP_MS = 300_000
SYMBOLS = "BTC ETH BNB SOL XRP DOGE ADA TRX LINK SUI AVAX LTC BCH DOT HBAR XLM FIL UNI NEAR AAVE".split()
MONTHS = pd.period_range("2024-01", "2026-02", freq="M").astype(str).tolist()
BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
COLS = [
    "open_time_utc_ms", "open", "high", "low", "close", "volume",
    "close_time_utc_ms", "quote_volume", "trades", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore",
]
KEEP = ["open_time_utc_ms", "open", "high", "low", "close",
        "quote_volume", "taker_buy_quote_volume"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def request(url: str, **kwargs) -> requests.Response:
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=30, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def validate_partition(frame: pd.DataFrame, month: str, step_ms: int) -> dict:
    values = frame[KEEP].apply(pd.to_numeric, errors="raise").to_numpy()
    rows = int(pd.Period(month).days_in_month * 86_400_000 // step_ms)
    start = pd.Timestamp(month + "-01", tz="UTC").value // 10**6
    if len(frame) != rows or not np.array_equal(values[:, 0], start + step_ms * np.arange(rows, dtype="int64")):
        raise ValueError(f"incomplete or unordered {month} timestamp grid")
    prices = values[:, 1:5]
    if not np.isfinite(values).all() or (prices <= 0).any() or (values[:, 5:] < 0).any():
        raise ValueError(f"non-finite/nonpositive prices or invalid volume: {month}")
    if ((prices[:, 1] < prices.max(axis=1)) | (prices[:, 2] > prices.min(axis=1))).any():
        raise ValueError(f"invalid OHLC range: {month}")
    if (values[:, 6] > values[:, 5] + 1e-8 + values[:, 5] * 1e-10).any():
        raise ValueError(f"taker volume exceeds quote volume: {month}")
    return {"rows": len(frame), "start": int(values[0, 0]), "end": int(values[-1, 0]),
            "zero_quote_volume_rows": int((values[:, 5] == 0).sum())}


def target(symbol: str, month: str) -> Path:
    return OUT / f"symbol={symbol}USDT" / f"month={month}.parquet"


def fetch(symbol: str, month: str, previous: dict | None = None) -> dict:
    dest = target(symbol, month)
    url = f"{BASE}/{symbol}USDT/{INTERVAL}/{symbol}USDT-{INTERVAL}-{month}.zip"
    record = {**(previous or {}), "symbol": symbol + "USDT", "month": month, "url": url}
    if dest.exists():
        try:
            checks = validate_partition(pd.read_parquet(dest), month, STEP_MS)
        except (ValueError, KeyError, OSError) as exc:
            record["replaced_invalid_cache"] = str(exc)
        else:
            return {**record, **checks, "status": "existing", "parquet_sha256": file_sha256(dest)}
    blob = request(url).content
    checksum = request(url + ".CHECKSUM").text.split()[0].lower()
    zip_sha = hashlib.sha256(blob).hexdigest()
    if zip_sha != checksum:
        raise ValueError(f"official checksum mismatch: {url}")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = [n for n in archive.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"unexpected members for {url}: {archive.namelist()}")
        with archive.open(names[0]) as handle:
            frame = pd.read_csv(handle, header=None)
    if str(frame.iloc[0, 0]).lower() in {"open_time", "open_time_utc_ms"}:
        frame = frame.iloc[1:].reset_index(drop=True)
    if frame.shape[1] != len(COLS):
        raise ValueError(f"unexpected columns for {url}: {frame.shape}")
    frame.columns = COLS
    for column in KEEP[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame[KEEP]
    frame["open_time_utc_ms"] = pd.to_numeric(frame["open_time_utc_ms"], errors="raise").astype("int64")
    checks = validate_partition(frame, month, STEP_MS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".part")
    frame.to_parquet(temporary, index=False)
    validate_partition(pd.read_parquet(temporary), month, STEP_MS)
    temporary.replace(dest)
    return {**record, **checks, "status": "downloaded", "bytes": len(blob), "sha256": zip_sha,
            "official_sha256": checksum, "parquet_sha256": file_sha256(dest)}


def main() -> None:
    jobs = [(s, m) for s in SYMBOLS for m in MONTHS]
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "download_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    records = {(r["symbol"], r["month"]): r for r in previous.get("records", [])}
    results, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, s, m, records.get((s + "USDT", m))): (s, m) for s, m in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                symbol, month = futures[future]
                errors.append({"symbol": symbol, "month": month, "error": str(exc)})
                continue
            results.append(result)
            print(result["symbol"], result["month"], result["status"], flush=True)
    if len(results) + len(errors) != len(jobs):
        errors.append({"error": f"partition accounting mismatch: {len(results)} results + {len(errors)} errors for {len(jobs)} jobs"})
    results.sort(key=lambda x: (x["symbol"], x["month"]))
    manifest = {
        "universe": [s + "USDT" for s in SYMBOLS], "interval": INTERVAL,
        "download_start_inclusive": "2024-01-01T00:00:00Z", "validation_start_inclusive": "2024-03-01T00:00:00Z",
        "end_exclusive": "2026-03-01T00:00:00Z",
        "rows_per_symbol": sum(pd.Period(m).days_in_month for m in MONTHS) * 24 * 60 * 60 * 1000 // STEP_MS,
        "source": BASE, "disjoint_from_existing": "2026-03-01/2026-09-01",
        "records": results, "errors": errors,
    }
    atomic_json(manifest_path if not errors else OUT / "download_failures.json", manifest)
    print(json.dumps({"records": len(results), "downloaded": sum(r["status"] == "downloaded" for r in results)}, indent=2))
    if errors:
        raise RuntimeError(f"{len(errors)} partitions failed; previous complete manifest retained")


if __name__ == "__main__":
    main()
