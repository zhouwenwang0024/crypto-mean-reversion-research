"""Download and validate two-year Binance UM funding rate plus mark price."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "extended_funding"
SYMBOLS = "BTC ETH BNB SOL XRP DOGE ADA TRX LINK SUI AVAX LTC BCH DOT HBAR XLM FIL UNI NEAR AAVE".split()
START = int(pd.Timestamp("2024-02-01", tz="UTC").timestamp() * 1000)
END = int(pd.Timestamp("2026-03-01", tz="UTC").timestamp() * 1000)
URL = "https://fapi.binance.com/fapi/v1/fundingRate"
STEP = 8 * 60 * 60 * 1000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(frame: pd.DataFrame) -> dict:
    frame = frame.sort_values("funding_time_utc_ms").reset_index(drop=True)
    if frame.empty or frame.funding_time_utc_ms.duplicated().any():
        raise ValueError("empty or duplicate funding timestamps")
    ts = frame.funding_time_utc_ms.to_numpy("int64")
    expected = START + STEP * np.arange(len(frame), dtype="int64")
    if not np.all(np.abs(ts - expected) <= 1000):
        raise ValueError("funding timestamps are not a complete 8h grid within 1s")
    if not frame.funding_time_utc_ms.between(START, END - 1).all():
        raise ValueError("funding timestamp outside requested range")
    if not frame[["funding_rate", "mark_price"]].apply(np.isfinite).all().all():
        raise ValueError("non-finite funding rate or mark price")
    if not frame.mark_price.gt(0).all():
        raise ValueError("non-positive funding mark price")
    return {"rows": len(frame), "start": int(ts[0]), "end": int(ts[-1]),
            "grid_step_ms": STEP, "max_grid_error_ms": int(np.max(np.abs(ts - expected)))}


def request(params: dict) -> requests.Response:
    for attempt in range(3):
        try:
            response = requests.get(URL, params=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def fetch(symbol: str) -> dict:
    dest = OUT / f"symbol={symbol}USDT.parquet"
    if dest.exists():
        try:
            frame = pd.read_parquet(dest)
            checks = validate(frame)
        except (OSError, ValueError, KeyError):
            pass
        else:
            return {"symbol": symbol + "USDT", **checks, "status": "existing", "parquet_sha256": sha256(dest)}
    rows = []; cursor = START
    while cursor < END:
        batch = request({"symbol": symbol + "USDT", "startTime": cursor,
                         "endTime": END - 1, "limit": 1000}).json()
        if not batch: break
        rows.extend(batch); last = int(batch[-1]["fundingTime"])
        if last < cursor: raise ValueError(f"funding cursor moved backwards: {symbol}")
        cursor = last + 1
        if len(batch) < 1000: break
    frame = pd.DataFrame(rows)
    if frame.empty: raise ValueError(f"no funding rows: {symbol}")
    frame = frame.rename(columns={"fundingTime": "funding_time_utc_ms", "fundingRate": "funding_rate",
                                  "markPrice": "mark_price"})[["funding_time_utc_ms", "funding_rate", "mark_price"]]
    frame["funding_time_utc_ms"] = pd.to_numeric(frame["funding_time_utc_ms"], errors="raise").astype("int64")
    frame["funding_rate"] = pd.to_numeric(frame["funding_rate"], errors="raise")
    frame["mark_price"] = pd.to_numeric(frame["mark_price"], errors="raise")
    checks = validate(frame)
    OUT.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".part")
    frame.to_parquet(temporary, index=False)
    validate(pd.read_parquet(temporary))
    temporary.replace(dest)
    return {"symbol": symbol + "USDT", **checks, "status": "downloaded", "parquet_sha256": sha256(dest)}


def main() -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, symbol): symbol for symbol in SYMBOLS}
        records = []
        for future in concurrent.futures.as_completed(futures):
            try:
                record = future.result()
            except Exception as exc:
                record = {"symbol": futures[future] + "USDT", "status": "error", "error": str(exc)}
            records.append(record)
            print(record, flush=True)
    records.sort(key=lambda x: x["symbol"])
    manifest = {"source": URL, "start": START, "end_exclusive": END,
        "expected_grid": "start + 8h*i within 1s", "records": records,
        "missing_policy": "error; no zero fill"}
    temporary = OUT / "manifest.json.part"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(OUT / "manifest.json")
    if any(r["status"] == "error" for r in records):
        raise RuntimeError("funding download failed")


if __name__ == "__main__": main()
