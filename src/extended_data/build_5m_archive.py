"""Build one complete 5-minute archive from the two local minute sources."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIRECT = ROOT / "data" / "extended_5m"
OLD = ROOT / "data" / "klines"
ARCHIVE = ROOT / "data" / "combined_5m_2024-02_to_2026-08.zip"
RESULT = ROOT / "results" / "combined_5m_manifest.json"
SYMBOLS = "BTC ETH BNB SOL XRP DOGE ADA TRX LINK SUI AVAX LTC BCH DOT HBAR XLM FIL UNI NEAR AAVE".split()
MONTHS = pd.period_range("2024-02", "2026-08", freq="M").astype(str).tolist()
KEEP = ["open_time_utc_ms", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]


def validate(frame: pd.DataFrame, month: str) -> None:
    step = 300_000; rows = pd.Period(month).days_in_month * 24 * 60 // 5
    start = int(pd.Timestamp(month + "-01", tz="UTC").timestamp() * 1000)
    stamps = frame.open_time_utc_ms.to_numpy("int64")
    if len(frame) != rows or not (stamps == start + step * pd.RangeIndex(rows).to_numpy()).all():
        raise ValueError(f"invalid 5m grid: {month}")
    values = frame[KEEP[1:]].to_numpy(float)
    if not pd.notna(values).all() or (values[:, :4] <= 0).any() or (values[:, 4:] < 0).any():
        raise ValueError(f"invalid 5m values: {month}")


def aggregate_old(symbol: str, month: str) -> pd.DataFrame:
    source = OLD / f"symbol={symbol}USDT" / f"month={month}.parquet"
    frame = pd.read_parquet(source, columns=["open_time_utc_ms", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"])
    index = pd.to_datetime(frame.open_time_utc_ms, unit="ms", utc=True)
    frame = frame.set_index(index)
    out = frame.resample("5min", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        quote_volume=("quote_volume", "sum"), taker_buy_quote_volume=("taker_buy_quote_volume", "sum"),
    ).reset_index(names="timestamp")
    out["open_time_utc_ms"] = out.pop("timestamp").astype("int64") // 10**6
    return out[KEEP]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def main() -> None:
    stage = ROOT / "data" / "_combined_5m_stage"
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir(parents=True)
    try:
        rows = 0
        for symbol in SYMBOLS:
            for month in MONTHS:
                source = DIRECT / f"symbol={symbol}USDT" / f"month={month}.parquet"
                frame = pd.read_parquet(source) if source.exists() else aggregate_old(symbol, month)
                validate(frame, month)
                dest = stage / f"symbol={symbol}USDT" / f"month={month}.parquet"; dest.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(dest, index=False, compression="snappy"); rows += len(frame)
        metadata = {"format": "parquet/snappy inside zip/deflate", "partitions": len(SYMBOLS) * len(MONTHS),
                    "rows": rows, "symbols": len(SYMBOLS), "months": MONTHS,
                    "warmup": "2024-02", "formal_sample": "2024-03-01/2026-09-01",
                    "sources": ["data/extended_5m (2024-02/2026-02)", "data/klines aggregated to 5m (2026-03/2026-08)"]}
        (stage / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file(): archive.write(path, path.relative_to(stage).as_posix())
        metadata.update({"archive": str(ARCHIVE.relative_to(ROOT)).replace("\\", "/"),
                         "archive_bytes": ARCHIVE.stat().st_size, "archive_sha256": sha256(ARCHIVE)})
        RESULT.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2))
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__": main()
