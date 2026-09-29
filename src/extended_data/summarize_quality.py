"""Read every extended partition and write a content-addressed quality summary."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "extended_1m"
FUNDING = ROOT / "data" / "extended_funding"
KEEP = ["open_time_utc_ms", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    for part in sorted(path.rglob("*.parquet")):
        # Normalize the path component so the digest is reproducible on
        # Windows and POSIX workers reading the same partition tree.
        relative = part.relative_to(path).as_posix()
        h.update(relative.encode())
        h.update(hashlib.sha256(part.read_bytes()).digest())
    return h.hexdigest()


def check_partition(path: Path) -> dict:
    month = path.stem.split("=", 1)[1]
    frame = pd.read_parquet(path, columns=KEEP)
    step = 60_000
    expected = int(pd.Period(month).days_in_month * 86_400_000 // step)
    start = pd.Timestamp(month + "-01", tz="UTC").value // 10**6
    ts = frame.open_time_utc_ms.to_numpy("int64")
    grid = start + step * np.arange(expected, dtype="int64")
    if len(frame) != expected or not np.array_equal(ts, grid):
        raise AssertionError(f"invalid minute grid: {path}")
    values = frame[KEEP[1:]].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4:] < 0).any():
        raise AssertionError(f"invalid finite/price/volume values: {path}")
    if ((values[:, 1] < values[:, :4].max(axis=1)) | (values[:, 2] > values[:, :4].min(axis=1))).any():
        raise AssertionError(f"invalid OHLC range: {path}")
    if (values[:, 5] > values[:, 4] + 1e-8 + values[:, 4] * 1e-10).any():
        raise AssertionError(f"taker volume exceeds quote volume: {path}")
    return {"rows": len(frame), "zero_quote_volume": int((values[:, 4] == 0).sum()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check_funding(path: Path) -> dict:
    frame = pd.read_parquet(path).sort_values("funding_time_utc_ms").reset_index(drop=True)
    if frame.empty or frame.funding_time_utc_ms.duplicated().any():
        raise AssertionError(f"empty/duplicate funding: {path}")
    ts = frame.funding_time_utc_ms.to_numpy("int64")
    start = int(pd.Timestamp("2024-02-01", tz="UTC").timestamp() * 1000)
    end = int(pd.Timestamp("2026-03-01", tz="UTC").timestamp() * 1000)
    expected = start + 8 * 60 * 60 * 1000 * np.arange(len(ts), dtype="int64")
    if not np.all(np.abs(ts - expected) <= 1000) or not ts[-1] < end:
        raise AssertionError(f"funding grid/bounds failure: {path}")
    values = frame[["funding_rate", "mark_price"]].to_numpy(float)
    if not np.isfinite(values).all() or not (values[:, 1] > 0).all():
        raise AssertionError(f"funding non-finite/nonpositive mark: {path}")
    return {"rows": len(frame), "max_grid_error_ms": int(np.max(np.abs(ts - expected))),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    partitions = sorted(DATA.glob("symbol=*/month=*.parquet"))
    if not partitions:
        raise RuntimeError("no extended 1m partitions")
    checks = {p.relative_to(DATA).as_posix(): check_partition(p) for p in partitions}
    symbols = sorted({p.parent.name.split("=", 1)[1] for p in partitions})
    per_symbol = {s: sum(v["rows"] for p, v in checks.items()
                          if Path(p).parts[0] == f"symbol={s}") for s in symbols}
    zero_1m = sum(v["zero_quote_volume"] for v in checks.values())
    zero_5m = 0
    for symbol in symbols:
        chunks = [pd.read_parquet(p, columns=["open_time_utc_ms", "quote_volume"])
                  for p in sorted((DATA / f"symbol={symbol}").glob("month=*.parquet"))]
        frame = pd.concat(chunks, ignore_index=True)
        frame.index = pd.to_datetime(frame.open_time_utc_ms, unit="ms", utc=True)
        zero_5m += int((frame.quote_volume.resample("5min").sum() == 0).sum())
    funding_files = sorted(FUNDING.glob("symbol=*.parquet"))
    funding = {p.relative_to(FUNDING).as_posix(): check_funding(p) for p in funding_files}
    manifest_path = DATA / "download_manifest.json"
    manifest_records = json.loads(manifest_path.read_text(encoding="utf-8")).get("records", [])
    checksum_pairs = [r for r in manifest_records if r.get("sha256") and r.get("official_sha256")]
    checksum_evidence = {
        "manifest_records": len(manifest_records),
        "official_checksum_records": len(checksum_pairs),
        "matching_records": sum(r["sha256"].lower() == r["official_sha256"].lower() for r in checksum_pairs),
        "mismatches": sum(r["sha256"].lower() != r["official_sha256"].lower() for r in checksum_pairs),
    }
    quality = {
        "source": "https://data.binance.vision/data/futures/um/monthly/klines",
        "funding_source": "https://fapi.binance.com/fapi/v1/fundingRate",
        "symbols": symbols, "partitions": len(partitions), "rows": sum(per_symbol.values()),
        "rows_per_symbol": per_symbol, "zero_quote_volume_minutes": zero_1m,
        "aggregated_zero_quote_volume_5m": zero_5m,
        "off_grid_gaps": 0, "sample": "2024-03-01/2026-03-01", "warmup": "2024-01/2024-02",
        "disjoint_from_existing": True, "content_digest": digest(DATA),
        "partitions_sha256": checks, "funding_files": len(funding_files),
        "funding_rows": sum(v["rows"] for v in funding.values()),
        "funding_max_grid_error_ms": max((v["max_grid_error_ms"] for v in funding.values()), default=None),
        "funding_sha256": funding,
        "zip_checksum_evidence": checksum_evidence,
        "zero_volume_policy": "kept in features; entry requires prior quote volume > 0",
    }
    out = ROOT / "results" / "extended_data_quality.json"
    temporary = out.with_suffix(out.suffix + ".part")
    temporary.write_text(json.dumps(quality, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(out)
    print(json.dumps({k: quality[k] for k in ("symbols", "partitions", "rows", "zero_quote_volume_minutes", "aggregated_zero_quote_volume_5m", "funding_files", "funding_rows", "content_digest")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
