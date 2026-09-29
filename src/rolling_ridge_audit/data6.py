"""Load the repository's already downloaded minute data without network access."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "klines"
RESULTS = ROOT / "results"
SYMBOLS = [s + "USDT" for s in "BTC ETH BNB SOL XRP DOGE ADA TRX LINK SUI AVAX LTC BCH DOT HBAR XLM FIL UNI NEAR AAVE".split()]
START = pd.Timestamp("2026-03-01", tz="UTC")
END = pd.Timestamp("2026-09-01", tz="UTC")
N = int((END - START) / pd.Timedelta(minutes=1))
MONTHS = [f"2026-{m:02d}" for m in range(3, 9)]


def minute(value) -> int:
    return int((pd.Timestamp(value, tz="UTC") - START) / pd.Timedelta(minutes=1))


def load() -> tuple[np.ndarray, ...]:
    arrays = {name: np.empty((N, len(SYMBOLS)), dtype=float) for name in ("open", "close", "quote_volume", "taker_buy_quote_volume")}
    expected = (START.value // 10**6) + 60_000 * np.arange(N, dtype=np.int64)
    for j, symbol in enumerate(SYMBOLS):
        frames = [pd.read_parquet(DATA / f"symbol={symbol}" / f"month={month}.parquet") for month in MONTHS]
        frame = pd.concat(frames, ignore_index=True)
        if not np.array_equal(frame.open_time_utc_ms.to_numpy(np.int64), expected):
            raise ValueError(f"non-contiguous timestamps for {symbol}")
        for name in arrays:
            arrays[name][:, j] = frame[name].to_numpy(float)
    return arrays["open"], arrays["close"], np.log(arrays["close"]), arrays["quote_volume"], arrays["taker_buy_quote_volume"]


if __name__ == "__main__":
    op, cl, lc, vol, buy = load()
    print({"symbols": len(SYMBOLS), "rows": len(cl), "start": str(START), "end": str(END), "close_finite": bool(np.isfinite(cl).all())})
