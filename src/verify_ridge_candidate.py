"""Independent integrity checks for the Ridge candidate artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v2 import END, START, RESULTS


def main() -> None:
    errors = []
    variants = []
    for p in sorted(RESULTS.glob("ridge_candidate_*_trades.csv")):
        if "pre_" in p.name:
            continue
        name = p.name.removeprefix("ridge_candidate_").removesuffix("_trades.csv")
        d = pd.read_csv(p)
        times = pd.to_datetime(d[["entry_time", "exit_time"]].stack(), utc=True)
        if len(times) and (times.min() < START or times.max() >= END):
            errors.append(f"{name}: trade outside sample")
        if len(d):
            pnl = d.side * d.notional * (d.exit_price / d.entry_price - 1.0)
            fee = d.notional * (1.0 + d.exit_price / d.entry_price) * float(name.split("_")[-2].replace("bp", "")) / 10000.0
            errors.extend([] if np.allclose(pnl, d.gross_price_pnl, atol=1e-8) else [f"{name}: pnl mismatch"])
            errors.extend([] if np.allclose(pnl - d.fee, d.net_pnl, atol=1e-8) else [f"{name}: net mismatch"])
            errors.extend([] if np.allclose(fee, d.fee, atol=1e-6) else [f"{name}: fee mismatch"])
            # Pairwise intervals prove the target-only account never has two
            # simultaneous lots of one target and never exceeds three lots.
            active = []
            for r in d.sort_values("entry_time").itertuples():
                en, ex = pd.Timestamp(r.entry_time), pd.Timestamp(r.exit_time)
                active = [(a, b, s) for a, b, s in active if b > en]
                if any(s == r.target for _, _, s in active): errors.append(f"{name}: overlapping {r.target}")
                active.append((en, ex, r.target))
                if len(active) > 3: errors.append(f"{name}: >3 positions")
        variants.append({"variant": name, "trades": int(len(d)), "fees": float(d.fee.sum()) if len(d) else 0.0})
    result = {"variants": variants, "errors": errors, "ok": not errors}
    (RESULTS / "ridge_candidate_integrity.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
