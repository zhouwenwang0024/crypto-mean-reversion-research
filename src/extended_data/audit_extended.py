"""Causal prefix and ledger sanity checks for the new validation engine."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from extended_data.validate_models import (RESULTS, bar_index, feature_set, load_bars,
                                           simulate)


def main() -> None:
    index, op, _, _, cl, vol = load_bars(); lc = np.log(cl)
    base = feature_set(lc, "ridge_std_l100")
    cut = bar_index(index, "2025-03-01")
    changed = cl.copy(); changed[cut:] *= np.exp(0.25 * np.sin(np.arange(len(cl) - cut)[:, None] / 19.0))
    alt = feature_set(np.log(changed), "ridge_std_l100")
    prefix = slice(None, cut + 1)
    finite = np.isfinite(base["dev"][prefix]) & np.isfinite(alt["dev"][prefix])
    result = {
        "cut": "2025-03-01", "dev_prefix_max_error": float(np.max(np.abs(base["dev"][prefix][finite] - alt["dev"][prefix][finite]))),
        "scale_prefix_max_error": float(np.max(np.abs(base["scale"][prefix][finite] - alt["scale"][prefix][finite]))),
        "weights_prefix_max_error": float(np.nanmax(np.abs(base["weights"][:cut // 288 + 1] - alt["weights"][:cut // 288 + 1]))),
        "future_dev_changed": float(np.nanmax(np.abs(base["dev"][cut + 1:] - alt["dev"][cut + 1:]))),
    }
    start = bar_index(index, "2025-09-01"); end = bar_index(index, "2026-03-01")
    no_fee, _, _ = simulate(op, cl, vol, base, start, end, 0.0)
    fee, normal_trades, _ = simulate(op, cl, vol, base, start, end, 5.0)
    reversed_run, reversed_trades, _ = simulate(op, cl, vol, base, start, end, 5.0, reverse_signal=True)
    result["fee_reduces_return"] = bool(fee["return"] <= no_fee["return"] + 1e-12)
    result["reverse_direction_changes"] = bool(len(normal_trades) and len(reversed_trades) and abs(fee["return"] - reversed_run["return"]) > 1e-8)
    # A deliberately faulty ledger that forgets fees must fail reconciliation by exactly its fees.
    faulty_no_fee_return = fee["return"] + fee["fees"]
    result["omitted_fee_detected"] = bool(abs(faulty_no_fee_return - no_fee["return"]) > 1e-8)
    result["all_pass"] = result["dev_prefix_max_error"] < 1e-10 and result["scale_prefix_max_error"] < 1e-10 and result["weights_prefix_max_error"] < 1e-10 and result["future_dev_changed"] > 1e-8 and result["fee_reduces_return"] and result["reverse_direction_changes"] and result["omitted_fee_detected"]
    (RESULTS / "extended_causal_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)); assert result["all_pass"]


if __name__ == "__main__": main()
