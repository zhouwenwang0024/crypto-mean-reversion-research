"""Causal suffix, unit-boundary and mutation checks on actual functions."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

try:
    from .data6 import N, RESULTS, SYMBOLS, load, minute
    from .engine import simulate6
    from . import engine_original
    from .independent_checks import independent_features
    from .models import SPECS, load_feat, make
    from .tests_core import toy
except ImportError:
    from data6 import N, RESULTS, SYMBOLS, load, minute
    from engine import simulate6
    import engine_original
    from independent_checks import independent_features
    from models import SPECS, load_feat, make
    from tests_core import toy


def run_arrays(op, cl, lc, vol, feat, start, end, original=False):
    ends = np.arange(15, N, 5, dtype=np.int64)
    rolling = pd.DataFrame(vol).rolling(5, min_periods=5).sum().to_numpy()
    valid = np.isfinite(rolling[ends - 1]).all(1) & (rolling[ends - 1] > 0).all(1)
    times = np.arange(31 * 1440, N + 1, 1440, dtype=np.int64)
    wi = np.maximum(0, np.searchsorted(times, np.arange(N + 1), side="right") - 1)
    fn = engine_original.simulate6 if original else simulate6
    return fn(op, cl, lc, np.zeros((1, 1)), ends, wi[ends], np.ascontiguousarray(feat["W"]),
              np.ascontiguousarray(feat["dev"][ends]), np.ascontiguousarray(feat["center"][ends]),
              np.ascontiguousarray(feat["scale"][ends]), valid, start, end, entry_z=2., gap_min=.015,
              structure=0, exit_type=8, repair=.5, hold_min=240, cost_bp=5., record=True,
              stop_fraction=.03, entry_delay=1, live_dev=feat["dev"], live_scale=feat["scale"])


def future_mutation(op, cl, lc, vol, base, cut_text):
    cut = minute(cut_text)
    alt_cl, alt_op = np.array(cl), np.array(op)
    k = np.arange(N - cut)[:, None]; coin = np.arange(len(SYMBOLS))[None, :]
    mult = np.exp(.1 * np.sin(k / 137 + coin) * np.minimum(1, (k + 1) / 100) + .05 * k / max(1, N - cut))
    alt_cl[cut:] *= mult; alt_op[cut + 1:] *= mult[:-1]
    alt = independent_features(np.log(alt_cl)); start = minute("2026-05-01"); end = min(cut + 3 * 1440, N)
    aa = run_arrays(op, cl, lc, vol, base, start, end)
    bb = run_arrays(alt_op, alt_cl, np.log(alt_cl), vol, alt, start, end)
    ix = cut - start
    fit_times = np.arange(31 * 1440, N + 1, 1440)
    upto = np.searchsorted(fit_times, cut, side="right")
    return {
        "cut": cut_text,
        "dev_prefix_error": float(np.nanmax(abs(base["dev"][:cut + 1] - alt["dev"][:cut + 1]))),
        "W_prefix_error": float(np.max(abs(base["W"][:upto] - alt["W"][:upto]))),
        "equity_prefix_error": float(np.max(abs(aa[0][:ix + 1] - bb[0][:ix + 1]))),
        "quantity_prefix_error": float(np.max(abs(aa[5][:ix + 1] - bb[5][:ix + 1]))),
        "future_equity_changed": float(np.max(abs(aa[0][ix + 1:] - bb[0][ix + 1:])))
    }


def main():
    op, cl, lc, vol, _ = load()
    make(SPECS[0], force=True); feat = load_feat("ridge_sma3"); base = independent_features(lc)
    causal = [future_mutation(op, cl, lc, vol, base, x) for x in ("2026-05-21 12:00", "2026-07-15 00:00")]

    factors = 10. ** np.linspace(-2, 2, len(SYMBOLS)); scaled_cl = cl * factors; scaled_op = op * factors
    scaled = independent_features(np.log(scaled_cl)); start = minute("2026-07-01"); end = minute("2026-09-01")
    old_a = run_arrays(op, cl, lc, vol, base, start, end, original=True)
    old_b = run_arrays(scaled_op, scaled_cl, np.log(scaled_cl), vol, scaled, start, end, original=True)
    new_a = run_arrays(op, cl, lc, vol, base, start, end)
    new_b = run_arrays(scaled_op, scaled_cl, np.log(scaled_cl), vol, scaled, start, end)
    unit = {"original_max_equity_error": float(np.max(abs(old_a[0] - old_b[0]))), "tolerant_max_equity_error": float(np.max(abs(new_a[0] - new_b[0])))}

    fee_case, no_fee_case = toy(7, 5), toy(7, 0)
    reverse_case = toy(0, 0, shock=(10, 99))
    future_case, changed_future = toy(7), toy(7, shock=(6, 120))
    # The first fill in the toy uses known price 100 and 30% equity.  A
    # faulty implementation sizing from the future 120 quote would produce
    # 0.0025 units instead of the actual 0.003 units.
    actual_qty = abs(float(future_case[4][0, 18])); future_sized_qty = 0.3 / 120.0
    mutation = {
        "remove_fees_detected": bool(abs(fee_case[0][-1] - no_fee_case[0][-1]) > 1e-9),
        "reverse_direction_detected": bool(reverse_case[4][0, 7] > 0 and -reverse_case[4][0, 7] <= 0),
        "size_using_future_detected": bool(abs(actual_qty - future_sized_qty) > 1e-6)
    }
    result = {"causal_suffix": causal, "unit_invariance": unit, "mutations": mutation,
              "all_pass": all(x["equity_prefix_error"] < 1e-9 and x["quantity_prefix_error"] < 1e-9 for x in causal) and unit["tolerant_max_equity_error"] < 1e-9 and all(mutation.values())}
    (RESULTS / "contract_checks_local.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
