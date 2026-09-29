"""Run independent feature, causality, account and mutant checks on minute validation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from extended_data.audit_minute import (assert_negative_mutants, compare_features,
                                        future_suffix_check, independent_ridge_features,
                                        replay_ledger, reconcile)
from extended_data.minute_validation import features, load_data, load_funding, run


def main() -> None:
    index, op, cl, vol = load_data(); lc = np.log(cl); fund, _, _ = load_funding(index)
    actual = features(lc, "ridge_sma3"); independent = independent_ridge_features(lc)
    checks = compare_features(actual, independent, atol=2e-9)
    start = int(index.searchsorted(pd.Timestamp("2025-09-01", tz="UTC")))
    end = int(index.searchsorted(pd.Timestamp("2025-12-01", tz="UTC")))
    account = run(op, cl, vol, actual, index, start, end, 5.0)
    ledger_eq, ledger_tape = replay_ledger(op, cl, None, account[4], start, end, cost_bp=5.0)
    checks.update({"account": reconcile(account[0], account[5], ledger_eq, ledger_tape),
                   "mutants": assert_negative_mutants(account[0], account[5], ledger_eq, ledger_tape,
                                                       op=op, cl=cl, fund=None, trades=account[4],
                                                       start=start, end=end, cost_bp=5.0)})
    cut = int(index.searchsorted(pd.Timestamp("2025-10-15", tz="UTC")))
    checks["causal"] = future_suffix_check(
        op, cl, None, cut=cut, start=start, end=end,
        build_features=lambda x: features(x, "ridge_sma3"),
        run_account=lambda o, c, f, feat, s, e: run(o, c, vol, feat, index, s, e, 5.0),
    )
    checks["all_pass"] = bool(checks["pass"] and checks["account"]["pass"] and
                               all(checks["mutants"].values()) and checks["causal"]["pass"])
    (ROOT / "results" / "extended_minute_causal_checks.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(checks, indent=2, ensure_ascii=False)); assert checks["all_pass"]


if __name__ == "__main__": main()
