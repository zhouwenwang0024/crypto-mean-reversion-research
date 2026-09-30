from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "extended_data"))
from target_predictor import build_target_models, expand_predictions  # noqa: E402


def _synthetic():
    rng = np.random.default_rng(43)
    idx = pd.date_range("2024-01-01", "2024-09-01", freq="h", inclusive="left", tz="UTC")
    n = len(idx)
    common = np.cumsum(rng.normal(0, 0.004, n))
    p0 = common + np.cumsum(rng.normal(0, 0.002, n))
    p1 = common + np.cumsum(rng.normal(0, 0.002, n))
    p2 = np.zeros(n)
    for i in range(1, n):
        p2[i] = 0.92 * p2[i - 1] + rng.normal(0, 0.003)
    # A target whose peer-implied log price has a genuinely mean-reverting error.
    target = 0.55 * p0 + 0.45 * p1 + p2
    logs = np.column_stack((target, p0, p1, common + rng.normal(0, 0.01, n)))
    return idx, np.exp(logs) * 100.0


def test_target_models_are_peer_only_and_causal():
    idx, close = _synthetic()
    kwargs = dict(index=idx, symbols=["A", "B", "C", "D"], fit_days=60,
                  cal_days=28, formation_start="2024-04-01", formation_end="2024-07-01",
                  hourly=False, estimators=("ridge", "ols"))
    selected, diag = build_target_models(close, **kwargs)
    assert len(diag) == 4 * 2 * 3
    assert set(diag.model) == {"ridge_peers", "ols_peers"}
    row = diag.query("target_symbol == 'A'").iloc[0]
    assert "A" not in row.feature_symbols
    assert row.cal_obs == 28 * 24
    assert np.isfinite(row.cal_r2)
    assert set(selected.columns) >= {"target", "coef_json", "resid_adf_pvalue", "eligible"}


def test_suffix_perturbation_does_not_change_earlier_formation():
    idx, close = _synthetic()
    kwargs = dict(index=idx, symbols=["A", "B", "C", "D"], fit_days=60,
                  cal_days=28, formation_start="2024-04-01", formation_end="2024-06-01",
                  hourly=False, estimators=("ridge",))
    _, before = build_target_models(close, **kwargs)
    changed = close.copy()
    changed[idx >= pd.Timestamp("2024-06-01", tz="UTC")] *= 7.0
    _, after = build_target_models(changed, **kwargs)
    cols = ["target", "model_end_time", "intercept", "coef_json", "cal_r2", "resid_adf_pvalue"]
    pd.testing.assert_frame_equal(before[cols].reset_index(drop=True), after[cols].reset_index(drop=True))


def test_expanded_monthly_rows_do_not_overlap():
    idx, close = _synthetic()
    selected, _ = build_target_models(
        close, index=idx, symbols=["A", "B", "C", "D"], fit_days=60, cal_days=28,
        formation_start="2024-04-01", formation_end="2024-07-01", hourly=False,
        estimators=("ridge",), min_r2=-10.0, adf_p=1.0,
    )
    pred = expand_predictions(selected, close, idx, hourly=False)
    assert not pred.duplicated(["timestamp", "target"]).any()
