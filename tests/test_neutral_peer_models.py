import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__file__.replace("\\tests\\test_neutral_peer_models.py", "\\src\\extended_data")))
from neutral_peer_models import MODEL_NAMES, _fit_model, build_neutral_models  # noqa: E402


def _sample():
    rng = np.random.default_rng(12)
    idx = pd.date_range("2024-01-01", "2024-11-01", freq="h", inclusive="left", tz="UTC")
    n = len(idx)
    common = np.cumsum(rng.normal(0, .003, n))
    peers = np.column_stack([common + np.cumsum(rng.normal(0, .002, n)) for _ in range(5)])
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = .97 * noise[i - 1] + rng.normal(0, .001)
    target = .2 * peers[:, 0] + .2 * peers[:, 1] + .2 * peers[:, 2] + .2 * peers[:, 3] + .2 * peers[:, 4] + noise
    return idx, np.exp(np.column_stack([target, peers])) * 100


def test_all_neutral_models_use_peers_and_sum_to_one():
    idx, close = _sample()
    y = np.log(close[:, 0]); x = np.log(close[:, 1:])
    for model in MODEL_NAMES:
        intercept, beta, chosen = _fit_model(model, y[:4000], x[:4000], .1)
        assert np.isfinite(intercept) and np.isfinite(beta).all()
        assert abs(beta.sum() - 1) < 1e-7
        assert model != "sparse3_neutral" or len(chosen) == 3


def test_formation_is_causal_and_schema_has_sensitivity():
    idx, close = _sample()
    kwargs = dict(index=idx, symbols=["T", "A", "B", "C", "D", "E"], fit_days=60,
                  cal_days=28, formation_start="2024-05-01", formation_end="2024-09-01",
                  models=MODEL_NAMES, hourly=False)
    selected, diag = build_neutral_models(close, **kwargs)
    assert len(diag) == 4 * 6 * 5
    assert set(diag.model) == set(MODEL_NAMES)
    assert (diag.beta_sum.dropna() - 1).abs().max() < 1e-7
    assert set(["feature_indices", "coef_json", "hedge_weights_json", "sensitivity_pred_rmse_log"]).issubset(diag.columns)
    changed = close.copy(); changed[idx >= pd.Timestamp("2024-09-01", tz="UTC")] *= 11
    _, after = build_neutral_models(changed, **kwargs)
    cols = ["target", "model", "model_end_time", "intercept", "coef_json", "cal_r2"]
    pd.testing.assert_frame_equal(diag[cols].reset_index(drop=True), after[cols].reset_index(drop=True))
