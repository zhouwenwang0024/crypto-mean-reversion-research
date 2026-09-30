"""Independent geometric oracles for the actual TLS spread fitter."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "extended_data"))
from statarb_models import _fit, _tls


def test_tls_exact_positive_and_negative_slopes():
    x = np.linspace(-3.0, 4.0, 401)
    for beta in (0.5, 1.0, 2.0, -0.75):
        y = 1.3 + beta * x
        alpha_hat, beta_hat = _fit("tls_log", np.column_stack((y, x)))
        assert abs(alpha_hat - 1.3) < 1e-12
        assert abs(beta_hat - beta) < 1e-12


def test_tls_noisy_geometry_matches_closed_form():
    rng = np.random.default_rng(614)
    latent = rng.normal(size=2000)
    x = latent + rng.normal(0.0, 0.12, len(latent))
    y = 0.7 + 1.8 * latent + rng.normal(0.0, 0.12, len(latent))
    vx, vy, cxy = np.var(x), np.var(y), np.mean((x - x.mean()) * (y - y.mean()))
    # Independent covariance closed form: no SVD or fitter output reused.
    expected_beta = (vy - vx + np.sqrt((vy - vx) ** 2 + 4.0 * cxy ** 2)) / (2.0 * cxy)
    alpha, beta = _tls(y, x)
    assert abs(beta - expected_beta) < 1e-12
    assert abs(alpha - (y.mean() - expected_beta * x.mean())) < 1e-12
    reverse_alpha, reverse_beta = _tls(x, y)
    assert abs(beta * reverse_beta - 1.0) < 1e-12
    assert abs(reverse_alpha + alpha / beta) < 1e-12


def test_tls_old_normal_as_direction_mutant_is_rejected():
    x = np.linspace(-2.0, 2.0, 301)
    y = 0.4 + 2.0 * x
    z = np.column_stack((x, y)); z -= z.mean(0)
    normal = np.linalg.svd(z, full_matrices=False)[2][-1]
    wrong_beta = normal[1] / normal[0]
    assert abs(wrong_beta - 2.0) > 1.0
    assert abs(_tls(y, x)[1] - 2.0) < 1e-12
