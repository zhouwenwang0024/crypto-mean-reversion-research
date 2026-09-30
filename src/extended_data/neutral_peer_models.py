"""Causal, neutral peer-price models for the five-minute archive.

Every formation month creates one model for every target and records the
out-of-sample quality on the preceding 28 days.  The peer coefficients always
sum to one, so the residual is a tradable target-versus-peer basket rather
than a directional forecast of the market level.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller


MODEL_NAMES = ("equal19", "simplex_ridge", "neutral_ridge", "pcr3_neutral", "sparse3_neutral")


def _periods(index: pd.DatetimeIndex, fit_days: int, cal_days: int,
             start: str | pd.Timestamp | None, end: str | pd.Timestamp | None) -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    first = idx.min().tz_convert(None).to_period("M").to_timestamp().tz_localize("UTC")
    last = idx.max().tz_convert(None).to_period("M").to_timestamp().tz_localize("UTC")
    lo = pd.Timestamp(start, tz="UTC") if start is not None else first + pd.Timedelta(days=fit_days + cal_days)
    hi = pd.Timestamp(end, tz="UTC") if end is not None else last + pd.offsets.MonthBegin(1)
    lo = lo.tz_convert(None).to_period("M").to_timestamp().tz_localize("UTC")
    hi = hi.tz_convert(None).to_period("M").to_timestamp().tz_localize("UTC")
    return list(pd.date_range(lo, hi, freq="MS", tz="UTC", inclusive="left"))


def _adf(x: np.ndarray) -> float:
    try:
        return float(adfuller(np.asarray(x, float), maxlag=1, regression="c", autolag=None,
                              result_object=False)[1])
    except Exception:
        return 1.0


def _quality(actual: np.ndarray, pred: np.ndarray) -> dict:
    resid = np.asarray(actual, float) - np.asarray(pred, float)
    if len(resid) < 100 or not np.isfinite(resid).all():
        raise ValueError("insufficient calibration observations")
    n = len(resid)
    var = float(np.sum((actual - actual.mean()) ** 2))
    z = resid - resid.mean()
    den = float(z[:-1] @ z[:-1])
    rho = float((z[1:] @ z[:-1]) / den) if den > 1e-14 else np.nan
    half = float(-np.log(2.0) / np.log(rho)) if 0.0 < rho < 1.0 else np.nan
    h = n // 2
    s1 = float(np.std(resid[:h], ddof=1)); s2 = float(np.std(resid[h:], ddof=1))
    def restoration_slope(part: np.ndarray) -> float:
        prior, delta = part[:-1], np.diff(part)
        prior = prior - prior.mean(); den = float(prior @ prior)
        return float(prior @ (delta - delta.mean()) / den) if den > 1e-14 else np.nan
    split_slope_1 = restoration_slope(resid[:h]); split_slope_2 = restoration_slope(resid[h:])
    return {
        "cal_obs": int(n), "cal_resid_mean": float(resid.mean()),
        "cal_resid_scale": float(np.std(resid, ddof=1)),
        "cal_rmse_log": float(np.sqrt(np.mean(resid * resid))),
        "cal_mae_log": float(np.mean(np.abs(resid))),
        "cal_r2": float(1.0 - np.sum(resid * resid) / var) if var > 1e-14 else np.nan,
        "cal_corr": float(np.corrcoef(actual, pred)[0, 1]) if np.std(pred) > 1e-12 else np.nan,
        "resid_adf_pvalue": _adf(resid), "resid_rho": rho,
        "resid_half_life_hours": half, "resid_scale_1": s1, "resid_scale_2": s2,
        "resid_mean_1": float(np.mean(resid[:h])), "resid_mean_2": float(np.mean(resid[h:])),
        "split_restoration_slope_1": split_slope_1, "split_restoration_slope_2": split_slope_2,
        "split_restoration_slope": float(np.nanmean([split_slope_1, split_slope_2])),
        "resid_scale_ratio": max(s1, s2) / max(min(s1, s2), 1e-12),
        "median_abs_error_pct": float(np.median(np.abs(np.expm1(resid))) * 100.0),
        "p95_abs_error_pct": float(np.quantile(np.abs(np.expm1(resid)), 0.95) * 100.0),
    }


def _norm_data(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    if y.ndim != 1 or x.ndim != 2 or len(y) != len(x) or not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("invalid regression data")
    yc = y - y.mean(); xc = x - x.mean(0)
    variance = float(np.var(y, ddof=1))
    if len(y) < 3 or variance <= 1e-14:
        raise ValueError("zero target variance")
    return yc, xc, float(y.mean()), variance


def _signed_ridge(y: np.ndarray, x: np.ndarray, alpha: float = 0.1) -> tuple[float, np.ndarray]:
    """Variance-normalized ridge with the raw-level constraint sum(beta)=1."""
    yc, xc, ymean, vy = _norm_data(y, x)
    n, p = xc.shape
    gram = xc.T @ xc / (n * vy) + alpha * np.eye(p)
    rhs = xc.T @ yc / (n * vy)
    c = np.ones(p)
    kkt = np.block([[gram, c[:, None]], [c[None, :], np.zeros((1, 1))]])
    sol = np.linalg.solve(kkt, np.r_[rhs, 1.0])
    beta = sol[:p]
    return float(ymean - x.mean(0) @ beta), beta


def _simplex_ridge(y: np.ndarray, x: np.ndarray, alpha: float = 0.1) -> tuple[float, np.ndarray]:
    """Variance-normalized ridge on the simplex, with explicit success checks."""
    yc, xc, ymean, vy = _norm_data(y, x)
    n, p = xc.shape
    gram = xc.T @ xc / (n * vy)
    rhs = xc.T @ yc / (n * vy)
    def objective(beta: np.ndarray) -> float:
        d = xc @ beta - yc
        return float((d @ d) / (n * vy) + alpha * (beta @ beta))
    def gradient(beta: np.ndarray) -> np.ndarray:
        return 2.0 * ((gram + alpha * np.eye(p)) @ beta - rhs)
    result = minimize(objective, np.full(p, 1.0 / p), jac=gradient, method="SLSQP",
                      bounds=[(0.0, 1.0)] * p,
                      constraints={"type": "eq", "fun": lambda b: float(b.sum() - 1.0),
                                   "jac": lambda b: np.ones(p)},
                      options={"ftol": 1e-12, "maxiter": 300})
    if not result.success or not np.isfinite(result.x).all() or abs(result.x.sum() - 1.0) > 1e-7:
        raise RuntimeError(f"simplex optimization failed: {result.message}")
    return float(ymean - x.mean(0) @ result.x), result.x


def _pcr3(y: np.ndarray, x: np.ndarray, alpha: float = 0.1) -> tuple[float, np.ndarray]:
    """Three standardized PCA factors, mapped back to raw peer coefficients."""
    yc, xc, ymean, vy = _norm_data(y, x)
    sd = x.std(0, ddof=1); sd = np.where(sd > 1e-12, sd, 1.0)
    xz = xc / sd
    _, _, vt = np.linalg.svd(xz, full_matrices=False)
    v = vt[:3].T
    factors = xz @ v
    theta = _constrained_factor_ridge(yc, factors, vy, alpha, v / sd[:, None])
    beta = (v @ theta) / sd
    if abs(beta.sum() - 1.0) > 1e-7:
        raise RuntimeError("PCA raw beta sum constraint failed")
    return float(ymean - x.mean(0) @ beta), beta


def _constrained_factor_ridge(yc: np.ndarray, factors: np.ndarray, vy: float,
                              alpha: float, raw_map: np.ndarray) -> np.ndarray:
    n, k = factors.shape
    gram = factors.T @ factors / (n * vy) + alpha * np.eye(k)
    rhs = factors.T @ yc / (n * vy)
    c = raw_map.sum(0)
    if abs(c @ c) <= 1e-14:
        raise ValueError("PCA constraint is singular")
    return np.linalg.solve(np.block([[gram, c[:, None]], [c[None, :], np.zeros((1, 1))]]),
                           np.r_[rhs, 1.0])[:k]


def _sparse3(y: np.ndarray, x: np.ndarray) -> tuple[float, np.ndarray, list[int]]:
    yc, xc, ymean, _ = _norm_data(y, x)
    returns = np.diff(np.column_stack((y, x)), axis=0)
    corr = np.array([np.corrcoef(returns[:, 0], returns[:, j + 1])[0, 1] for j in range(x.shape[1])])
    if not np.isfinite(corr).all():
        raise ValueError("non-finite return correlation")
    chosen = np.argsort(-np.abs(corr), kind="mergesort")[:3].tolist()
    xs = xc[:, chosen]; n = len(y)
    gram = xs.T @ xs / n; rhs = xs.T @ yc / n; c = np.ones(3)
    beta3 = np.linalg.solve(np.block([[gram, c[:, None]], [c[None, :], np.zeros((1, 1))]]),
                            np.r_[rhs, 1.0])[:3]
    beta = np.zeros(x.shape[1]); beta[chosen] = beta3
    return float(ymean - x.mean(0) @ beta), beta, chosen


def _fit_model(model: str, y: np.ndarray, x: np.ndarray, alpha: float) -> tuple[float, np.ndarray, list[int]]:
    if model == "equal19":
        beta = np.full(x.shape[1], 1.0 / x.shape[1]); return float(y.mean() - x.mean(0) @ beta), beta, list(range(x.shape[1]))
    if model == "simplex_ridge":
        b0, b = _simplex_ridge(y, x, alpha); return b0, b, list(range(x.shape[1]))
    if model == "neutral_ridge":
        b0, b = _signed_ridge(y, x, alpha); return b0, b, list(range(x.shape[1]))
    if model == "pcr3_neutral":
        b0, b = _pcr3(y, x, alpha); return b0, b, list(range(x.shape[1]))
    if model == "sparse3_neutral":
        return _sparse3(y, x)
    raise ValueError(f"unknown model {model}")


def _row_base(target: int, target_symbol: str, peers: list[int], symbols: list[str], model: str,
              month: pd.Timestamp, fit_start: pd.Timestamp, fit_end: pd.Timestamp,
              cal_start: pd.Timestamp) -> dict:
    return {"target": target, "target_symbol": target_symbol, "model": model,
            "model_end_time": month, "fit_start": fit_start, "fit_end": fit_end,
            "cal_start": cal_start, "cal_end": month, "ridge_alpha": .1,
            "feature_indices": json.dumps(peers), "feature_symbols": json.dumps([symbols[j] for j in peers]),
            "intercept": np.nan, "coef_json": "", "hedge_weights_json": "{}",
            "eligible": False, "error": "", "optimization_success": False}


def _one_fit(model: str, frame: pd.DataFrame, target: int, peers: list[int], alpha: float,
             fit_start: pd.Timestamp, fit_end: pd.Timestamp, cal_start: pd.Timestamp,
             month: pd.Timestamp, symbols: list[str]) -> dict:
    fit = frame[(frame.index >= fit_start) & (frame.index < fit_end)]
    cal = frame[(frame.index >= cal_start) & (frame.index < month)]
    row = _row_base(target, symbols[target], peers, symbols, model, month, fit_start, fit_end, cal_start)
    try:
        b0, beta, chosen = _fit_model(model, fit.iloc[:, target].to_numpy(), fit.iloc[:, peers].to_numpy(), alpha)
        pred = b0 + cal.iloc[:, peers].to_numpy() @ beta
        quality = _quality(cal.iloc[:, target].to_numpy(), pred)
        shifted_fit_start, shifted_fit_end = fit_start - pd.Timedelta(days=7), fit_end - pd.Timedelta(days=7)
        sf = frame[(frame.index >= shifted_fit_start) & (frame.index < shifted_fit_end)]
        sb0, sbeta, _ = _fit_model(model, sf.iloc[:, target].to_numpy(), sf.iloc[:, peers].to_numpy(), alpha)
        shifted_pred = sb0 + cal.iloc[:, peers].to_numpy() @ sbeta
        out_peers = [peers[k] for k in chosen] if model == "sparse3_neutral" else peers
        out_beta = beta[chosen] if model == "sparse3_neutral" else beta
        row["feature_indices"] = json.dumps(out_peers); row["feature_symbols"] = json.dumps([symbols[j] for j in out_peers])
        row.update({"intercept": b0, "coef_json": json.dumps(out_beta.tolist(), separators=(",", ":")),
                    "hedge_weights_json": json.dumps({symbols[j]: float(-beta[k]) for k, j in enumerate(peers) if abs(beta[k]) > 1e-12},
                                                       sort_keys=True, separators=(",", ":")),
                    "selected_peer_indices": json.dumps([peers[k] for k in chosen]),
                    "feature_mean_json": json.dumps(fit.iloc[:, out_peers].mean().tolist(), separators=(",", ":")),
                    "feature_scale_json": json.dumps(fit.iloc[:, out_peers].std(ddof=1).tolist(), separators=(",", ":")),
                    "beta_sum": float(beta.sum()),
                    "sensitivity_pred_rmse_log": float(np.sqrt(np.mean((pred - shifted_pred) ** 2))),
                    "sensitivity_beta_l1": float(np.abs(beta - sbeta).sum()),
                    "sensitivity_pred_corr": float(np.corrcoef(pred, shifted_pred)[0, 1]),
                    "optimization_success": True, **quality})
        row["eligible"] = bool(np.isfinite([row["cal_r2"], row["cal_corr"], row["resid_half_life_hours"]]).all()
                                and row["cal_r2"] >= 0 and row["resid_adf_pvalue"] <= .10
                                and 6 <= row["resid_half_life_hours"] <= 168
                                and row["cal_resid_scale"] > 0 and row["resid_scale_ratio"] <= 2.5)
    except Exception as exc:
        row["error"] = type(exc).__name__ + ": " + str(exc)[:180]
    return row


def _bh(pvalues: Iterable[float], q: float = .10) -> tuple[np.ndarray, np.ndarray]:
    p = np.clip(np.nan_to_num(np.asarray(list(pvalues), float), nan=1.0), 0, 1)
    order = np.argsort(p, kind="mergesort"); ranks = np.arange(1, len(p) + 1, dtype=float)
    qv = np.minimum.accumulate((p[order] * len(p) / ranks)[::-1])[::-1]
    out = np.empty_like(p); out[order] = np.minimum(qv, 1.0)
    return out, out <= q


def build_neutral_models(close: np.ndarray | pd.DataFrame, index: pd.DatetimeIndex | None = None,
                         *, fit_days: int = 60, cal_days: int = 28,
                         symbols: list[str] | None = None,
                         formation_start: str | pd.Timestamp | None = None,
                         formation_end: str | pd.Timestamp | None = None,
                         models: tuple[str, ...] = MODEL_NAMES, alpha: float = .1,
                         hourly: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(close, pd.DataFrame):
        index = pd.DatetimeIndex(close.index) if index is None else index
        symbols = symbols or [str(c) for c in close.columns]; values = close.to_numpy(float)
    else:
        values = np.asarray(close, float)
    if index is None or values.ndim != 2 or len(index) != len(values) or values.shape[1] < 4:
        raise ValueError("close/index shape mismatch")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("close must be finite and positive")
    idx = pd.DatetimeIndex(index); idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    order = np.argsort(idx.view("i8")); idx, values = idx[order], values[order]
    symbols = symbols or [str(i) for i in range(values.shape[1])]
    if len(symbols) != values.shape[1] or not set(models).issubset(MODEL_NAMES):
        raise ValueError("invalid symbols or models")
    frame = pd.DataFrame(np.log(values), index=idx, columns=symbols)
    if hourly:
        frame = frame.resample("1h", label="left", closed="left").last().dropna(how="any")
    rows = []
    for month in _periods(frame.index, fit_days, cal_days, formation_start, formation_end):
        cal_start = month - pd.Timedelta(days=cal_days); fit_end = cal_start; fit_start = fit_end - pd.Timedelta(days=fit_days)
        fit = frame[(frame.index >= fit_start) & (frame.index < fit_end)]
        cal = frame[(frame.index >= cal_start) & (frame.index < month)]
        if len(fit) < fit_days * 24 or len(cal) < cal_days * 24:
            continue
        for target in range(frame.shape[1]):
            peers = [j for j in range(frame.shape[1]) if j != target]
            for model in models:
                rows.append(_one_fit(model, frame, target, peers, alpha, fit_start, fit_end, cal_start, month, symbols))
    diagnostics = pd.DataFrame(rows)
    if diagnostics.empty:
        return diagnostics.copy(), diagnostics
    diagnostics["adf_bh_qvalue"] = 1.0; diagnostics["adf_bh_pass"] = False
    # One family of 20 x 5 hypotheses for each month, as predeclared before looking at PnL.
    for _, ids in diagnostics.groupby("model_end_time", sort=False).groups.items():
        qv, passed = _bh(diagnostics.loc[ids, "resid_adf_pvalue"].to_numpy(), .10)
        diagnostics.loc[ids, "adf_bh_qvalue"] = qv; diagnostics.loc[ids, "adf_bh_pass"] = passed
    diagnostics["eligible"] &= diagnostics["adf_bh_pass"]
    return diagnostics[diagnostics.eligible].copy().reset_index(drop=True), diagnostics


__all__ = ["MODEL_NAMES", "build_neutral_models", "_fit_model"]
