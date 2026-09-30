"""Causal pair-statistical-arbitrage formation for the complete 5-minute lake.

The estimators operate on hourly closes, while the caller may retain the
original five-minute clock for execution.  At each calendar-month formation
date the preceding ``fit_days`` and the immediately preceding ``cal_days``
are disjoint.  Coefficients are frozen after formation; the calibration
sample is used only for stationarity diagnostics and candidate selection.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


MODEL_NAMES = ("ols_log", "tls_log", "huber_log", "minvar_return", "johansen")


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    xx = np.column_stack((np.ones(len(x)), x))
    alpha, beta = np.linalg.lstsq(xx, y, rcond=None)[0]
    return float(alpha), float(beta)


def _tls(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    z = np.column_stack((x, y))
    centre = z.mean(0)
    _, _, vh = np.linalg.svd(z - centre, full_matrices=False)
    # The smallest right singular vector is the line's *normal*: its
    # components satisfy nx*(x-mean_x) + ny*(y-mean_y) = 0.
    normal = vh[-1]
    if abs(normal[1]) < 1e-12:
        raise ValueError("vertical TLS direction")
    beta = -normal[0] / normal[1]
    return float(centre[1] - beta * centre[0]), float(beta)


def _huber(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Deterministic Huber IRLS with a fixed tuning constant."""
    alpha, beta = _ols(y, x)
    xx = np.column_stack((np.ones(len(x)), x))
    for _ in range(25):
        resid = y - alpha - beta * x
        mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        scale = max(float(mad), float(np.std(resid, ddof=1)), 1e-10)
        u = np.abs(resid) / (1.345 * scale)
        weights = np.where(u <= 1.0, 1.0, 1.0 / u)
        rootw = np.sqrt(weights)
        nxt = np.linalg.lstsq(xx * rootw[:, None], y * rootw, rcond=None)[0]
        if np.max(np.abs(nxt - (alpha, beta))) < 1e-10:
            alpha, beta = map(float, nxt)
            break
        alpha, beta = map(float, nxt)
    return alpha, beta


def _minvar_return(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    dy, dx = np.diff(y), np.diff(x)
    den = float(np.dot(dx - dx.mean(), dx - dx.mean()))
    if den <= 1e-14:
        raise ValueError("constant hedge return")
    beta = float(np.dot(dx - dx.mean(), dy - dy.mean()) / den)
    alpha = float(np.mean(y - beta * x))
    return alpha, beta


def _johansen(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """First Johansen trace vector, normalised to y - beta*x."""
    result = coint_johansen(np.column_stack((y, x)), det_order=0, k_ar_diff=1)
    vec = np.asarray(result.evec[:, 0], float)
    if abs(vec[0]) <= 1e-12:
        raise ValueError("singular Johansen vector")
    beta = float(-vec[1] / vec[0])
    alpha = float(np.mean(y - beta * x))
    return alpha, beta


_FITTERS = {
    "ols_log": _ols,
    "tls_log": _tls,
    "huber_log": _huber,
    "minvar_return": _minvar_return,
    "johansen": _johansen,
}


def _fit(model: str, levels: np.ndarray) -> tuple[float, float]:
    y, x = levels[:, 0], levels[:, 1]
    alpha, beta = _FITTERS[model](y, x)
    if not np.isfinite(alpha + beta):
        raise ValueError("non-finite coefficient")
    return alpha, beta


def _adf_pvalue(spread: np.ndarray) -> float:
    try:
        # Fixed lag and deterministic term avoid an adaptive look-ahead-like
        # choice.  The calibration mean is computed outside this function.
        return float(adfuller(
            spread, maxlag=1, regression="c", autolag=None, result_object=False
        )[1])
    except Exception:
        return 1.0


def _ar1(spread: np.ndarray) -> tuple[float, float]:
    z = np.asarray(spread, float) - float(np.mean(spread))
    den = float(np.dot(z[:-1], z[:-1]))
    if den <= 1e-14:
        return np.nan, np.nan
    rho = float(np.dot(z[1:], z[:-1]) / den)
    half = float(-np.log(2.0) / np.log(rho)) if 0.0 < rho < 1.0 else np.nan
    return rho, half


def _recovery_slope(spread: np.ndarray) -> float:
    prior, delta = spread[:-1], np.diff(spread)
    prior = prior - prior.mean()
    den = float(np.dot(prior, prior))
    return float(np.dot(prior, delta - delta.mean()) / den) if den > 1e-14 else np.nan


def _bh(pvalues: Iterable[float], q: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    p = np.nan_to_num(np.asarray(list(pvalues), float), nan=1.0, posinf=1.0, neginf=1.0)
    p = np.clip(p, 0.0, 1.0)
    order = np.argsort(p, kind="mergesort")
    ranks = np.arange(1, len(p) + 1, dtype=float)
    adjusted_sorted = np.minimum.accumulate((p[order] * len(p) / ranks)[::-1])[::-1]
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted, adjusted <= q


def _periods(index: pd.DatetimeIndex, fit_days: int, cal_days: int,
             formation_start: str | pd.Timestamp | None,
             formation_end: str | pd.Timestamp | None) -> list[pd.Timestamp]:
    # Convert to naive only for Period's calendar arithmetic, then restore UTC
    # explicitly.  This avoids silently dropping timezone information.
    first = index.min().tz_convert(None).to_period("M").to_timestamp().tz_localize("UTC")
    last = index.max().tz_convert(None).to_period("M").to_timestamp().tz_localize("UTC")
    start = (pd.Timestamp(formation_start, tz="UTC") if formation_start is not None
             else first + pd.Timedelta(days=fit_days + cal_days))
    start = start.tz_convert(None).to_period("M").to_timestamp().tz_localize("UTC")
    end = (pd.Timestamp(formation_end, tz="UTC") if formation_end is not None
           else last + pd.offsets.MonthBegin(1))
    end = end.tz_convert(None).to_period("M").to_timestamp().tz_localize("UTC")
    return list(pd.date_range(start, end, freq="MS", tz="UTC", inclusive="left"))


def build_models(
    cl: np.ndarray | pd.DataFrame,
    index: pd.DatetimeIndex | None = None,
    fit_days: int = 60,
    cal_days: int = 28,
    symbols: list[str] | None = None,
    formation_start: str | pd.Timestamp | None = None,
    formation_end: str | pd.Timestamp | None = None,
    fdr_q: float = 0.10,
    max_selected: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit and screen every fixed pair at each month formation.

    ``cl`` must contain five-minute close prices.  The data are reduced to
    hourly closes with left-labelled UTC bars.  Returned diagnostic rows
    include failed fits (with ``fit_ok=False``), while ``selected`` contains
    at most ``max_selected`` non-overlapping pairs per model/month.
    """
    if isinstance(cl, pd.DataFrame):
        if index is None:
            index = pd.DatetimeIndex(cl.index)
        values = cl.to_numpy(float)
        if symbols is None:
            symbols = [str(x) for x in cl.columns]
    else:
        values = np.asarray(cl, float)
    if index is None or len(index) != len(values):
        raise ValueError("index and close rows must have the same length")
    index = pd.DatetimeIndex(index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("close must be a finite positive matrix")
    ncoin = values.shape[1]
    symbols = symbols or [str(i) for i in range(ncoin)]
    if len(symbols) != ncoin:
        raise ValueError("symbols length mismatch")

    order = np.argsort(index.view("i8"))
    index, values = index[order], values[order]
    hourly = pd.DataFrame(np.log(values), index=index, columns=symbols)
    hourly = hourly.resample("1h", label="left", closed="left").last().dropna(how="any")
    periods = _periods(hourly.index, fit_days, cal_days, formation_start, formation_end)
    pairs = list(combinations(range(ncoin), 2))
    rows: list[dict] = []
    for month in periods:
        cal_end = month
        cal_start = month - pd.Timedelta(days=cal_days)
        fit_end = cal_start
        fit_start = fit_end - pd.Timedelta(days=fit_days)
        fit = hourly[(hourly.index >= fit_start) & (hourly.index < fit_end)].to_numpy()
        cal = hourly[(hourly.index >= cal_start) & (hourly.index < cal_end)].to_numpy()
        if len(fit) < fit_days * 24 or len(cal) < cal_days * 24:
            continue
        for model in MODEL_NAMES:
            for a, b in pairs:
                row = {
                    "model": model, "model_end_time": month, "fit_start": fit_start,
                    "fit_end": fit_end, "cal_start": cal_start, "cal_end": cal_end,
                    "a": a, "b": b, "symbol_a": symbols[a], "symbol_b": symbols[b],
                    "fit_ok": False, "beta": np.nan, "alpha": np.nan,
                    "beta_half_1": np.nan, "beta_half_2": np.nan,
                    "beta_relative_gap": np.nan, "adf_pvalue": 1.0,
                    "rho": np.nan, "half_life_hours": np.nan,
                    "cal_mean": np.nan, "cal_scale": np.nan,
                    "recovery_slope_1": np.nan, "recovery_slope_2": np.nan,
                    "beta_pass": False, "recovery_pass": False,
                    "adf_bh_qvalue": 1.0, "adf_bh_pass": False, "ar_pass": False,
                    "eligible": False, "selected": False, "error": "",
                }
                try:
                    pair_fit = fit[:, [a, b]]
                    alpha, beta = _fit(model, pair_fit)
                    h = len(pair_fit) // 2
                    _, beta1 = _fit(model, pair_fit[:h])
                    _, beta2 = _fit(model, pair_fit[h:])
                    cal_spread = cal[:, a] - alpha - beta * cal[:, b]
                    c1, c2 = np.array_split(cal_spread, 2)
                    rho, half = _ar1(cal_spread)
                    row.update({
                        "fit_ok": True, "beta": beta, "alpha": alpha,
                        "beta_half_1": beta1, "beta_half_2": beta2,
                        "beta_relative_gap": abs(beta1 - beta2) / max(abs(beta), 1e-12),
                        "adf_pvalue": _adf_pvalue(cal_spread), "rho": rho,
                        "half_life_hours": half, "cal_mean": float(np.mean(cal_spread)),
                        "cal_scale": float(np.std(cal_spread, ddof=1)),
                        "recovery_slope_1": _recovery_slope(c1),
                        "recovery_slope_2": _recovery_slope(c2),
                    })
                    row["beta_pass"] = (0.25 <= beta <= 4.0 and row["beta_relative_gap"] <= 0.5)
                    row["recovery_pass"] = row["recovery_slope_1"] < 0 and row["recovery_slope_2"] < 0
                    row["ar_pass"] = bool(0.0 < rho < 1.0 and 6.0 <= half <= 168.0)
                    row["eligible"] = bool(row["fit_ok"] and row["beta_pass"] and row["recovery_pass"] and row["ar_pass"])
                except Exception as exc:
                    row["error"] = type(exc).__name__ + ": " + str(exc)[:180]
                rows.append(row)

    diagnostics = pd.DataFrame(rows)
    if diagnostics.empty:
        return pd.DataFrame(), diagnostics
    diagnostics["adf_bh_qvalue"] = 1.0
    diagnostics["adf_bh_pass"] = False
    for (_, model), ids in diagnostics.groupby(["model_end_time", "model"], sort=False).groups.items():
        qvals, passed = _bh(diagnostics.loc[ids, "adf_pvalue"].to_numpy(), fdr_q)
        diagnostics.loc[ids, "adf_bh_qvalue"] = qvals
        diagnostics.loc[ids, "adf_bh_pass"] = passed
    diagnostics["eligible"] &= diagnostics["adf_bh_pass"]
    diagnostics["selected"] = False
    chosen: list[pd.Series] = []
    for (_, _), group in diagnostics.groupby(["model_end_time", "model"], sort=False):
        candidate = group[group["eligible"]].sort_values(
            ["adf_bh_qvalue", "adf_pvalue", "beta_relative_gap", "a", "b"]
        )
        used: set[int] = set()
        for idx, row in candidate.iterrows():
            if row.a in used or row.b in used:
                continue
            diagnostics.loc[idx, "selected"] = True
            used.update((int(row.a), int(row.b)))
            chosen.append(diagnostics.loc[idx])
            if len(chosen) and sum(
                1 for x in chosen if x["model_end_time"] == row["model_end_time"] and x["model"] == row["model"]
            ) >= max_selected:
                break
    selected = diagnostics[diagnostics["selected"]].copy().reset_index(drop=True)
    return selected, diagnostics


__all__ = ["MODEL_NAMES", "build_models"]
