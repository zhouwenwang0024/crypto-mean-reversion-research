"""Causal peer-price prediction and residual diagnostics.

At each calendar month, every target coin is regressed on the other coins' log
prices using a past fit window.  The immediately preceding calibration window
is out-of-sample: it is used only to measure whether the peer-implied price is
useful (residual stationarity, half-life, prediction error and stability).
Coefficients are frozen until the next formation month.
"""
from __future__ import annotations

import json
from itertools import chain
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def _periods(index: pd.DatetimeIndex, fit_days: int, cal_days: int,
             start: str | pd.Timestamp | None,
             end: str | pd.Timestamp | None) -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    first = idx.min().tz_convert(None).to_period("M").to_timestamp().tz_localize("UTC")
    last = idx.max().tz_convert(None).to_period("M").to_timestamp().tz_localize("UTC")
    lo = (pd.Timestamp(start, tz="UTC") if start is not None
          else first + pd.Timedelta(days=fit_days + cal_days))
    hi = (pd.Timestamp(end, tz="UTC") if end is not None
          else last + pd.offsets.MonthBegin(1))
    lo = lo.tz_convert(None).to_period("M").to_timestamp().tz_localize("UTC")
    hi = hi.tz_convert(None).to_period("M").to_timestamp().tz_localize("UTC")
    return list(pd.date_range(lo, hi, freq="MS", tz="UTC", inclusive="left"))


def _ridge_fit(y: np.ndarray, x: np.ndarray, alpha: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Ridge on standardized peer levels, returning prediction parameters."""
    if y.ndim != 1 or x.ndim != 2 or len(y) != len(x):
        raise ValueError("invalid regression shapes")
    if len(y) < 3 or not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("non-finite regression data")
    mu_x = x.mean(0)
    sd_x = x.std(0, ddof=1)
    sd_x = np.where(sd_x > 1e-12, sd_x, 1.0)
    mu_y = float(y.mean())
    z = (x - mu_x) / sd_x
    gram = z.T @ z + float(alpha) * max(len(y) - 1, 1) * np.eye(x.shape[1])
    coef_z = np.linalg.solve(gram, z.T @ (y - mu_y))
    coef = coef_z / sd_x
    intercept = mu_y - float(mu_x @ coef)
    return float(intercept), coef, mu_x, sd_x


def _predict(x: np.ndarray, intercept: float, coef: np.ndarray) -> np.ndarray:
    return float(intercept) + np.asarray(x) @ np.asarray(coef)


def _adf(resid: np.ndarray) -> float:
    try:
        return float(adfuller(resid, maxlag=1, regression="c", autolag=None,
                              result_object=False)[1])
    except Exception:
        return 1.0


def _ar1(resid: np.ndarray) -> tuple[float, float]:
    x = np.asarray(resid, float)
    z = x - x.mean()
    den = float(z[:-1] @ z[:-1])
    if den <= 1e-14:
        return np.nan, np.nan
    rho = float((z[1:] @ z[:-1]) / den)
    half = float(-np.log(2.0) / np.log(rho)) if 0.0 < rho < 1.0 else np.nan
    return rho, half


def _quality(actual: np.ndarray, pred: np.ndarray, *, min_obs: int = 100) -> dict:
    resid = np.asarray(actual, float) - np.asarray(pred, float)
    if len(resid) < min_obs or not np.isfinite(resid).all():
        raise ValueError("insufficient calibration observations")
    n = len(resid)
    rmse = float(np.sqrt(np.mean(resid * resid)))
    mae = float(np.mean(np.abs(resid)))
    var = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1.0 - np.sum(resid * resid) / var) if var > 1e-14 else np.nan
    corr = float(np.corrcoef(actual, pred)[0, 1]) if np.std(pred) > 1e-12 else np.nan
    rho, half = _ar1(resid)
    h = n // 2
    s1 = float(np.std(resid[:h], ddof=1)); s2 = float(np.std(resid[h:], ddof=1))
    m1 = float(np.mean(resid[:h])); m2 = float(np.mean(resid[h:]))
    scale_ratio = max(s1, s2) / max(min(s1, s2), 1e-12)
    return {
        "cal_obs": int(n), "cal_resid_mean": float(resid.mean()),
        "cal_resid_scale": float(np.std(resid, ddof=1)),
        "cal_rmse_log": rmse, "cal_mae_log": mae, "cal_r2": r2,
        "cal_corr": corr, "resid_adf_pvalue": _adf(resid), "resid_rho": rho,
        "resid_half_life_hours": half, "resid_scale_1": s1, "resid_scale_2": s2,
        "resid_mean_1": m1, "resid_mean_2": m2, "resid_scale_ratio": scale_ratio,
        "median_abs_error_pct": float(np.median(np.abs(np.expm1(resid))) * 100.0),
        "p95_abs_error_pct": float(np.quantile(np.abs(np.expm1(resid)), 0.95) * 100.0),
    }


def _eligible(q: dict, *, adf_p: float = 0.10, min_r2: float = 0.0,
              half_min: float = 6.0, half_max: float = 168.0,
              max_scale_ratio: float = 2.5) -> bool:
    return bool(
        np.isfinite([q["cal_r2"], q["cal_corr"], q["resid_half_life_hours"],
                     q["cal_resid_scale"]]).all()
        and q["cal_r2"] >= min_r2 and q["resid_adf_pvalue"] <= adf_p
        and half_min <= q["resid_half_life_hours"] <= half_max
        and q["cal_resid_scale"] > 0.0 and q["resid_scale_ratio"] <= max_scale_ratio
    )


def _bh(values: Iterable[float], q: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    p = np.clip(np.nan_to_num(np.asarray(list(values), float), nan=1.0), 0.0, 1.0)
    order = np.argsort(p, kind="mergesort")
    ranks = np.arange(1, len(p) + 1, dtype=float)
    adj_sorted = np.minimum.accumulate((p[order] * len(p) / ranks)[::-1])[::-1]
    adj = np.empty_like(p); adj[order] = np.minimum(adj_sorted, 1.0)
    return adj, adj <= q


def build_target_models(
    close: np.ndarray | pd.DataFrame,
    index: pd.DatetimeIndex | None = None,
    *, fit_days: int = 60,
    cal_days: int = 28,
    symbols: list[str] | None = None,
    formation_start: str | pd.Timestamp | None = None,
    formation_end: str | pd.Timestamp | None = None,
    ridge_alpha: float = 10.0,
    estimators: tuple[str, ...] = ("ridge", "ols"),
    min_r2: float = 0.0,
    adf_p: float = 0.10,
    max_scale_ratio: float = 2.5,
    hourly: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return frozen monthly target models and their out-of-sample diagnostics."""
    if isinstance(close, pd.DataFrame):
        if index is None:
            index = pd.DatetimeIndex(close.index)
        values = close.to_numpy(float)
        symbols = symbols or [str(x) for x in close.columns]
    else:
        values = np.asarray(close, float)
    if index is None or values.ndim != 2 or len(index) != len(values):
        raise ValueError("close/index shape mismatch")
    if values.shape[1] < 2 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("close must be finite and positive")
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    symbols = symbols or [str(i) for i in range(values.shape[1])]
    if len(symbols) != values.shape[1]:
        raise ValueError("symbols length mismatch")
    order = np.argsort(idx.view("i8")); idx = idx[order]; values = values[order]
    frame = pd.DataFrame(np.log(values), index=idx, columns=symbols)
    if hourly:
        frame = frame.resample("1h", label="left", closed="left").last().dropna(how="any")
    rows: list[dict] = []
    for month in _periods(frame.index, fit_days, cal_days, formation_start, formation_end):
        cal_start, fit_end = month - pd.Timedelta(days=cal_days), month - pd.Timedelta(days=cal_days)
        fit_start = fit_end - pd.Timedelta(days=fit_days)
        fit = frame[(frame.index >= fit_start) & (frame.index < fit_end)]
        cal = frame[(frame.index >= cal_start) & (frame.index < month)]
        if len(fit) < fit_days * 24 or len(cal) < cal_days * 24:
            continue
        for target in range(frame.shape[1]):
            peers = [j for j in range(frame.shape[1]) if j != target]
            row = {
                "target": target, "target_symbol": symbols[target], "model": "",
                "model_end_time": month, "fit_start": fit_start, "fit_end": fit_end,
                "cal_start": cal_start, "cal_end": month, "ridge_alpha": float(ridge_alpha),
                "feature_indices": json.dumps(peers),
                "feature_symbols": json.dumps([symbols[j] for j in peers]),
                "intercept": np.nan, "coef_json": "", "eligible": False, "error": "",
            }
            for estimator in estimators:
                current = row.copy()
                current["model"] = f"{estimator}_peers"
                try:
                    alpha = float(ridge_alpha) if estimator == "ridge" else 0.0
                    intercept, coef, mu_x, sd_x = _ridge_fit(
                        fit.iloc[:, target].to_numpy(), fit.iloc[:, peers].to_numpy(), alpha
                    )
                    pred = _predict(cal.iloc[:, peers].to_numpy(), intercept, coef)
                    quality = _quality(cal.iloc[:, target].to_numpy(), pred)
                    current.update({"intercept": intercept,
                                    "coef_json": json.dumps(coef.tolist(), separators=(",", ":")),
                                    "hedge_weights_json": json.dumps(
                                        {symbols[j]: float(-coef[k]) for k, j in enumerate(peers)},
                                        sort_keys=True, separators=(",", ":")
                                    ),
                                    "feature_mean_json": json.dumps(mu_x.tolist(), separators=(",", ":")),
                                    "feature_scale_json": json.dumps(sd_x.tolist(), separators=(",", ":")),
                                    **quality})
                    current["eligible"] = _eligible(current, adf_p=adf_p, min_r2=min_r2,
                                                      max_scale_ratio=max_scale_ratio)
                except Exception as exc:
                    current["error"] = type(exc).__name__ + ": " + str(exc)[:180]
                rows.append(current)
    diagnostics = pd.DataFrame(rows)
    if diagnostics.empty:
        return diagnostics.copy(), diagnostics
    diagnostics["adf_bh_qvalue"] = 1.0
    diagnostics["adf_bh_pass"] = False
    for (_, model), ids in diagnostics.groupby(["model_end_time", "model"], sort=False).groups.items():
        qvals, passed = _bh(diagnostics.loc[ids, "resid_adf_pvalue"].to_numpy(), adf_p)
        diagnostics.loc[ids, "adf_bh_qvalue"] = qvals
        diagnostics.loc[ids, "adf_bh_pass"] = passed
    diagnostics["eligible"] &= diagnostics["adf_bh_pass"]
    selected = diagnostics[diagnostics["eligible"]].copy().reset_index(drop=True)
    return selected, diagnostics


def expand_predictions(frame: pd.DataFrame, close: np.ndarray | pd.DataFrame,
                       index: pd.DatetimeIndex, *, hourly: bool = True) -> pd.DataFrame:
    """Expand frozen selected models into causal calibration/trading predictions.

    The returned rows start at each model's formation month and use only the
    frozen coefficients.  A caller should execute a signal from row ``t`` at
    the next bar, avoiding same-bar close look-ahead.
    """
    if frame.empty:
        return pd.DataFrame()
    if isinstance(close, pd.DataFrame):
        values = close.to_numpy(float); symbols = [str(c) for c in close.columns]
    else:
        values = np.asarray(close, float); symbols = [str(i) for i in range(values.shape[1])]
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    order = np.argsort(idx.view("i8")); idx, values = idx[order], values[order]
    logf = pd.DataFrame(np.log(values), index=idx, columns=symbols)
    if hourly:
        logf = logf.resample("1h", label="left", closed="left").last().dropna(how="any")
    out: list[pd.DataFrame] = []
    records = frame.to_dict("records")
    # A frozen row is valid only until the next formation for the same target
    # and estimator.  Without this bound, monthly rows would overlap and a
    # downstream 5-minute ledger would see duplicate predictions.
    starts = {}
    for rec in records:
        key = (str(rec.get("model", "")), str(rec["target_symbol"]))
        ts = pd.Timestamp(rec["model_end_time"])
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        starts.setdefault(key, []).append(ts)
    for key in starts:
        starts[key].sort()
    for row in records:
        target, peers = int(row["target"]), json.loads(row["feature_indices"])
        start = pd.Timestamp(row["model_end_time"])
        if start.tzinfo is None: start = start.tz_localize("UTC")
        else: start = start.tz_convert("UTC")
        key = (str(row.get("model", "")), str(row["target_symbol"]))
        future = [x for x in starts.get(key, []) if x > start]
        stop = future[0] if future else logf.index.max() + pd.Timedelta(minutes=5)
        sub = logf[(logf.index >= start) & (logf.index < stop)]
        if sub.empty: continue
        coef = np.asarray(json.loads(row["coef_json"]), float)
        pred = _predict(sub.iloc[:, peers].to_numpy(), float(row["intercept"]), coef)
        actual = sub.iloc[:, target].to_numpy()
        out.append(pd.DataFrame({
            "timestamp": sub.index, "target": row["target_symbol"],
            "actual_log_price": actual, "predicted_log_price": pred,
            "residual": actual - pred, "formation_time": start,
            "residual_scale": float(row.get("cal_resid_scale", np.nan)),
            "resid_adf_pvalue": float(row.get("resid_adf_pvalue", 1.0)),
            "cal_r2": float(row.get("cal_r2", np.nan)),
            "hedge_weights_json": row.get("hedge_weights_json", "{}"),
            "eligible": bool(row.get("eligible", False)),
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


__all__ = ["build_target_models", "expand_predictions"]
