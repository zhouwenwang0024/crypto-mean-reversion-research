"""Independent audits for the minute validation runner.

This module deliberately does not import the validation runner or use its PnL
summaries.  A runner can provide its feature builder and account function to
the small adapters at the bottom of the file.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


def independent_ridge_features(
    log_close: np.ndarray,
    *,
    update_minutes: int = 1440,
    window_hours: int = 672,
    center_minutes: int = 180,
    scale_days: int = 7,
    penalty: float = 0.1,
    warmup_days: int = 29,
    weight_builder: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Rebuild causal Ridge residual features with direct augmented least squares.

    A snapshot at ``t`` only uses closes through ``t-1``.  ``dev[t]`` is the
    residual of ``close[t-1]`` and the scale uses the seven complete days before
    ``t``.  The implementation is intentionally separate from the model code.
    """
    lc = np.asarray(log_close, dtype=float)
    if lc.ndim != 2 or not np.isfinite(lc).all():
        raise ValueError("log_close must be a finite 2-D array")
    n, p = lc.shape
    pref = np.vstack((np.zeros((1, p)), np.cumsum(lc, axis=0)))
    mean = np.full((n, p), np.nan)
    if center_minutes <= 0 or center_minutes >= n:
        raise ValueError("invalid center_minutes")
    ix = np.arange(center_minutes, n)
    mean[ix] = (pref[ix] - pref[ix - center_minutes]) / center_minutes
    delta = lc - mean
    dev = np.full((n + 1, p), np.nan)
    center = np.full_like(dev, np.nan)
    scale = np.full_like(dev, np.nan)
    times: list[int] = []
    weights: list[np.ndarray] = []
    betas: list[np.ndarray] = []
    fit_rows = window_hours * 60
    scale_rows = scale_days * 1440
    first = max(fit_rows, center_minutes + scale_rows, warmup_days * 1440)
    for t in range(first, n + 1, update_minutes):
        points = np.arange(t - fit_rows, t + 1, 60, dtype=np.int64) - 1
        returns = np.diff(lc[points], axis=0)
        sd = returns.std(axis=0, ddof=1)
        if not np.isfinite(sd).all() or np.any(sd <= 0):
            raise ValueError(f"invalid return scale at minute {t}")
        z = (returns - returns.mean(axis=0)) / sd
        beta_snapshot = np.full((p, p - 1), np.nan)
        if weight_builder is not None:
            w = np.asarray(weight_builder(returns), dtype=float)
        else:
            w = np.eye(p)
            ridge_rows = np.sqrt((len(z) - 1) * penalty) * np.eye(p - 1)
            for target in range(p):
                peers = np.delete(np.arange(p), target)
                beta = np.linalg.lstsq(
                    np.vstack((z[:, peers], ridge_rows)),
                    np.r_[z[:, target], np.zeros(p - 1)],
                    rcond=None,
                )[0]
                beta_snapshot[target] = beta
                w[target, peers] = -beta * sd[target] / sd[peers]
        stop = min(t + update_minutes, n + 1)
        emitted = np.arange(t, stop, dtype=np.int64)
        source = emitted - 1
        dev[emitted] = delta[source] @ w.T
        center[emitted] = mean[source] @ w.T
        scale[emitted] = (delta[t - scale_rows:t] @ w.T).std(axis=0, ddof=1)
        times.append(t)
        weights.append(w)
        betas.append(beta_snapshot)
    if not weights:
        raise ValueError("input is shorter than the feature warmup")
    return {"dev": dev, "center": center, "scale": scale,
            "W": np.stack(weights), "times": np.asarray(times, dtype=np.int64),
            "beta": np.stack(betas)}


def independent_pca_weights(returns: np.ndarray, factors: int,
                            penalty: float = 0.1) -> np.ndarray:
    """Leave-one-target-out PCA hedge using only the supplied return window."""
    r = np.asarray(returns, float); n, p = r.shape
    sd = r.std(axis=0, ddof=1); z = (r - r.mean(axis=0)) / sd
    if not np.isfinite(z).all() or np.any(sd <= 0) or factors < 1:
        raise ValueError("invalid PCA return window")
    out = np.eye(p)
    for target in range(p):
        peers = np.delete(np.arange(p), target); _, _, vh = np.linalg.svd(z[:, peers], full_matrices=False)
        v = vh[:min(factors, len(peers))].T; x = z[:, peers] @ v
        ridge = np.sqrt((n - 1) * penalty) * np.eye(v.shape[1])
        beta = np.linalg.lstsq(np.vstack((x, ridge)), np.r_[z[:, target], np.zeros(v.shape[1])], rcond=None)[0]
        out[target, peers] = -(sd[target] / sd[peers]) * (v @ beta)
    return out


def independent_pca_features(log_close: np.ndarray, factors: int, **kwargs: Any) -> dict[str, np.ndarray]:
    """Build the same causal feature arrays with an independent PCA hedge."""
    return independent_ridge_features(
        log_close, weight_builder=lambda r: independent_pca_weights(r, factors, kwargs.get("penalty", 0.1)), **kwargs)


def compare_features(actual: Mapping[str, Any], independent: Mapping[str, Any],
                     *, atol: float = 1e-9) -> dict[str, float | bool]:
    """Return finite maximum errors for the independent feature cross-check."""
    out: dict[str, float | bool] = {}
    for key in ("dev", "center", "scale"):
        a, b = np.asarray(actual[key]), np.asarray(independent[key])
        finite = np.isfinite(a) & np.isfinite(b)
        out[f"{key}_max_error"] = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else np.nan
    a, b = np.asarray(actual["W"]), np.asarray(independent["W"])
    if "times" in actual and "times" in independent:
        ai = {int(t): i for i, t in enumerate(np.asarray(actual["times"]))}
        pairs = [(ai[int(t)], i) for i, t in enumerate(np.asarray(independent["times"])) if int(t) in ai]
        av = np.stack([a[i] for i, _ in pairs]) if pairs else np.empty((0,))
        bv = np.stack([b[j] for _, j in pairs]) if pairs else np.empty((0,))
    else:
        ai = {i * 1440: i for i in range(len(a)) if np.isfinite(a[i]).all()}
        bt = np.asarray(independent.get("times", np.arange(len(b)) * 1440))
        bi = {int(t): i for i, t in enumerate(bt) if np.isfinite(b[i]).all()}
        pairs = [(ai[t], bi[t]) for t in bi if t in ai]
        av = np.stack([a[i] for i, _ in pairs]) if pairs else np.empty((0,))
        bv = np.stack([b[j] for _, j in pairs]) if pairs else np.empty((0,))
    out["W_max_error"] = float(np.max(np.abs(av - bv))) if len(av) and len(bv) else np.nan
    if "beta" in actual and "beta" in independent:
        x, y = np.asarray(actual["beta"]), np.asarray(independent["beta"]); m = min(len(x), len(y))
        finite = np.isfinite(x[:m]) & np.isfinite(y[:m]); out["beta_max_error"] = float(np.max(np.abs(x[:m][finite] - y[:m][finite]))) if finite.any() else np.nan
    checks = ["dev", "center", "scale", "W"] + (["beta"] if "beta_max_error" in out else [])
    out["pass"] = bool(all(float(out[f"{k}_max_error"]) <= atol for k in checks))
    return out


def _trade_quantities(trades: Any, n: int) -> np.ndarray:
    """Extract the entry quantity vectors from an engine trade array/table."""
    if isinstance(trades, np.ndarray):
        if trades.ndim != 2 or trades.shape[1] < 18 + n:
            raise ValueError("trade array has no quantity columns")
        return np.asarray(trades[:, 18:18 + n], dtype=float)
    if hasattr(trades, "columns"):
        names = list(trades.columns)
        for prefix in ("q_", "qty_", "quantity_"):
            cols = [f"{prefix}{j}" for j in range(n)]
            if all(c in names for c in cols):
                return trades[cols].to_numpy(float)
        cols = [c for c in names if c.startswith("q_")]
        if len(cols) == n:
            return trades[cols].to_numpy(float)
    raise ValueError("trades must expose q_0..q_n quantity columns or engine columns")


def replay_ledger(
    op: np.ndarray,
    cl: np.ndarray,
    fund: np.ndarray | None,
    trades: Any,
    start: int,
    end: int,
    *,
    cost_bp: float,
    quantity_sign: float = 1.0,
    fee_bp_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay cash and inventory from quantities and fill timestamps only.

    Trade PnL, fee and funding columns are ignored.  Prices, signed quantities,
    funding cash flow and net order turnover are recomputed independently.
    """
    op, cl = np.asarray(op, float), np.asarray(cl, float)
    p = cl.shape[1]; q = _trade_quantities(trades, p) * quantity_sign
    def col(name: str, i: int) -> np.ndarray:
        if isinstance(trades, np.ndarray): return trades[:, i].astype(np.int64)
        return trades[name].to_numpy(np.int64)
    entries = col("entry_bar", 0); exits = col("exit_bar", 1)
    enter_at: dict[int, list[int]] = {}; exit_at: dict[int, list[int]] = {}
    for i in range(len(q)):
        enter_at.setdefault(int(entries[i]), []).append(i)
        exit_at.setdefault(int(exits[i]), []).append(i)
    active: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    cash = 1.0; eq = np.ones(end - start + 1); tape = np.zeros((end - start + 1, p))
    fee_rate = (cost_bp if fee_bp_override is None else fee_bp_override) / 10000.0
    for t in range(start, end + 1):
        px = cl[end - 1].copy() if t == end else op[t].copy()
        netq = np.sum([v[0] for v in active.values()], axis=0) if active else np.zeros(p)
        if fund is not None and t < end and t < len(fund) and active:
            cash += float(netq @ np.asarray(fund[t], float))
        actions: list[tuple[str, int, np.ndarray]] = []
        for i in exit_at.get(t, []):
            if i not in active: raise AssertionError("trade exit has no active entry")
            lot = active[i]; cash += float(lot[0] @ (px - lot[1]))
            actions.append(("exit", i, -lot[0])); del active[i]
        for i in enter_at.get(t, []):
            if i in active: raise AssertionError("duplicate active trade")
            actions.append(("entry", i, q[i]))
        delta = np.sum([a[2] for a in actions], axis=0) if actions else np.zeros(p)
        if actions: cash -= fee_rate * float(np.sum(np.abs(delta * px)))
        for kind, i, signed in actions:
            if kind == "entry": active[i] = (signed, px.copy())
        mark = cash + sum(float(v[0] @ (px - v[1])) for v in active.values())
        eq[t - start] = mark; tape[t - start] = np.sum([v[0] for v in active.values()], axis=0) if active else 0.0
    return eq, tape


def reconcile(engine_eq: np.ndarray, engine_tape: np.ndarray,
              ledger_eq: np.ndarray, ledger_tape: np.ndarray, *, atol: float = 1e-8) -> dict[str, float | bool]:
    e = float(np.max(np.abs(np.asarray(engine_eq) - ledger_eq)))
    q = float(np.max(np.abs(np.asarray(engine_tape) - ledger_tape)))
    return {"equity_max_error": e, "quantity_max_error": q, "pass": bool(e <= atol and q <= atol)}


def assert_negative_mutants(
    engine_eq: np.ndarray, engine_tape: np.ndarray, ledger_eq: np.ndarray,
    ledger_tape: np.ndarray, *, op: np.ndarray, cl: np.ndarray,
    fund: np.ndarray | None, trades: Any, start: int, end: int, cost_bp: float,
    atol: float = 1e-8,
) -> dict[str, bool]:
    """Ensure a real direction reversal and fee omission are rejected."""
    if not reconcile(engine_eq, engine_tape, ledger_eq, ledger_tape, atol=atol)["pass"]:
        raise AssertionError("base independent ledger does not reconcile")
    reverse_eq, reverse_tape = replay_ledger(op, cl, fund, trades, start, end, cost_bp=cost_bp, quantity_sign=-1)
    nofee_eq, nofee_tape = replay_ledger(op, cl, fund, trades, start, end, cost_bp=cost_bp, fee_bp_override=0.0)
    reverse_rejected = not bool(reconcile(engine_eq, engine_tape, reverse_eq, reverse_tape, atol=atol)["pass"])
    fee_rejected = cost_bp > 0 and not bool(reconcile(engine_eq, engine_tape, nofee_eq, nofee_tape, atol=atol)["pass"])
    if not reverse_rejected or not fee_rejected:
        raise AssertionError("negative ledger mutant was accepted")
    return {"reverse_quantity_rejected": reverse_rejected, "omitted_fee_rejected": fee_rejected}


def audit_account(engine_result: tuple[Any, ...], op: np.ndarray, cl: np.ndarray,
                  fund: np.ndarray | None, trades: Any, start: int, end: int,
                  *, cost_bp: float, atol: float = 1e-8) -> dict[str, Any]:
    """One-call independent ledger reconciliation plus negative mutants."""
    ledger_eq, ledger_tape = replay_ledger(op, cl, fund, trades, start, end, cost_bp=cost_bp)
    base = reconcile(engine_result[0], engine_result[-1], ledger_eq, ledger_tape, atol=atol)
    mutants = assert_negative_mutants(
        engine_result[0], engine_result[-1], ledger_eq, ledger_tape,
        op=op, cl=cl, fund=fund, trades=trades, start=start, end=end,
        cost_bp=cost_bp, atol=atol)
    return {**base, **mutants}


def future_suffix_check(
    op: np.ndarray, cl: np.ndarray, fund: np.ndarray | None, *, cut: int,
    start: int, end: int, build_features: Callable[[np.ndarray], Mapping[str, Any]],
    run_account: Callable[..., Any], perturb: Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, float | bool]:
    """Run a suffix price perturbation and compare feature/account prefixes."""
    if perturb is None:
        alt_cl = np.asarray(cl, float).copy(); k = np.arange(len(cl) - cut)[:, None]
        alt_cl[cut:] *= np.exp(0.1 * np.sin(k / 137.0))
        alt_op = np.asarray(op, float).copy(); alt_op[cut + 1:] *= alt_cl[cut + 1:] / cl[cut + 1:]
    else:
        alt_cl, alt_op = perturb(np.asarray(cl, float), np.asarray(op, float), cut)
    base_feat, alt_feat = build_features(np.log(cl)), build_features(np.log(alt_cl))
    def run(o, c, f):
        return run_account(o, c, fund, f, start, end)
    base = run(op, cl, base_feat); alt = run(alt_op, alt_cl, alt_feat)
    upto = max(0, min(cut - start + 1, len(base[0]), len(alt[0])))
    def prefix_error(key: str) -> float:
        x, y = np.asarray(base_feat[key]), np.asarray(alt_feat[key])
        finite = np.isfinite(x[:cut + 1]) & np.isfinite(y[:cut + 1])
        return float(np.max(np.abs(x[:cut + 1][finite] - y[:cut + 1][finite]))) if finite.any() else 0.0
    feature_error, scale_error = prefix_error("dev"), prefix_error("scale")
    bw, aw = np.asarray(base_feat["W"]), np.asarray(alt_feat["W"])
    if "times" in base_feat and "times" in alt_feat:
        bi = np.asarray(base_feat["times"]) <= cut; ai = np.asarray(alt_feat["times"]) <= cut
        weights_error = float(np.max(np.abs(bw[bi][:min(bi.sum(), ai.sum())] - aw[ai][:min(bi.sum(), ai.sum())]))) if bi.any() and ai.any() else 0.0
    else:
        bi = np.arange(len(bw)) * 1440 <= cut
        ai = np.arange(len(aw)) * 1440 <= cut
        bf = bw[bi & np.isfinite(bw).all(axis=(1, 2))]; af = aw[ai & np.isfinite(aw).all(axis=(1, 2))]
        m = min(len(bf), len(af)); weights_error = float(np.max(np.abs(bf[:m] - af[:m]))) if m else 0.0
    eq_error = float(np.max(np.abs(np.asarray(base[0])[:upto] - np.asarray(alt[0])[:upto])))
    tape_error = float(np.max(np.abs(np.asarray(base[-1])[:upto] - np.asarray(alt[-1])[:upto]))) if len(base) > 1 else np.nan
    changed = float(np.max(np.abs(np.asarray(base[0])[upto:] - np.asarray(alt[0])[upto:]))) if upto < len(base[0]) else 0.0
    return {"feature_prefix_max_error": feature_error, "scale_prefix_max_error": scale_error,
            "weights_prefix_max_error": weights_error, "account_prefix_max_error": eq_error,
            "tape_prefix_max_error": tape_error, "future_equity_change": changed,
            "pass": bool(feature_error <= 1e-9 and scale_error <= 1e-9 and weights_error <= 1e-9 and eq_error <= 1e-9 and changed > 1e-9)}
