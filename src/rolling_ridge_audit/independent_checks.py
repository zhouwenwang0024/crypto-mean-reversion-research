"""Independent beta and cash-account checks for the rolling Ridge replay."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .data6 import END, N, RESULTS, SYMBOLS, load, minute
    from .models import load_feat, make, SPECS
    from .replay import run_once
except ImportError:
    from data6 import END, N, RESULTS, SYMBOLS, load, minute
    from models import load_feat, make, SPECS
    from replay import run_once


def independent_features(lc: np.ndarray) -> dict[str, np.ndarray]:
    n, p = lc.shape
    pref = np.vstack([np.zeros((1, p)), np.cumsum(lc, axis=0)])
    means = np.full((n, p), np.nan)
    ix = np.arange(180, n)
    means[ix] = (pref[ix] - pref[ix - 180]) / 180
    delta = lc - means
    dev = np.full((n + 1, p), np.nan); center = np.full_like(dev, np.nan); scale = np.full_like(dev, np.nan)
    weights = []
    for t in range(31 * 1440, n + 1, 1440):
        points = np.arange(t - 28 * 1440, t + 1, 60) - 1
        rr = np.diff(lc[points], axis=0)
        sd = rr.std(axis=0, ddof=1); z = (rr - rr.mean(axis=0)) / sd
        w = np.eye(p)
        for j in range(p):
            peers = np.delete(np.arange(p), j)
            x, y = z[:, peers], z[:, j]
            aug_x = np.vstack([x, np.sqrt((len(x) - 1) * 0.1) * np.eye(p - 1)])
            beta = np.linalg.lstsq(aug_x, np.r_[y, np.zeros(p - 1)], rcond=None)[0]
            w[j, peers] = -beta * sd[j] / sd[peers]
        hi = min(t + 1440, n + 1); emitted = np.arange(t, hi); source = emitted - 1
        dev[emitted] = delta[source] @ w.T; center[emitted] = means[source] @ w.T
        scale[emitted] = (delta[t - 7 * 1440:t] @ w.T).std(axis=0, ddof=1)
        weights.append(w)
    return {"dev": dev, "center": center, "scale": scale, "W": np.stack(weights)}


def single_oracle(op, cl, features, volume, start, end, cost_bp=5.0, stop_tolerance=1e-12):
    cost = cost_bp / 10000.0; p = cl.shape[1]
    active = {}; pending = {}; armed = np.ones(p, bool)
    cash = 1.0; eq = np.empty(end - start + 1); tv = np.zeros_like(eq); qtrace = np.zeros((len(eq), p))
    events = []; signals = []
    threshold = np.log1p(0.0025)
    pref = np.vstack([np.zeros((1, p)), np.cumsum(volume, axis=0)])
    for t in range(start, end + 1):
        px = cl[end - 1] if t == end else op[t]
        actions = pending.pop(t, [])
        if t == end: actions = [('exit', j, 4) for j in active]
        delta = np.zeros(p)
        for kind, j, arg in actions: delta[j] += arg['q'] if kind == 'entry' else -active[j]['q']
        fee = cost * np.sum(np.abs(delta * px)); cash -= fee; tv[t - start] = np.sum(np.abs(delta * px))
        for kind, j, arg in actions:
            action_fee = cost * abs(delta[j] * px[j])
            if kind == 'entry':
                lot = arg.copy(); lot.update(entry=t, entry_price=float(px[j]), fee=action_fee, pending=False, gross=abs(lot['q'] * px[j])); active[j] = lot
            else:
                lot = active.pop(j); pnl = lot['q'] * (px[j] - lot['entry_price'])
                cash += pnl; events.append({"target": j, "signal": lot["signal"], "entry": lot["entry"], "exit": t, "q": lot["q"], "entry_price": lot["entry_price"], "exit_price": float(px[j]), "pnl": pnl, "fee": lot["fee"] + action_fee, "reason": arg, "gross": lot["gross"]})
        eq[t - start] = cash + sum(v['q'] * (px[j] - v['entry_price']) for j, v in active.items())
        for j, v in active.items(): qtrace[t - start, j] = v['q']
        if t == end: break
        for j, v in list(active.items()):
            if v['pending'] or v['entry'] >= t: continue
            loss = v['q'] * (cl[t - 1, j] - v['entry_price']) / v['gross']
            support = -np.sign(v['q']) * features['dev'][t, j]
            stop = loss <= -0.03 + stop_tolerance
            timed = t + 1 - v['entry'] >= 240
            if (stop or timed or support <= threshold) and t + 1 < end:
                pending.setdefault(t + 1, []).append(('exit', j, 1 if stop else 2 if timed else 0)); v['pending'] = True
        if t % 5 or t + 1 >= end or eq[t - start] <= 0: continue
        dv, ss = features['dev'][t], features['scale'][t]
        if not np.isfinite(dv).all() or not np.isfinite(ss).all() or not ((pref[t] - pref[t - 5]) > 0).all(): continue
        z = dv / ss; known = cl[t - 1]
        for j in range(p):
            if j not in active and abs(z[j]) < 1: armed[j] = True
        equity = cash + sum(v['q'] * (known[j] - v['entry_price']) for j, v in active.items())
        exposure = sum(abs(v['q'] * known[j]) for j, v in active.items())
        for j in np.argsort(-np.abs(z)):
            if len(active) + sum(a[0] == 'entry' for aa in pending.values() for a in aa) >= 3: break
            gap = np.expm1(dv[j])
            if not armed[j] or j in active or abs(z[j]) < 2 or abs(gap) < .015 or abs(gap) > 1: continue
            budget = min(.3 * equity, .9 * equity - exposure)
            if budget <= equity * 1e-8: break
            q = -np.sign(z[j]) * budget / known[j]
            pending.setdefault(t + 1, []).append(('entry', int(j), {"q": q, "signal": t})); armed[j] = False; exposure += budget
            signals.append({"signal": t, "target": int(j), "q": q, "known_equity": equity})
    return eq, tv, qtrace, pd.DataFrame(events), pd.DataFrame(signals)


def main() -> None:
    op, cl, lc, vol, _ = load(); make(SPECS[0], force=True); f = load_feat("ridge_sma3"); ind = independent_features(lc)
    checks = {}
    for key in ("dev", "center", "scale"):
        finite = np.isfinite(f[key]) & np.isfinite(ind[key]); checks[f"{key}_maxerr"] = float(np.max(np.abs(f[key][finite] - ind[key][finite])))
    fit_days = np.arange(31 * 1440, N + 1, 1440); checks["W_maxerr"] = float(np.max(np.abs(f["W"][np.searchsorted(f["times"], fit_days)] - ind["W"]))); checks["daily_snapshots"] = int(len(fit_days)); checks["per_target_direct_fits"] = int(len(fit_days) * len(SYMBOLS))
    (RESULTS / "feature_crosscheck_local.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    rows = []
    for st, en in (("2026-05-01", "2026-07-01"), ("2026-07-01", "2026-09-01")):
        _, a = run_once(st, en, 5.0, raw=False)
        e, tv, q, trades, signals = single_oracle(op, cl, ind, vol, minute(st), minute(en), 5.0)
        rows.append({"start": st, "end": en, "nav_error": float(np.max(np.abs(e - a[0]))), "quantity_error": float(np.max(np.abs(q - a[5]))), "turnover_error": float(np.max(np.abs(tv - a[1]))), "trades": int(len(trades)), "return": float(e[-1] - 1)})
        trades.to_csv(RESULTS / f"independent_trades_{st}.csv", index=False); signals.to_csv(RESULTS / f"independent_signals_{st}.csv", index=False)
    (RESULTS / "independent_accounts_local.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"features": checks, "accounts": rows}, indent=2))


if __name__ == "__main__": main()
