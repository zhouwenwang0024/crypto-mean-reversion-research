"""Expanded retrospective mean-reversion study.

This layer reuses the audited v2 ledger and compares price-center formulas,
lookback periods, and entry/exit rules.  July--August was viewed in earlier
studies, so it is a reused evaluation period, not a fresh statistical holdout.
The mechanical selection rule ranks only May--June. All results are retained.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v2 import (
    END, INITIAL, RESULTS, START, SYMBOLS, TRAIN_END, VALID_END,
    Feature, backtest, feature_grid, load_funding_events, load_prices,
    make_bars, metrics, residual_feature, self_feature, peer_feature,
)


def median_peer_feature(close: pd.DataFrame, frequency: int) -> Feature:
    """Cumulative leave-one-out median-return residual.

    Returns make the signal invariant to each coin's arbitrary price unit.
    Execution still uses the fixed equal-peer hedge as a low-turnover proxy.
    """
    ret = np.log(close).diff().fillna(0.0).to_numpy()
    peer = np.column_stack([np.median(np.delete(ret, i, axis=1), axis=1)
                            for i in range(ret.shape[1])])
    level = pd.DataFrame(np.cumsum(ret - peer, axis=0), index=close.index, columns=SYMBOLS)
    hedge = np.full((len(SYMBOLS), len(SYMBOLS)), -0.5 / (len(SYMBOLS) - 1))
    np.fill_diagonal(hedge, 0.5)
    days = {pd.Timestamp(d): hedge for d in close.index.normalize().unique()}
    return Feature("MEDIAN", frequency, close, level, days, True)


def ar1_feature(close: pd.DataFrame, frequency: int, window_days: int = 7,
                horizon_hours: int = 4) -> Feature:
    """Causal AR(1) forecast residual, frozen one bar at a time.

    The forecast is fit from the prior window only.  It is a fixed control,
    not a family whose horizon is selected on the holdout.
    """
    level = np.log(close).to_numpy(float)
    n = max(20, window_days * 24 * 60 // frequency)
    horizon = max(1, horizon_hours * 60 // frequency)
    out = np.full_like(level, np.nan)
    for t in range(n + 1, len(level)):
        h = level[t - n:t]
        mu = np.nanmean(h, axis=0)
        x, y = h[:-1], h[1:]
        xc, yc = x - mu, y - mu
        den = np.nansum(xc * xc, axis=0)
        rho = np.divide(np.nansum(xc * yc, axis=0), den,
                        out=np.zeros(level.shape[1]), where=den > 1e-12)
        rho = np.clip(np.nan_to_num(rho), 0.0, 0.995)
        fair = mu + rho ** horizon * (level[t - 1] - mu)
        out[t] = level[t] - fair
    hedge_by_day = {pd.Timestamp(d): np.eye(len(SYMBOLS))
                    for d in close.index.normalize().unique()}
    return Feature("AR1", frequency,
                   close, pd.DataFrame(out, index=close.index, columns=SYMBOLS),
                   hedge_by_day, False)


def score(level: pd.DataFrame, frequency: int, center_hours: int,
          scale_days: int, center_kind: str = "sma") -> pd.DataFrame:
    """Causal z-score with an explicitly chosen price-center formula."""
    center_n = max(1, round(center_hours * 60 / frequency))
    scale_n = max(center_n + 1, round(scale_days * 24 * 60 / frequency))
    min_center = max(1, center_n // 2)
    if center_kind == "sma":
        raw_center = level.rolling(center_n, min_periods=min_center).mean()
    elif center_kind == "ewma":
        raw_center = level.ewm(span=center_n, adjust=False,
                               min_periods=min_center).mean()
    elif center_kind == "median":
        raw_center = level.rolling(center_n, min_periods=min_center).median()
    else:
        raise ValueError(center_kind)
    # Scale the causal residual, rather than the trending raw level.  For
    # every historical bar s, its center is also computed from s-1 and before;
    # the final shift keeps the current bar out of its own scale estimate.
    residual = level - raw_center.shift(1)
    scale = residual.rolling(scale_n, min_periods=max(10, scale_n // 2)).std().shift(1)
    return residual / scale.replace(0, np.nan)


def entry_mask(z: pd.DataFrame, rule: str, threshold: float = 2.0) -> pd.DataFrame | None:
    """Pre-signal filters; every shift is backward-looking."""
    if rule in {"band", "wide_exit", "zero_cross", "ewma", "median"}:
        return None
    absz = z.abs()
    if rule == "confirm2":
        return (absz >= threshold) & (absz.shift(1) >= threshold) & (np.sign(z) == np.sign(z.shift(1)))
    if rule == "turn":
        return (absz >= threshold) & (np.sign(z) * z.diff(1) < 0)
    if rule == "slope4":
        return (absz >= threshold) & (np.sign(z) * (z - z.shift(4)) < 0)
    raise ValueError(rule)


def feature_set(close1: pd.DataFrame, volume1: pd.DataFrame) -> dict[tuple[str, int], Feature]:
    out: dict[tuple[str, int], Feature] = {}
    for frequency in (5, 15, 60):
        c = make_bars(close1, volume1, frequency)
        out[("B0", frequency)] = self_feature(c, frequency)
        out[("PEER", frequency)] = peer_feature(c, frequency)
        out[("MEDIAN", frequency)] = median_peer_feature(c, frequency)
        out[("RIDGE", frequency)] = residual_feature(c, frequency, "RIDGE")
        out[("PCA1", frequency)] = residual_feature(c, frequency, "PCA1")
        out[("PCA3", frequency)] = residual_feature(c, frequency, "PCA3")
        out[("PCA5", frequency)] = residual_feature(c, frequency, "PCA5")
        if frequency in (15, 60):
            out[("AR1", frequency)] = ar1_feature(c, frequency)
    return out


def add_config(rows: list[dict], model: str, frequency: int, center_hours: int,
               scale_days: int, center_kind: str, logic: str, stage: str,
               entry: float = 2.0, exit_: float = 0.5,
               hold_hours: int = 4, exit_rule: str = "band") -> None:
    rows.append({"model": model, "frequency_min": frequency,
                 "center_hours": center_hours, "scale_days": scale_days,
                 "center_kind": center_kind, "logic": logic,
                 "entry_sigma": entry, "exit_sigma": exit_,
                 "hold_hours": hold_hours, "exit_rule": exit_rule,
                 "stage": stage})


def configs() -> list[dict]:
    rows: list[dict] = []
    # Stage A: formula and signal-frequency comparison at the common baseline.
    for model in ("B0", "PEER", "MEDIAN", "RIDGE", "PCA3"):
        for f in (5, 15, 60):
            add_config(rows, model, f, 4, 7, "sma", "band", "formula_frequency")
    for model in ("PCA1", "PCA5"):
        add_config(rows, model, 15, 4, 7, "sma", "band", "factor_control")
    for model in ("AR1",):
        for f in (15, 60):
            add_config(rows, model, f, 4, 7, "sma", "band", "ar1_control")
    # Stage B: price-center and scale periods, fixed 15m signal.
    for model in ("B0", "PEER", "MEDIAN", "PCA3", "AR1"):
        for center, scale in ((1, 1), (4, 7), (12, 14)):
            add_config(rows, model, 15, center, scale, "sma", "band", "period")
    # Stage C: independent entry/exit logic controls at two frequencies.
    for model in ("B0", "PEER", "MEDIAN", "PCA3"):
        for f in (15, 60):
            add_config(rows, model, f, 4, 7, "sma", "band", "logic")
            add_config(rows, model, f, 4, 7, "sma", "wide_exit", "logic", exit_=1.0)
            add_config(rows, model, f, 4, 7, "sma", "confirm2", "logic")
            add_config(rows, model, f, 4, 7, "sma", "turn", "logic")
            add_config(rows, model, f, 4, 7, "sma", "zero_cross", "logic", exit_=0.0, exit_rule="zero_cross")
    # Stage D: fixed EWMA/median center controls, no result-based tuning.
    for model in ("B0", "PEER", "PCA3"):
        for kind in ("ewma", "median"):
            add_config(rows, model, 15, 4, 7, kind, kind, "center_control")
    operational = ("model", "frequency_min", "center_hours", "scale_days",
                   "center_kind", "logic", "entry_sigma", "exit_sigma",
                   "hold_hours", "exit_rule")
    dedup: dict[tuple, dict] = {}
    for row in rows:
        key = tuple(row[k] for k in operational)
        dedup.setdefault(key, row)
    return list(dedup.values())


def run() -> None:
    opens, closes, volumes = load_prices()
    funding = load_funding_events()
    features = feature_set(closes, volumes)
    grid = configs()
    (RESULTS / "v3_experiment_manifest.json").write_text(
        json.dumps(grid, indent=2), encoding="utf-8")
    rows = []
    daily_equity = {}
    cache: dict[tuple, pd.DataFrame] = {}
    masks: dict[tuple, pd.DataFrame | None] = {}
    for n, cfg in enumerate(grid, 1):
        key = (cfg["model"], cfg["frequency_min"], cfg["center_hours"],
               cfg["scale_days"], cfg["center_kind"])
        ft = features[(cfg["model"], cfg["frequency_min"])]
        if key not in cache:
            cache[key] = score(ft.level, cfg["frequency_min"], cfg["center_hours"],
                               cfg["scale_days"], cfg["center_kind"])
        z = cache[key]
        mkey = (key, cfg["logic"], cfg["entry_sigma"])
        if mkey not in masks:
            masks[mkey] = entry_mask(z, cfg["logic"], cfg["entry_sigma"])
        eq, trades, meta = backtest(
            ft, z, opens, closes, cfg["entry_sigma"], cfg["exit_sigma"],
            cfg["hold_hours"], 2.0, flat_boundaries=(TRAIN_END, VALID_END),
            funding_events=funding, entry_mask=masks[mkey],
            exit_rule=cfg["exit_rule"], fee_mode="gross")
        row = {**cfg, "cost_bp": 2.0}
        for name, a, b in (("train", START, TRAIN_END),
                           ("valid", TRAIN_END, VALID_END),
                           ("holdout", VALID_END, END),
                           ("all", START, END)):
            row.update({f"{name}_{k}": v for k, v in metrics(eq, trades, a, b, meta["funding_rows"]).items()})
        row["funding_cash"] = meta["funding_cash"]
        row["all_gross_fees"] = float(sum(x.get("gross_fee", 0.0) for x in meta["ledger"]))
        row["all_net_fees_reference"] = float(sum(x.get("net_fee", 0.0) for x in meta["ledger"]))
        row["run_index"] = n
        rows.append(row)
        daily_equity[str(n)] = eq.resample("1D").last()
        print(f"{n}/{len(grid)} {cfg['model']} {cfg['frequency_min']}m {cfg['center_kind']} {cfg['logic']}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "v3_grid_2bp.csv", index=False)
    pd.DataFrame(daily_equity).to_csv(RESULTS / "v3_all_daily_equity_2bp.csv", index_label="time")

    # Cost sensitivity is restricted to fixed controls plus the validation leaders.
    controls = table[(table.stage == "formula_frequency") &
                     (table.frequency_min == 15) & (table.center_kind == "sma")]
    leaders = table[table.valid_trades >= 30].sort_values("valid_return", ascending=False).head(5)
    selected = pd.concat([controls, leaders]).drop_duplicates(subset=[
        "model", "frequency_min", "center_hours", "scale_days", "center_kind",
        "logic", "entry_sigma", "exit_sigma", "hold_hours", "exit_rule"])
    selected_rows = []
    for rank, (_, cfg) in enumerate(selected.reset_index(drop=True).iterrows(), 1):
        ft = features[(cfg.model, int(cfg.frequency_min))]
        key = (cfg.model, int(cfg.frequency_min), int(cfg.center_hours),
               int(cfg.scale_days), cfg.center_kind)
        z = cache[key]; mask = masks[(key, cfg.logic, float(cfg.entry_sigma))]
        for bp in (0.0, 1.0, 2.0, 3.0):
            eq, trades, meta = backtest(
                ft, z, opens, closes, float(cfg.entry_sigma), float(cfg.exit_sigma),
                int(cfg.hold_hours), bp, flat_boundaries=(TRAIN_END, VALID_END),
                funding_events=funding, entry_mask=mask,
                exit_rule=cfg.exit_rule, fee_mode="gross")
            for period, a, b in (("valid", TRAIN_END, VALID_END), ("holdout", VALID_END, END)):
                period_gross_fee = sum(float(x.get("gross_fee", 0.0)) for x in meta["ledger"]
                                       if a <= pd.Timestamp(x["time"]) < b)
                period_net_fee = sum(float(x.get("net_fee", 0.0)) for x in meta["ledger"]
                                     if a <= pd.Timestamp(x["time"]) < b)
                m = metrics(eq, trades, a, b, meta["funding_rows"])
                selected_rows.append({"selection_rank": rank, "cost_bp": bp,
                    **{k: cfg[k] for k in ("model", "frequency_min", "center_hours", "scale_days", "center_kind", "logic", "entry_sigma", "exit_sigma", "hold_hours", "exit_rule")},
                    "period": period, **m,
                    "period_gross_fees": period_gross_fee,
                    "period_net_fees_reference": period_net_fee})
            if bp == 2.0:
                stem = f"v3_selected_{rank}_{cfg.model}_{int(cfg.frequency_min)}m_2bp"
                eq.rename("equity").to_csv(RESULTS / f"{stem}_equity.csv", header=True, index_label="time")
                trades.to_csv(RESULTS / f"{stem}_trades.csv", index=False)
                (RESULTS / f"{stem}_ledger.json").write_text(
                    json.dumps(meta["ledger"], ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(selected_rows).to_csv(RESULTS / "v3_selected_costs.csv", index=False)
    selected_cfg = selected.to_dict("records")
    (RESULTS / "v3_selected_configs.json").write_text(json.dumps(selected_cfg, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    summary = {"grid_rows": len(table), "selected_configs": len(selected),
               "selected_cost_rows": len(selected_rows),
               "models": sorted(table.model.unique().tolist()),
               "frequencies_min": [5, 15, 60], "costs_bp": [0, 1, 2, 3],
               "fee_mode": "gross action turnover for primary v3 results",
               "selection": "fixed 15m controls plus five May-June leaders; July-August reused after prior v2/v3 inspection, not an untouched holdout",
               "confirmation_status": "retrospective exploratory; independent new data required",
               "train": f"{START.date()}/{TRAIN_END.date()}",
               "validation": f"{TRAIN_END.date()}/{VALID_END.date()}",
               "holdout": f"{VALID_END.date()}/{END.date()}"}
    (RESULTS / "v3_run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
