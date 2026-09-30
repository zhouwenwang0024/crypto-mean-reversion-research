"""Independent inventory/cash audit of the fixed long-horizon Johansen candidate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_statarb_backtest as bt

CONFIG = dict(fee_bp=5.0, use_funding=True, hold_bars=56 * 288,
              entry_z=3.0, exit_z=0.5, rearm_z=1.0, force_month_boundary=False)
PREFIX = bt.RESULTS / "statarb_candidate"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(index, op, close, rates, marks, orders, start, end):
    """Spend cash on buys, receive cash on sells, and mark cash + inventory.

    No engine PnL, equity, funding or fee totals enter this calculation.
    The only execution input is signed quantity, timestamp and fill kind;
    fill prices, fees and funding are independently reconstructed from data.
    """
    grouped = {t: g.to_dict("records") for t, g in
               orders.assign(slot=pd.to_datetime(orders.time, utc=True)).groupby("slot", sort=False)}
    ncoin = close.shape[1]
    q = np.zeros(ncoin); cash = 1.0; previous = 1.0
    coin_pnl = np.zeros(ncoin); coin_fee = np.zeros(ncoin); coin_fund = np.zeros(ncoin)
    lots = {}; attribution = {}; rows = []
    fee_error = fill_error = entry_budget_error = 0.0
    for t in range(start, end):
        valid = np.isfinite(rates[t]) & np.isfinite(marks[t])
        fund = np.zeros(ncoin)
        fund[valid] = -q[valid] * rates[t, valid] * marks[t, valid]
        cash += float(fund.sum()); coin_fund += fund
        for pid, lot in lots.items():
            flow = float((-lot[valid] * rates[t, valid] * marks[t, valid]).sum())
            attribution[pid]["funding"] += flow
        pnl = q * (close[t] - close[t - 1]) + fund
        fees = np.zeros(ncoin)
        current_orders = grouped.get(index[t], [])
        for pid in dict.fromkeys(r["position_id"] for r in current_orders):
            part = [r for r in current_orders if r["position_id"] == pid]
            kind = part[0]["kind"]
            if kind == "entry":
                equity_at_open = cash + float(q @ op[t])
                entry_notional = sum(abs(float(r["quantity_change"]) * op[t, int(r["symbol_index"])]) for r in part)
                entry_budget_error = max(entry_budget_error, entry_notional / equity_at_open - .30)
                if any(abs(q[int(r["symbol_index"])]) > 1e-12 for r in part):
                    raise AssertionError("Overlapping coin in independent inventory")
                lots[pid] = np.zeros(ncoin)
                attribution[pid] = dict(position_id=pid, pair="/".join(sorted(r["symbol"] for r in part)),
                                         price_pnl=0.0, funding=0.0, fees=0.0, net_pnl=0.0)
            for row in part:
                j, dq = int(row["symbol_index"]), float(row["quantity_change"])
                px = float(close[t, j] if kind == "forced_exit" else op[t, j])
                fee = abs(dq * px) * CONFIG["fee_bp"] / 10_000
                fee_error = max(fee_error, abs(fee - row["fee"]))
                fill_error = max(fill_error, abs(px - row["price"]))
                cash -= dq * px + fee
                q[j] += dq; lots[pid][j] += dq
                pnl[j] += dq * (close[t, j] - px) - fee
                fees[j] += fee
                attribution[pid]["price_pnl"] -= dq * px
                attribution[pid]["fees"] += fee
            if kind != "entry":
                if np.max(np.abs(lots.pop(pid))) > 1e-12:
                    raise AssertionError("Pair exit did not flatten independent quantities")
        coin_pnl += pnl; coin_fee += fees
        equity = cash + float(q @ close[t])
        gross = float(np.abs(q * close[t]).sum()); net = float(q @ close[t])
        rows.append((index[t], equity, equity - previous, float(pnl.sum()), float(fund.sum()),
                     float(fees.sum()), gross, net, len(lots), np.count_nonzero(np.abs(q) > 1e-12)))
        previous = equity
    if lots or np.max(np.abs(q)) > 1e-12:
        raise AssertionError("Terminal independent inventory is not flat")
    bars = pd.DataFrame(rows, columns=["time", "equity", "equity_change", "net_pnl", "funding", "fees",
                                      "gross_exposure", "net_exposure", "open_pairs", "open_coins"])
    bars["gross_equity_ratio"] = bars.gross_exposure / bars.equity
    bars["net_equity_ratio"] = bars.net_exposure / bars.equity
    coins = pd.DataFrame(dict(symbol=bt.SYMBOLS, net_pnl=coin_pnl,
                             price_pnl=coin_pnl - coin_fund + coin_fee, fees=coin_fee, funding=coin_fund))
    positions = pd.DataFrame(attribution.values())
    positions["net_pnl"] = positions.price_pnl + positions.funding - positions.fees
    return bars, coins, positions, dict(max_order_fee_error=fee_error, max_order_fill_error=fill_error,
                                       max_entry_budget_excess=entry_budget_error,
                                       terminal_max_abs_quantity=float(np.max(np.abs(q))))


def event_attribution(bars, trades, zone):
    a = pd.Timestamp("2025-10-11", tz=zone).tz_convert("UTC"); b = a + pd.Timedelta(days=1)
    mask = (bars.time >= a) & (bars.time < b)
    changes = bars.equity / bars.equity.shift(1, fill_value=1.0) - 1.0
    event_net = float(bars.loc[mask, "equity_change"].sum())
    fee_flow = bars["fees"].diff().fillna(bars["fees"])
    fees = float(fee_flow.loc[mask].sum()); fund = float(bars.loc[mask, "funding"].sum())
    crossing = (pd.to_datetime(trades.entry_time, utc=True) < b) & (pd.to_datetime(trades.exit_time, utc=True) >= a)
    return dict(timezone=zone, event_start_utc=str(a), event_end_utc_exclusive=str(b),
                bars=int(mask.sum()), event_net_pnl=event_net, event_price_pnl=event_net + fees - fund,
                event_fees=fees, event_funding=fund, crossing_trades=int(crossing.sum()),
                full_return=float(bars.equity.iloc[-1] - 1),
                fixed_original_ledger_return_minus_event=float(bars.equity.iloc[-1] - 1 - event_net),
                return_zeroing_event_interval_returns=float((1 + changes.where(~mask, 0)).prod() - 1))


def main():
    model_path = bt.RESULTS / "statarb_formation_selected.csv"
    provenance = dict(runner_sha256=sha(bt.__file__), formation_sha256=sha(model_path))
    index, op, close, _ = bt.load_prices(); rates, marks, _ = bt.load_funding(index)
    selected = pd.read_csv(model_path)
    start, end = (int(index.searchsorted(pd.Timestamp(t, tz="UTC"))) for t in ("2024-06-01", "2026-09-01"))
    actual, trades, orders, summary = bt.run_model(index, op, close, rates, marks, selected,
                                                   "johansen", start, end, **CONFIG)
    bars, coins, positions, checks = replay(index, op, close, rates, marks, orders, start, end)
    error = bars.equity.to_numpy() - actual.equity.to_numpy()
    order_view = orders.copy()
    order_view["abs_notional"] = (order_view.quantity_change * order_view.price).abs()
    same = order_view.groupby(["time", "symbol"], as_index=False).agg(
        abs_notional=("abs_notional", "sum"), signed_notional=("quantity_change", lambda x: 0.0),
    )
    signed = order_view.assign(signed_notional=order_view.quantity_change * order_view.price)
    signed = signed.groupby(["time", "symbol"], as_index=False).signed_notional.sum()
    same = same.drop(columns="signed_notional").merge(signed, on=["time", "symbol"])
    same["cancelled_abs_notional"] = same.abs_notional - same.signed_notional.abs()
    forced = (order_view.assign(has_forced_exit=order_view.kind.eq("forced_exit"))
              .groupby(["time", "symbol"], as_index=False).has_forced_exit.any())
    same = same.merge(forced, on=["time", "symbol"], how="left")
    position_check = positions.merge(trades[["position_id", "net_pnl", "funding", "fee"]], on="position_id", suffixes=("_independent", "_engine"))
    checks.update(provenance | dict(
        config=CONFIG, bars_checked=len(bars), order_rows=len(orders),
        max_bar_equity_error=float(np.abs(error).max()),
        first_equity_divergence=None if np.abs(error).max() < 1e-10 else str(bars.time.iloc[np.flatnonzero(np.abs(error) >= 1e-10)[0]]),
        max_bar_pnl_identity_error=float((bars.net_pnl - bars.equity_change).abs().max()),
        max_bar_funding_error=float(np.max(np.abs(bars.funding.to_numpy() - actual.funding.to_numpy()))),
        max_bar_gross_error=float(np.max(np.abs(bars.gross_exposure.to_numpy() - actual.gross_exposure.to_numpy()))),
        max_trade_net_error=float((position_check.net_pnl_independent - position_check.net_pnl_engine).abs().max()),
        max_trade_funding_error=float((position_check.funding_independent - position_check.funding_engine).abs().max()),
        independent_fee_total_error=float(abs(bars.fees.sum() - summary["fees"])),
        max_same_timestamp_symbol_cancellation=float(same.cancelled_abs_notional.max()) if len(same) else 0.0,
        same_timestamp_symbol_groups=int((same.cancelled_abs_notional > 1e-12).sum()),
        same_timestamp_forced_exit_groups=int(same.has_forced_exit.sum()) if len(same) else 0,
        independent_coin_net_error=float(abs(coins.net_pnl.sum() - summary["return"])),
        independent_position_net_error=float(abs(positions.net_pnl.sum() - summary["return"])),
        source_unchanged_during_audit=provenance == dict(runner_sha256=sha(bt.__file__), formation_sha256=sha(model_path)),
        method="Signed inventory cash flows; fees and funding reconstructed independently from original data; every 5m equity compared",
        event_exclusion="Retrospective attribution only; original prices, quantities and compounding path retained",
    ))
    error_keys = [k for k in checks if k.endswith("error") or k.endswith("excess")]
    checks["all_pass"] = bool(all(checks[k] < 1e-10 for k in error_keys) and checks["source_unchanged_during_audit"])
    if not checks["all_pass"]:
        raise AssertionError(checks)
    bars["fee_flow"] = bars.fees.diff().fillna(bars.fees)
    daily = bars.set_index("time").resample("D").agg(
        equity=("equity", "last"), net_pnl=("equity_change", "sum"), fees=("fee_flow", "sum"), funding=("funding", "sum"),
        mean_open_pairs=("open_pairs", "mean"), max_open_pairs=("open_pairs", "max"),
        mean_open_coins=("open_coins", "mean"), max_open_coins=("open_coins", "max"),
        mean_gross_equity_ratio=("gross_equity_ratio", "mean"), max_gross_equity_ratio=("gross_equity_ratio", "max"))
    pairs = positions.groupby("pair", as_index=False).agg(trades=("position_id", "size"), price_pnl=("price_pnl", "sum"),
                                                         fees=("fees", "sum"), funding=("funding", "sum"), net_pnl=("net_pnl", "sum"))
    positive_days = daily.net_pnl.clip(lower=0); positive_pairs = pairs.net_pnl.clip(lower=0)
    summary.update(dict(max_gross_exposure_over_equity=float(bars.gross_equity_ratio.max()),
                        mean_gross_exposure_over_equity=float(bars.gross_equity_ratio.mean()),
                        max_abs_net_exposure_over_equity=float(bars.net_equity_ratio.abs().max()),
                        mean_open_pairs=float(bars.open_pairs.mean()), max_open_pairs=int(bars.open_pairs.max()),
                        fraction_bars_invested=float(bars.open_pairs.gt(0).mean()),
                        mean_hold_days=float(trades.hold_bars.mean() / 288), median_hold_days=float(trades.hold_bars.median() / 288),
                        max_hold_days=float(trades.hold_bars.max() / 288),
                        positive_days=int(daily.net_pnl.gt(0).sum()), negative_days=int(daily.net_pnl.lt(0).sum()),
                        largest_daily_pnl=float(daily.net_pnl.max()), worst_daily_pnl=float(daily.net_pnl.min()),
                        top5_positive_day_share=float(positive_days.nlargest(5).sum() / positive_days.sum()),
                        top5_days_pnl_over_total_net=float(daily.net_pnl.nlargest(5).sum() / summary["return"]),
                        top_pair_positive_pnl_share=float(positive_pairs.max() / positive_pairs.sum()),
                        largest_pair_net_pnl=float(pairs.net_pnl.max())))
    outputs = {"orders": orders, "trades": trades, "daily": daily.reset_index(), "coins": coins,
               "positions": positions, "pairs": pairs, "summary": pd.DataFrame([summary]),
               "events": pd.DataFrame([event_attribution(bars, trades, z) for z in ("UTC", "Asia/Shanghai")])}
    for name, frame in outputs.items():
        frame.to_csv(str(PREFIX) + "_" + name + ".csv", index=False)
    Path(str(PREFIX) + "_checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(summary=summary, checks=checks), ensure_ascii=False, indent=2))
    print(outputs["events"].to_string(index=False)); print(pairs.sort_values("net_pnl", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
