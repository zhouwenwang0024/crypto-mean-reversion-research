"""Attribute the old rank-reversal account to 2025-10-11 without deleting data.

This is an audit of an old directional reversal rule, not a statistical
arbitrage strategy. Event-excluded curves are retrospective diagnostics only.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from validate_5m_mean_reversion import (
    RESULTS, SYMBOLS, load_prices, load_funding, run_rule,
)


def replay(index, op, cl, rates, marks, start, end, *, reverse=False, cost_bp=5.0):
    """Independent inventory cash ledger; never sums the original trade PnL.

    Purchases spend cash and sales receive cash. Equity is cash plus inventory
    marked to market, unlike the source's realized futures collateral ledger.
    The old implementation's separate exit/entry executions are preserved.
    """
    lookback, hold, count = 14 * 288, 28 * 288, 2
    entries = {t for t in range(lookback + 1, len(cl) - hold, hold)
               if start <= t < end - hold}
    exits = {t + hold for t in entries}
    quantity = np.zeros(len(SYMBOLS)); cash = 1.0; previous_equity = 1.0
    records = []; trades = []; orders = []; lot = None
    for t in range(start, end):
        gap_pnl = float(quantity @ (op[t] - cl[t - 1]))
        valid = np.isfinite(rates[t]) & np.isfinite(marks[t])
        funding = float((-quantity[valid] * rates[t, valid] * marks[t, valid]).sum())
        cash += funding
        if lot is not None:
            lot['funding'] += funding
        fees = 0.0
        for side in ('exit', 'entry'):
            if side == 'exit' and (t not in exits or lot is None):
                continue
            if side == 'entry' and t not in entries:
                continue
            if side == 'entry':
                trailing = np.log(cl[t - 1] / cl[t - 1 - lookback])
                ranked = np.argsort(trailing)
                target = np.zeros(len(SYMBOLS))
                target[ranked[:count]] = 0.9 * cash / (2 * count) / cl[t - 1, ranked[:count]]
                target[ranked[-count:]] = -0.9 * cash / (2 * count) / cl[t - 1, ranked[-count:]]
                if reverse:
                    target *= -1
                lot = {'entry_time': str(index[t]), 'entry_index': t,
                       'quantity': target.copy(), 'entry_price': op[t].copy(),
                       'funding': 0.0, 'long': ','.join(np.array(SYMBOLS)[target > 0]),
                       'short': ','.join(np.array(SYMBOLS)[target < 0])}
            else:
                target = np.zeros(len(SYMBOLS))
            delta = target - quantity
            fee = float(np.abs(delta * op[t]).sum()) * cost_bp / 10_000
            cash -= float(delta @ op[t]) + fee
            fees += fee
            orders.extend({'time': str(index[t]), 'side': side, 'symbol': SYMBOLS[j],
                           'quantity_change': float(delta[j]), 'price': float(op[t, j]),
                           'fee': float(abs(delta[j] * op[t, j]) * cost_bp / 10_000)}
                          for j in np.flatnonzero(delta))
            quantity = target
            if side == 'entry':
                lot['entry_fee'] = fee
            else:
                pnl = float(lot['quantity'] @ (op[t] - lot['entry_price']))
                trades.append({k: v for k, v in lot.items() if k not in ('quantity', 'entry_price', 'entry_index')}
                              | {'exit_time': str(index[t]), 'gross_pnl': pnl,
                                 'fee': fee + lot['entry_fee'],
                                 'net_pnl_including_funding': pnl + lot['funding'] - fee - lot['entry_fee']})
                lot = None
        intrabar_pnl = float(quantity @ (cl[t] - op[t]))
        equity = cash + float(quantity @ cl[t])
        change = equity - previous_equity
        records.append((index[t], equity, change, gap_pnl, intrabar_pnl, funding, fees,
                        change - gap_pnl - intrabar_pnl - funding + fees))
        previous_equity = equity
    if lot is not None:
        raise AssertionError('The original full-sample schedule must finish flat')
    columns = ['time', 'equity', 'cash_pnl', 'gap_pnl', 'intrabar_pnl', 'funding', 'fees', 'component_error']
    return pd.DataFrame(records, columns=columns), pd.DataFrame(trades), pd.DataFrame(orders)


def drawdown(curve):
    curve = np.asarray(curve, dtype=float)
    return float(np.min(curve / np.maximum.accumulate(np.r_[1.0, curve])[1:] - 1))


def attribute(bars, trades, zone):
    a = pd.Timestamp('2025-10-11', tz=zone).tz_convert('UTC')
    b = pd.Timestamp('2025-10-12', tz=zone).tz_convert('UTC')
    mask = (bars.time >= a) & (bars.time < b)
    cash_pnl = float(bars.loc[mask, 'cash_pnl'].sum())
    original_return = float(bars.equity.iloc[-1] - 1)
    fixed_pnl_curve = 1 + bars.cash_pnl.where(~mask, 0).cumsum()
    returns = bars.equity / bars.equity.shift(1, fill_value=1.0) - 1
    zeroed_return_curve = (1 + returns.where(~mask, 0)).cumprod()
    crossing = trades[(pd.to_datetime(trades.entry_time) < b) & (pd.to_datetime(trades.exit_time) >= a)].copy()
    crossing.insert(0, 'event_timezone', zone)
    record = {'event_timezone': zone, 'event_start_utc': str(a), 'event_end_utc_exclusive': str(b),
              'event_bars': int(mask.sum()), 'original_return': original_return,
              'original_mdd': drawdown(bars.equity), 'event_cash_pnl': cash_pnl,
              'event_pnl_share_of_original_profit': cash_pnl / original_return,
              'event_price_pnl': float(bars.loc[mask, ['gap_pnl', 'intrabar_pnl']].to_numpy().sum()),
              'event_funding': float(bars.loc[mask, 'funding'].sum()),
              'event_fees': float(bars.loc[mask, 'fees'].sum()),
              'original_return_minus_event_cash_pnl': original_return - cash_pnl,
              'mdd_fixed_original_ledger_without_event_cash_pnl': drawdown(fixed_pnl_curve),
              'return_ex_event_by_zeroing_interval_returns': float(zeroed_return_curve.iloc[-1] - 1),
              'mdd_zeroing_event_interval_returns': drawdown(zeroed_return_curve),
              'crossing_trades': len(crossing),
              'crossing_trade_complete_net_pnl_including_funding': float(crossing.net_pnl_including_funding.sum())}
    return record, mask, crossing


def main():
    index, op, cl, _ = load_prices()
    rates, marks, _ = load_funding(index)
    start, end = (int(index.searchsorted(pd.Timestamp(s, tz='UTC')))
                  for s in ('2024-03-01', '2026-09-01'))
    bars, trades, orders = replay(index, op, cl, rates, marks, start, end)
    original, _, original_bars = run_rule(index, op, cl, rates, marks, (14, 28, 2), start, end)
    errors = np.abs(bars.equity.to_numpy() - original_bars.equity.to_numpy())
    checks = {'original_return': original['return'], 'independent_return': float(bars.equity.iloc[-1] - 1),
              'max_independent_equity_error': float(errors.max()),
              'first_equity_divergence_over_1e_10': None if errors.max() < 1e-10 else str(bars.time.iloc[np.flatnonzero(errors > 1e-10)[0]]),
              'max_bar_component_error': float(bars.component_error.abs().max()),
              'trade_funding_reconciliation_error': float(bars.cash_pnl.sum() - trades.net_pnl_including_funding.sum()),
              'trade_count': len(trades), 'cost_bp_one_way': 5.0}
    if errors.max() >= 1e-10 or bars.component_error.abs().max() >= 1e-10:
        raise AssertionError(checks)
    summaries = []; crossing = []; union = np.zeros(len(bars), dtype=bool)
    for zone in ('Asia/Shanghai', 'UTC'):
        summary, mask, selected = attribute(bars, trades, zone)
        summaries.append(summary); crossing.append(selected); union |= mask.to_numpy()
        bars[f'in_event_{zone.replace("/", "_")}'] = mask
    pd.DataFrame(summaries).to_csv(RESULTS / 'reversal_event_summary.csv', index=False)
    pd.concat(crossing).to_csv(RESULTS / 'reversal_event_crossing_trades.csv', index=False)
    bars.loc[union].to_csv(RESULTS / 'reversal_event_bars.csv', index=False)
    orders.to_csv(RESULTS / 'reversal_event_independent_orders.csv', index=False)
    daily = bars.set_index('time')[['cash_pnl', 'gap_pnl', 'intrabar_pnl', 'funding', 'fees']].resample('D').sum()
    daily['equity'] = bars.set_index('time').equity.resample('D').last()
    daily.to_csv(RESULTS / 'reversal_event_daily.csv')
    checks['interpretation'] = ('Retrospective attribution on the unchanged old rank-reversal ledger. '
                                'Event-excluded curves are not executable strategies or unseen validation. '
                                'Crossing trade PnL includes its entire life, both fees, and all funding.')
    (RESULTS / 'reversal_event_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps({'checks': checks, 'event_attribution': summaries}, indent=2))


if __name__ == '__main__':
    main()
