"""Fixed diagnostic contrasts after seeing the long-horizon candidate.

These are exploratory stress tests, not untouched validation or a new search.
"""
import pandas as pd

import run_statarb_backtest as bt


def main():
    index, op, close, _ = bt.load_prices()
    rates, marks, _ = bt.load_funding(index)
    selected = pd.read_csv(bt.RESULTS / 'statarb_formation_selected.csv')
    start, end = (int(index.searchsorted(bt._ts(x))) for x in ('2024-06-01', '2026-09-01'))
    variants = {
        'new_orders_cap_90pct': {'gross_limit': 0.90},
        'extra_delay_5min': {'delay_bars': 1},
        'extra_delay_10min': {'delay_bars': 2},
        'hourly_observation': {'observe_bars': 12},
        'hourly_shift_5min': {'observe_bars': 12, 'phase_bars': 1},
        'hourly_shift_30min': {'observe_bars': 12, 'phase_bars': 6},
    }
    rows = []
    for name, change in variants.items():
        bars, trades, orders, summary = bt.run_model(
            index, op, close, rates, marks, selected, 'johansen', start, end,
            hold_bars=56 * 288, entry_z=3.0, exit_z=0.5,
            force_month_boundary=False, **change)
        summary.update(variant=name, max_gross_over_equity=float((bars.gross_exposure / bars.equity).max()))
        for zone in ('UTC', 'Asia/Shanghai'):
            result = bt.attribute(bars, trades, zone, 'johansen')
            summary['return_ex_event_' + zone.replace('/', '_')] = result['event_return_zeroed']
        rows.append(summary)
        pd.DataFrame(rows).to_csv(bt.RESULTS / 'statarb_execution_stress.csv', index=False)
        print(name, summary['return'], flush=True)


if __name__ == '__main__':
    main()
