"""When the strategy works, and when it quietly stops working.

A single Sharpe ratio over a decade hides the only thing a risk taker needs to
know: which states of the world the edge lives in. Mean reversion on a spread
is a bet that nothing structural has changed, and it is precisely in the
regimes where something has changed that it loses the most.

Regimes here are defined from information available at the time, never from
the full sample. Splitting on a full-sample median is a subtle but complete
form of look-ahead: it uses the end of the history to decide what counted as
"high volatility" at the beginning.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def tercile_regime(driver, window=500, labels=('low', 'mid', 'high')):
    """Label each day by where the driver sits in its own trailing distribution.

    Thresholds come from a rolling window ending the day before, so the label
    on any date could have been assigned on that date.
    """
    rank = driver.rolling(window).apply(
        lambda block: (block[-1] > block[:-1]).mean(), raw=True
    ).shift(1)

    regime = pd.Series(index=driver.index, dtype=object)
    regime[rank <= 1 / 3] = labels[0]
    regime[(rank > 1 / 3) & (rank <= 2 / 3)] = labels[1]
    regime[rank > 2 / 3] = labels[2]
    return regime


def binary_regime(driver, threshold=0.0, labels=('below', 'above')):
    """Split on a fixed, economically meaningful level.

    Use this when the threshold means something on its own, such as a curve
    being inverted or a futures curve being in backwardation. No estimation is
    involved, so there is nothing to leak.
    """
    return pd.Series(np.where(driver.shift(1) > threshold, labels[1], labels[0]),
                     index=driver.index)


def performance_by_regime(pnl, regime, min_days=60):
    """Annualised statistics of a P&L series within each regime.

    Regimes with fewer than `min_days` observations are reported but flagged,
    because a Sharpe computed on thirty days is a number, not evidence.
    """
    frame = pd.concat([pnl.rename('pnl'), regime.rename('regime')],
                      axis=1).dropna()
    rows = []
    for name, block in frame.groupby('regime'):
        series = block['pnl']
        if series.std() == 0:
            continue
        rows.append({
            'regime': name,
            'days': int(len(series)),
            'share': float(len(series) / len(frame)),
            'total': float(series.sum()),
            'mean_daily': float(series.mean()),
            'sharpe': float(series.mean() / series.std() * np.sqrt(TRADING_DAYS)),
            'hit_rate': float((series > 0).sum() / (series != 0).sum())
            if (series != 0).any() else np.nan,
            'worst_day': float(series.min()),
            'reliable': bool(len(series) >= min_days),
        })
    return pd.DataFrame(rows).set_index('regime').sort_values('sharpe',
                                                              ascending=False)


def drawdown_by_regime(pnl, regime):
    """Worst peak-to-trough within each regime, on the cumulative P&L."""
    frame = pd.concat([pnl.rename('pnl'), regime.rename('regime')],
                      axis=1).dropna()
    rows = {}
    for name, block in frame.groupby('regime'):
        cumulative = block['pnl'].cumsum()
        rows[name] = float((cumulative - cumulative.cummax()).min())
    return pd.Series(rows, name='max_drawdown')


def spread_of_sharpes(table):
    """Gap between the best and worst regime.

    A strategy whose Sharpe is roughly the same everywhere is robust. One
    whose Sharpe ranges from two to minus one is a bet on a regime, and should
    be described as one.
    """
    if table.empty:
        return np.nan
    reliable = table[table['reliable']] if 'reliable' in table else table
    if reliable.empty:
        return np.nan
    return float(reliable['sharpe'].max() - reliable['sharpe'].min())
