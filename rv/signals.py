"""Mean-reversion signals on a spread, and a backtest that charges for trading.

Nothing here is specific to a crack spread or to a butterfly: both are series
that are supposed to revert, so both get the same treatment and the same
scepticism.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def zscore(series, window=60, min_periods=None):
    """Rolling z-score, computed only from data available at the time.

    The mean and standard deviation use a trailing window ending at t, so the
    score at t never sees t+1. Using a full-sample mean instead, which is the
    usual shortcut, leaks the future into every single observation and is the
    single most common way to produce a mean-reversion backtest that cannot
    be traded.
    """
    rolling = series.rolling(window, min_periods=min_periods or window)
    return (series - rolling.mean()) / rolling.std()


def positions(score, entry=1.5, exit_at=0.5):
    """Fade the spread: short when it is rich, long when it is cheap.

    A position opens when the score passes `entry` and closes when it comes
    back inside `exit_at`. The band between the two stops the strategy
    flickering in and out around a single threshold, which is what turns a
    plausible signal into a transaction cost machine.

    The result is lagged one day: a score computed on today's close can only
    be traded from tomorrow.
    """
    raw = pd.Series(np.nan, index=score.index, dtype=float)
    raw[score >= entry] = -1.0
    raw[score <= -entry] = 1.0
    raw[score.abs() <= exit_at] = 0.0

    held = raw.ffill().fillna(0.0)
    return held.shift(1).fillna(0.0)


def backtest(spread, position, cost_per_trade=0.0):
    """P&L of holding `position` units of `spread`.

    `spread` is in its own units (dollars per barrel for a crack, currency for
    a DV01-weighted fly), so the P&L comes out in those units too. Costs are
    charged per unit of position changed, in the same units, which keeps the
    accounting honest for both instruments.
    """
    change = spread.diff()
    traded = position.diff().abs().fillna(position.abs())
    costs = traded * cost_per_trade

    frame = pd.DataFrame({
        'spread': spread,
        'position': position,
        'gross': position * change,
        'cost': costs,
    }).dropna(subset=['gross'])
    frame['net'] = frame['gross'] - frame['cost']
    frame['cumulative'] = frame['net'].cumsum()
    return frame


def statistics(pnl, capital=None):
    """Performance summary for a P&L series expressed in currency.

    Sharpe is computed on the P&L directly rather than on returns, because a
    spread has no natural notional to divide by: a crack spread position is
    not a percentage of anything. Reporting the ratio of mean to standard
    deviation, annualised, is the honest version of a Sharpe here, and it is
    what a relative value desk quotes.
    """
    pnl = pnl.dropna()
    if pnl.empty or pnl.std() == 0:
        return {}

    cumulative = pnl.cumsum()
    drawdown = cumulative - cumulative.cummax()

    stats = {
        'days': int(len(pnl)),
        'total': float(pnl.sum()),
        'mean_daily': float(pnl.mean()),
        'daily_vol': float(pnl.std()),
        'sharpe': float(pnl.mean() / pnl.std() * np.sqrt(TRADING_DAYS)),
        'hit_rate': float((pnl > 0).sum() / (pnl != 0).sum())
        if (pnl != 0).any() else np.nan,
        'max_drawdown': float(drawdown.min()),
        'best_day': float(pnl.max()),
        'worst_day': float(pnl.min()),
    }
    if capital:
        stats['return_on_capital'] = float(pnl.sum() / capital)
    return stats


def half_life(series):
    """Half-life of mean reversion from an Ornstein-Uhlenbeck fit.

    Regress the daily change on the lagged level. If the slope is negative the
    series pulls back towards its mean, and the half-life is -ln(2)/slope in
    days. A positive slope means it does not revert at all, and the function
    says so with a NaN rather than returning a meaningless number.
    """
    level = series.shift(1)
    change = series.diff()
    frame = pd.concat([level, change], axis=1).dropna()
    frame.columns = ['level', 'change']
    if len(frame) < 30:
        return np.nan

    x = frame['level'] - frame['level'].mean()
    y = frame['change']
    slope = float((x * y).sum() / (x * x).sum())
    if slope >= 0:
        return np.nan
    return float(-np.log(2) / slope)


def sweep(spread, entries=(1.0, 1.5, 2.0, 2.5), windows=(20, 40, 60, 120),
          cost_per_trade=0.0):
    """Sharpe across the parameter grid.

    A signal whose Sharpe only survives at one pair of settings has been
    fitted, not found. Printing the whole grid is the cheapest defence
    against believing your own backtest.
    """
    rows = []
    for window in windows:
        for entry in entries:
            score = zscore(spread, window=window)
            pos = positions(score, entry=entry, exit_at=entry / 3.0)
            frame = backtest(spread, pos, cost_per_trade)
            stats = statistics(frame['net'])
            rows.append({
                'window': window,
                'entry': entry,
                'sharpe': stats.get('sharpe', np.nan),
                'total': stats.get('total', np.nan),
                'trades': int(pos.diff().abs().sum() / 2),
            })
    return pd.DataFrame(rows)
