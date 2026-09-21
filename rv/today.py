"""The position today: what the model says, how big, and when to give up.

Research that stops at a Sharpe ratio is unfinished. A desk needs the four
numbers that turn a signal into a trade: where fair value is, how far the
market is from it, how long convergence should take, and at what point the
model was simply wrong.

The last one matters most and is the one usually missing. A mean-reversion
signal has no natural stop, because by construction it gets more attractive as
it loses money. The invalidation level here is therefore set on the residual,
not on the P&L: beyond it, the deviation is too large to be explained by the
model's own historical errors, which means the model has broken rather than
the market being wrong.
"""

import numpy as np
import pandas as pd


def convergence_horizon(half_life, fraction=0.75):
    """Days for `fraction` of a deviation to decay at a given half-life."""
    if not np.isfinite(half_life) or half_life <= 0:
        return np.nan
    return float(half_life * np.log(1 - fraction) / np.log(0.5))


def projection_cone(residual, half_life, horizons=(21, 63, 126),
                    quantiles=(0.1, 0.5, 0.9)):
    """Where the residual should be in a month, a quarter, six months.

    Under an Ornstein-Uhlenbeck process the expected level decays towards zero
    at the half-life, while the uncertainty around it grows and then flattens
    out at the unconditional standard deviation. Both effects are included;
    showing only the decay would draw a cone that narrows to a point and
    promise a precision that does not exist.
    """
    residual = residual.dropna()
    if residual.empty or not np.isfinite(half_life) or half_life <= 0:
        return pd.DataFrame()

    current = float(residual.iloc[-1])
    sigma = float(residual.std())
    kappa = np.log(2) / half_life

    rows = []
    for horizon in horizons:
        decay = np.exp(-kappa * horizon)
        expected = current * decay
        # Stationary OU: variance of the forecast grows to sigma^2.
        spread = sigma * np.sqrt(max(1.0 - decay ** 2, 1e-12))
        row = {'horizon_days': horizon, 'expected': expected}
        for q in quantiles:
            from math import erf, sqrt
            z = _norm_ppf(q)
            row[f'q{int(q * 100)}'] = expected + z * spread
        rows.append(row)
    return pd.DataFrame(rows).set_index('horizon_days')


def _norm_ppf(p):
    """Inverse normal CDF, Acklam's rational approximation."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    lo, hi = 0.02425, 1 - 0.02425
    if p < lo:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > hi:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def size_to_volatility(spread, target_daily_vol, window=60):
    """Units of the spread to hold for a chosen daily P&L volatility.

    Sizing on realised volatility rather than on conviction is what keeps a
    relative value book from being dominated by whichever trade happened to
    become volatile.
    """
    realised = spread.diff().rolling(window).std()
    if realised.empty or not np.isfinite(realised.iloc[-1]) or realised.iloc[-1] == 0:
        return np.nan
    return float(target_daily_vol / realised.iloc[-1])


def call(residual, half_life, entry=1.5, exit_at=0.5, invalidate=3.0,
         units=None, unit_label='unit'):
    """The whole position summary, as a desk would write it down."""
    residual = residual.dropna()
    if residual.empty:
        return {}

    current = float(residual.iloc[-1])
    sigma = float(residual.std())
    z = current / sigma if sigma else np.nan
    percentile = float((residual < current).mean())

    if z >= entry:
        direction, rationale = 'short the spread', 'rich to model'
    elif z <= -entry:
        direction, rationale = 'long the spread', 'cheap to model'
    elif abs(z) <= exit_at:
        direction, rationale = 'flat', 'at fair value'
    else:
        direction, rationale = 'no new position', 'inside the entry band'

    return {
        'as_of': str(residual.index[-1].date()),
        'residual': current,
        'z_score': z,
        'percentile': percentile,
        'direction': direction,
        'rationale': rationale,
        'half_life_days': float(half_life) if np.isfinite(half_life) else np.nan,
        'days_to_75pct_convergence': convergence_horizon(half_life, 0.75),
        'invalidation_residual': float(np.sign(current) * invalidate * sigma)
        if sigma else np.nan,
        'units': units,
        'unit_label': unit_label,
    }
