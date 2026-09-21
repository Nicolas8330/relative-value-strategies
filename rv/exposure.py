"""What a spread position is actually exposed to, and whether you can hedge it.

The thesis of this repository, demonstrated twice on unrelated markets: a
spread is not a hedge. Both the 3-2-1 crack and the 2s5s10s butterfly are
constructed to be neutral to something, and both carry a large unintended
exposure to exactly that thing.

    3-2-1 crack     beta to crude            -0.35
    2s5s10s fly     dollar loading on slope  -305 per 1153 of curvature

Whether that exposure can be hedged away turns out to depend on something the
textbooks rarely mention: the stability of the hedge ratio. The butterfly's
factor loadings barely move, so solving for them works. The crack's beta to
crude swings from -0.99 to +0.27, so yesterday's hedge ratio is a poor guide
to today's, and hedging removes only a third of the exposure.

A hedge ratio is only as good as its own standard deviation. That is the part
worth taking away.
"""

import numpy as np
import pandas as pd


def rolling_beta(target, factor, window=250, lag=True):
    """Hedge ratio of `target` on `factor`, estimated on daily changes.

    Changes, not levels: a position takes risk in the space it moves in. Two
    series can share a long-run relationship in levels and still have a large,
    unstable short-run beta, which is exactly the crack spread's problem.

    With `lag=True` the ratio is shifted one day, so the hedge in force on any
    date was computable the evening before.
    """
    dt = target.diff()
    df = factor.diff()
    beta = dt.rolling(window).cov(df) / df.rolling(window).var()
    return beta.shift(1) if lag else beta


def hedge(target, factor, window=250):
    """Daily change of the beta-hedged spread, and the ratio used."""
    beta = rolling_beta(target, factor, window)
    residual = (target.diff() - beta * factor.diff()).dropna()
    residual.name = 'hedged_change'
    return residual, beta.reindex(residual.index)


def exposure_report(target, factor, window=250, label='spread'):
    """Before and after a rolling beta hedge, plus the stability of the ratio.

    `hedge_quality` is the fraction of the original exposure removed. A number
    near one means the hedge works; the crack spread scores about a third,
    and the reason is in `beta_std`.
    """
    changes_t, changes_f = target.diff(), factor.diff()
    hedged, beta = hedge(target, factor, window)

    raw_beta = _ols_slope(changes_t, changes_f)
    net_beta = _ols_slope(hedged, changes_f.reindex(hedged.index))

    return {
        'label': label,
        'beta_raw': raw_beta,
        'beta_hedged': net_beta,
        'correlation_raw': float(changes_t.corr(changes_f)),
        'correlation_hedged': float(hedged.corr(changes_f.reindex(hedged.index))),
        'hedge_quality': float(1.0 - abs(net_beta) / abs(raw_beta))
        if raw_beta else np.nan,
        'beta_mean': float(beta.mean()),
        'beta_std': float(beta.std()),
        'beta_min': float(beta.min()),
        'beta_max': float(beta.max()),
        'vol_ratio': float(hedged.std() / changes_t.reindex(hedged.index).std()),
    }


def _ols_slope(y, x):
    frame = pd.concat([y, x], axis=1).dropna()
    if len(frame) < 30:
        return np.nan
    frame.columns = ['y', 'x']
    centred = frame['x'] - frame['x'].mean()
    return float((centred * frame['y']).sum() / (centred * centred).sum())


def stability(ratios, window=250):
    """How much a hedge ratio moves, which decides whether hedging is possible.

    A ratio whose rolling standard deviation is large relative to its own mean
    cannot be hedged with a lagged estimate, however careful the estimation.
    `signal_to_noise` below one means the ratio is mostly noise.
    """
    clean = pd.Series(ratios).dropna()
    if clean.empty:
        return {}
    return {
        'mean': float(clean.mean()),
        'std': float(clean.std()),
        'signal_to_noise': float(abs(clean.mean()) / clean.std())
        if clean.std() else np.inf,
        'min': float(clean.min()),
        'max': float(clean.max()),
        'range': float(clean.max() - clean.min()),
        'autocorrelation_1m': float(clean.autocorr(lag=21)),
    }
