"""The 2s5s10s butterfly: a trade on the curvature of the yield curve.

The fly in yield space is

    F = 2 * y5 - y2 - y10

Positive means the five year sits above the line joining the two and the ten,
so the belly is cheap relative to the wings.

The part that matters is the weighting, and the received wisdom about it turns
out to be wrong on the data.

The usual argument goes: a 1-2-1 fly traded in equal notionals is not duration
neutral, because a basis point on a two year is worth far less in cash than a
basis point on a ten year, so weight the legs by DV01 instead. That is true as
far as it goes, and it does not go far enough. DV01 neutrality protects the
trade against a *parallel* shift of the curve. The curve does not move in
parallel. Its empirical level factor loads unequally across tenors, so a
DV01-neutral fly still carries level risk, and both classic weightings carry a
large slope exposure that nobody mentions.

Measured on US Treasury yields since 2015, dollar loading of each scheme on
each component for a $1m body:

    weighting            level     slope   curvature
    equal notional       +0.02   -304.68    +1152.65
    50/50 DV01          +28.56    +53.75     +535.56
    PCA-neutral          +0.30     -5.78     +563.21

The equal-notional fly turns out to be almost level neutral by accident, and
carries a slope exposure a quarter the size of the curvature it is meant to
trade. The DV01-weighted fly fixes neither. Only solving the wing weights
against the estimated factor loadings isolates curvature, and even then not
perfectly: those weights are fitted on a trailing window, so they neutralise
out of sample rather than by construction. Cutting the slope exposure by a
factor of fifty is what a hedge can actually deliver, and claiming zero would
mean having fitted the hedge on the data it is measured against.
"""

import numpy as np
import pandas as pd


def modified_duration(yield_pct, maturity_years, frequency=2):
    """Modified duration of a par bond, the standard desk approximation.

    For a bond priced at par with `frequency` coupons a year, duration
    collapses to a closed form:

        D_mod = (1 / y) * [1 - (1 + y/f)^(-f*n)]

    Using a par bond is the usual simplification for a benchmark Treasury,
    which by construction trades close to par when it is issued.
    """
    index = yield_pct.index if isinstance(yield_pct, pd.Series) else None
    y = np.asarray(yield_pct, dtype=float) / 100.0
    n = maturity_years

    # At zero yield the expression is indeterminate; the limit is the maturity.
    safe = np.where(np.abs(y) < 1e-9, 1.0, y)
    with np.errstate(divide='ignore', invalid='ignore'):
        duration = np.where(
            np.abs(y) < 1e-9,
            float(n),
            (1.0 / safe) * (1.0 - (1.0 + y / frequency) ** (-frequency * n)),
        )

    # np.where drops the index, and every caller downstream needs it back.
    return pd.Series(duration, index=index) if index is not None else duration


def dv01(yield_pct, maturity_years, face=100.0, frequency=2):
    """Cash value of one basis point, per `face` of notional.

    DV01 = D_mod * price * 1bp, and for a par bond price = face.
    """
    return modified_duration(yield_pct, maturity_years, frequency) * face * 1e-4


def fly_yield(y2, y5, y10):
    """Butterfly in yield space, in basis points."""
    fly = (2.0 * y5 - y2 - y10) * 100.0
    fly.name = 'fly_2s5s10s_bp'
    return fly


def dv01_weights(y2, y5, y10, body_notional=1_000_000.0):
    """Notionals for a DV01-neutral, 50/50 weighted butterfly.

    The body's DV01 is split equally between the two wings, which is the
    market convention. Each wing's notional is then whatever makes its DV01
    equal to half the body's.

    Returns notionals in the same currency unit as `body_notional`, positive
    meaning long that leg. The trade here is long the belly, short the wings.
    """
    d2 = dv01(y2, 2, face=1.0)
    d5 = dv01(y5, 5, face=1.0)
    d10 = dv01(y10, 10, face=1.0)

    body_dv01 = body_notional * d5
    frame = pd.DataFrame({
        'notional_2y': -0.5 * body_dv01 / d2,
        'notional_5y': float(body_notional),
        'notional_10y': -0.5 * body_dv01 / d10,
        'dv01_2y': d2, 'dv01_5y': d5, 'dv01_10y': d10,
    })
    frame['net_dv01'] = (frame['notional_2y'] * frame['dv01_2y']
                         + frame['notional_5y'] * frame['dv01_5y']
                         + frame['notional_10y'] * frame['dv01_10y'])
    return frame


def pnl_dv01_weighted(y2, y5, y10, body_notional=1_000_000.0):
    """Daily P&L of the DV01-neutral fly, in currency.

    A leg gains when its yield falls, hence the minus sign on the yield
    change. Weights are those of the previous close, so the position is not
    rebalanced using the same day's move.
    """
    weights = dv01_weights(y2, y5, y10, body_notional).shift(1)
    changes = pd.DataFrame({
        'd2': y2.diff() * 100.0,     # basis points
        'd5': y5.diff() * 100.0,
        'd10': y10.diff() * 100.0,
    })
    pnl = -(
        weights['notional_2y'] * weights['dv01_2y'] * changes['d2']
        + weights['notional_5y'] * weights['dv01_5y'] * changes['d5']
        + weights['notional_10y'] * weights['dv01_10y'] * changes['d10']
    )
    pnl.name = 'pnl_dv01_weighted'
    return pnl.dropna()


def pnl_equal_notional(y2, y5, y10, body_notional=1_000_000.0):
    """Daily P&L of a naive 1-2-1 fly traded in equal notionals.

    This is the version that looks right on a yield chart and carries an
    unintended duration position. Comparing its P&L with the DV01-weighted
    one is the whole point of this module.
    """
    d2 = dv01(y2, 2, face=1.0).shift(1)
    d5 = dv01(y5, 5, face=1.0).shift(1)
    d10 = dv01(y10, 10, face=1.0).shift(1)

    pnl = -(
        -body_notional * d2 * y2.diff() * 100.0
        + 2.0 * body_notional * d5 * y5.diff() * 100.0
        - body_notional * d10 * y10.diff() * 100.0
    )
    pnl.name = 'pnl_equal_notional'
    return pnl.dropna()


def residual_duration(y2, y5, y10, body_notional=1_000_000.0):
    """Net DV01 carried by the naive 1-2-1 fly.

    Zero would mean the trade is neutral to a parallel shift. It is not,
    though the residual is small: the wings' combined DV01 does not quite
    match twice the belly's. The larger exposure this construction carries is
    to the slope, not the level, which `factor_exposure` shows.
    """
    d2 = dv01(y2, 2, face=1.0)
    d5 = dv01(y5, 5, face=1.0)
    d10 = dv01(y10, 10, face=1.0)
    net = body_notional * (-d2 + 2.0 * d5 - d10)
    net.name = 'residual_dv01'
    return net


def curve_pca(y2, y5, y10, window=None):
    """Principal components of daily yield changes across the three tenors.

    The textbook result is that the first three components of a yield curve
    are level, slope and curvature, in that order, and that they explain
    almost all of the variance. The butterfly is a curvature trade, so it
    should load on the third component and be close to orthogonal to the
    first. This function exists to check that claim on the data rather than
    assert it.
    """
    changes = pd.concat([y2.diff(), y5.diff(), y10.diff()], axis=1).dropna()
    changes.columns = ['2y', '5y', '10y']
    if window:
        changes = changes.tail(window)

    matrix = changes.values
    matrix = matrix - matrix.mean(axis=0)
    covariance = np.cov(matrix, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)

    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]

    # Sign is arbitrary in an eigendecomposition. Fix it so that the first
    # component is a parallel rise and the third has a positive belly, which
    # is how level and curvature are conventionally drawn.
    if vectors[:, 0].sum() < 0:
        vectors[:, 0] *= -1
    if vectors[1, 2] < 0:
        vectors[:, 2] *= -1

    scores = pd.DataFrame(matrix @ vectors, index=changes.index,
                          columns=['level', 'slope', 'curvature'])
    return {
        'explained': (values / values.sum()).tolist(),
        'loadings': pd.DataFrame(vectors, index=['2y', '5y', '10y'],
                                 columns=['level', 'slope', 'curvature']),
        'scores': scores,
    }


def pca_neutral_weights(y2, y5, y10, neutralise=('level', 'slope'),
                        lookback=500):
    """Wing weights, in units of body DV01, that zero the chosen factors.

    Solves for w2 and w10 such that the position's loading on each named
    principal component is zero, with the body fixed at one unit of DV01.
    With two constraints and two unknowns the system is exactly determined.

    Loadings are estimated on a trailing window ending the day before, never
    on the full sample, because a hedge ratio fitted on data that includes the
    future is not a hedge ratio. Rows before the window has filled are NaN.

    Returns a frame of w2 and w10; both are normally negative, since the
    wings are sold against the belly.
    """
    changes = pd.concat([y2.diff(), y5.diff(), y10.diff()], axis=1).dropna()
    changes.columns = ['2y', '5y', '10y']

    out = pd.DataFrame(index=changes.index, columns=['w2', 'w10'], dtype=float)
    for i in range(lookback, len(changes)):
        window = changes.iloc[i - lookback:i]        # ends at i-1, excludes i
        loadings = _loadings(window)
        A = np.array([[loadings[f]['2y'], loadings[f]['10y']] for f in neutralise])
        b = -np.array([loadings[f]['5y'] for f in neutralise])
        try:
            out.iloc[i] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
    return out


def _loadings(changes):
    """Principal component loadings of a block of yield changes."""
    matrix = changes.values - changes.values.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(matrix, rowvar=False))
    order = np.argsort(values)[::-1]
    vectors = vectors[:, order]
    if vectors[:, 0].sum() < 0:
        vectors[:, 0] *= -1
    if vectors[1, 2] < 0:
        vectors[:, 2] *= -1
    return pd.DataFrame(vectors, index=['2y', '5y', '10y'],
                        columns=['level', 'slope', 'curvature'])


def pnl_pca_neutral(y2, y5, y10, body_notional=1_000_000.0, lookback=500):
    """Daily P&L of the factor-neutral fly, in currency.

    Weights come from the previous close, so the hedge in force on any day was
    computable the evening before.
    """
    weights = pca_neutral_weights(y2, y5, y10, lookback=lookback).shift(1)
    body_dv01 = body_notional * dv01(y5, 5, face=1.0).shift(1)

    changes = pd.DataFrame({
        'd2': y2.diff() * 100.0,
        'd5': y5.diff() * 100.0,
        'd10': y10.diff() * 100.0,
    })
    pnl = -body_dv01 * (
        weights['w2'] * changes['d2']
        + changes['d5']
        + weights['w10'] * changes['d10']
    )
    pnl.name = 'pnl_pca_neutral'
    return pnl.dropna()


def factor_exposure(y2, y5, y10, body_notional=1_000_000.0):
    """Loading of each weighting scheme on level, slope and curvature.

    This is the table that settles the weighting argument. Each scheme is
    expressed as a vector of dollar DV01 per leg, then projected onto the
    principal components estimated over the whole sample. A clean curvature
    trade has zero in the first two columns.
    """
    loadings = curve_pca(y2, y5, y10)['loadings']
    d2 = float(dv01(y2.iloc[-1], 2, face=1.0))
    d5 = float(dv01(y5.iloc[-1], 5, face=1.0))
    d10 = float(dv01(y10.iloc[-1], 10, face=1.0))
    n = body_notional

    schemes = {
        'equal notional': np.array([-n * d2, 2 * n * d5, -n * d10]),
        '50/50 DV01': np.array([-0.5 * n * d5, n * d5, -0.5 * n * d5]),
    }
    weights = pca_neutral_weights(y2, y5, y10).dropna()
    if not weights.empty:
        last = weights.iloc[-1]
        schemes['PCA-neutral'] = np.array(
            [last['w2'] * n * d5, n * d5, last['w10'] * n * d5])

    rows = {}
    for name, vector in schemes.items():
        rows[name] = {factor: float(vector @ loadings[factor])
                      for factor in ('level', 'slope', 'curvature')}
    return pd.DataFrame(rows).T


def tradeable_pnl(y2, y5, y10, body_notional=1_000_000.0, lookback=500):
    """P&L of the factor-neutral fly as a position, not as a residual.

    A signal built on the fair-value residual is not by itself tradeable: the
    residual is the fly minus its fitted exposure to level, slope and vol, so
    capturing it means holding the hedges too. Reporting a Sharpe on the
    residual therefore flatters the strategy, because it charges nothing for
    the hedge and assumes it is free and exact.

    This is the honest version: the P&L of the PCA-neutral fly itself, weighted
    on a trailing window, which is a position that can actually be put on.
    """
    return pnl_pca_neutral(y2, y5, y10, body_notional, lookback)
