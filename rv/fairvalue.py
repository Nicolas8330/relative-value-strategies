"""Conditional fair value, and the residual you actually trade.

The motivation is a diagnostic, not a preference. Run an augmented
Dickey-Fuller test on the 2s5s10s butterfly and the answer depends entirely on
the window:

    1990-2026   p = 0.002   stationary
    2010-2019   p = 0.120   unit root
    2015-2026   p = 0.084   unit root
    2020-2026   p = 0.249   unit root

and the sample mean slides from +2.4bp over the full history to -19.2bp since
2020. The fly reverts to a level that itself moves. A z-score against a
trailing mean is therefore mis-specified: it keeps insisting the fly is cheap
while the level it reverts to walks away underneath it.

The fix is to model the level it should be at, conditional on what is already
known, and trade the residual. The test of whether the model earned its keep
is simple and hard to fudge: the raw series must fail a stationarity test and
the residual must pass it.

Everything here is fitted on a trailing window that ends the day before the
observation it prices. A fair value fitted on the whole sample and then
"traded" is not a model, it is a description of what already happened.
"""

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller, coint
except ImportError:                                   # pragma: no cover
    adfuller = coint = None


def rolling_ols(y, X, window=500, expanding=False, min_periods=250):
    """Coefficients of y on X, refitted at each date on past data only.

    Returns a frame of coefficients indexed like `y`, with a constant in the
    first column. Row t holds the fit estimated on observations strictly
    before t, so it can be used to price t without peeking.

    Computed from running sums rather than by refitting in a loop. A window of
    w observations needs X'X and X'y over that window, and both are differences
    of cumulative sums, so the whole path costs O(n k^2) instead of O(n w k^2).
    With four regressors and four thousand days that is the difference between
    a few milliseconds and half a minute, which is what makes it usable as a
    slider rather than a batch job.
    """
    frame = pd.concat([y.rename('y'), X], axis=1).dropna()
    names = ['const'] + list(X.columns)

    values = frame.values
    n = len(frame)
    out = pd.DataFrame(index=frame.index, columns=names, dtype=float)
    start = min_periods if expanding else window
    if n <= start:
        return out

    yy = values[:, 0]
    xx = np.column_stack([np.ones(n), values[:, 1:]])
    k = xx.shape[1]

    # Cumulative X'X and X'y, with a zero row in front so that a window can be
    # taken as a plain difference of two rows.
    gram = np.zeros((n + 1, k, k))
    moment = np.zeros((n + 1, k))
    gram[1:] = np.cumsum(xx[:, :, None] * xx[:, None, :], axis=0)
    moment[1:] = np.cumsum(xx * yy[:, None], axis=0)

    ends = np.arange(start, n)
    starts = np.zeros_like(ends) if expanding else ends - window

    A = gram[ends] - gram[starts]
    b = moment[ends] - moment[starts]

    # A singular window would raise for the whole batch, so solve in one call
    # and fall back to a least-squares solve only for the rows that fail.
    try:
        # NumPy 2 reads a 2-D right-hand side as a matrix, so the batch has to
        # be given an explicit trailing axis and squeezed back afterwards.
        beta = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        beta = np.full((len(ends), k), np.nan)
        for i in range(len(ends)):
            try:
                beta[i] = np.linalg.solve(A[i], b[i])
            except np.linalg.LinAlgError:
                beta[i], *_ = np.linalg.lstsq(A[i], b[i], rcond=None)

    out.iloc[start:] = beta
    return out


def fair_value(y, X, window=500, expanding=False, min_periods=250):
    """Out-of-sample fitted value and residual.

    The residual is the trading signal: positive means the series sits above
    what the model says it should, given the conditioning variables.
    """
    coefficients = rolling_ols(y, X, window, expanding, min_periods)
    aligned = X.reindex(coefficients.index)

    fitted = coefficients['const'].copy()
    for name in X.columns:
        fitted += coefficients[name] * aligned[name]

    residual = y.reindex(coefficients.index) - fitted
    return pd.DataFrame({
        'actual': y.reindex(coefficients.index),
        'fair_value': fitted,
        'residual': residual,
    }).dropna()


def newey_west_tstats(y, X, lags=None):
    """OLS with HAC standard errors.

    Daily financial residuals are strongly autocorrelated, so ordinary
    standard errors overstate significance, often by a factor of two or three.
    Newey-West corrects for that. The default lag follows the usual
    4*(n/100)^(2/9) rule.
    """
    frame = pd.concat([y.rename('y'), X], axis=1).dropna()
    yy = frame['y'].values
    xx = np.column_stack([np.ones(len(frame)), frame[X.columns].values])
    n, k = xx.shape

    beta, *_ = np.linalg.lstsq(xx, yy, rcond=None)
    residuals = yy - xx @ beta

    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))

    XtX_inv = np.linalg.pinv(xx.T @ xx)
    S = (residuals[:, None] * xx).T @ (residuals[:, None] * xx)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        u_t = residuals[lag:, None] * xx[lag:]
        u_lag = residuals[:-lag, None] * xx[:-lag]
        cross = u_t.T @ u_lag
        S += weight * (cross + cross.T)

    covariance = XtX_inv @ S @ XtX_inv
    errors = np.sqrt(np.diag(covariance))

    names = ['const'] + list(X.columns)
    total = ((yy - yy.mean()) ** 2).sum()
    return pd.DataFrame({
        'coefficient': beta,
        'hac_se': errors,
        't_stat': beta / errors,
    }, index=names), {
        'observations': n,
        'lags': lags,
        'r_squared': 1.0 - (residuals ** 2).sum() / total,
    }


def stationarity(series, name='series'):
    """Augmented Dickey-Fuller, reported rather than interpreted for you."""
    if adfuller is None:
        raise ImportError('statsmodels is required for stationarity testing')
    clean = pd.Series(series).dropna()
    statistic, pvalue, lags, nobs, critical, _ = adfuller(clean, autolag='AIC')
    return {
        'name': name,
        'adf': float(statistic),
        'p_value': float(pvalue),
        'lags': int(lags),
        'observations': int(nobs),
        'critical_5pct': float(critical['5%']),
        'stationary_5pct': bool(pvalue < 0.05),
    }


def model_earns_its_keep(raw, residual):
    """The test the whole approach stands or falls on.

    A conditional fair value is worth building only if it turns something that
    wanders into something that reverts. Comparing the two ADF results is the
    cheapest way to see whether that happened, and it is not a test you can
    pass by accident.
    """
    before = stationarity(raw, 'raw series')
    after = stationarity(residual, 'model residual')
    return {
        'raw': before,
        'residual': after,
        'verdict': (
            'model adds information'
            if after['stationary_5pct'] and not before['stationary_5pct']
            else 'both stationary, model may be unnecessary'
            if after['stationary_5pct'] and before['stationary_5pct']
            else 'residual still wanders, model does not help'
        ),
    }


def half_life(series, significance=0.05):
    """Ornstein-Uhlenbeck half-life in days, NaN when the series does not revert.

    The significance gate is not decoration. Regressing the change of a random
    walk on its own lagged level produces a slightly negative slope in any
    finite sample - this is the Dickey-Fuller bias - so the naive formula
    returns a confident-looking half-life for a series that does not revert at
    all. Requiring the series to pass a stationarity test first is what stops
    the function inventing mean reversion out of a unit root.

    Pass significance=None to get the raw estimate anyway.
    """
    clean = pd.Series(series).dropna()
    if len(clean) < 30:
        return np.nan

    if significance is not None:
        if adfuller is None:
            raise ImportError('statsmodels is required; pass significance=None '
                              'for the ungated estimate')
        if adfuller(clean, autolag='AIC')[1] > significance:
            return np.nan

    level = clean.shift(1)
    change = clean.diff()
    frame = pd.concat([level, change], axis=1).dropna()
    frame.columns = ['level', 'change']

    x = frame['level'] - frame['level'].mean()
    slope = float((x * frame['change']).sum() / (x * x).sum())
    return float(-np.log(2) / slope) if slope < 0 else np.nan


def engle_granger(y, x):
    """Cointegration test, for the case where levels really are related.

    The 3-2-1 crack is not stationary on its own but is cointegrated with
    crude, which is the statistical statement of something obvious: a refining
    margin cannot drift away from the price of its own input forever. That
    makes an error-correction specification legitimate where a plain
    regression in levels would be spurious.
    """
    if coint is None:
        raise ImportError('statsmodels is required for cointegration testing')
    frame = pd.concat([y, x], axis=1).dropna()
    statistic, pvalue, critical = coint(frame.iloc[:, 0], frame.iloc[:, 1])
    return {
        'statistic': float(statistic),
        'p_value': float(pvalue),
        'critical_5pct': float(critical[1]),
        'cointegrated_5pct': bool(pvalue < 0.05),
    }


def zscore(residual, window=250, min_periods=None):
    """Z-score of the residual against its own trailing distribution.

    Applied to the residual rather than to the raw series, which is the whole
    point: the scaling is done on what the model could not explain.
    """
    rolling = residual.rolling(window, min_periods=min_periods or window // 2)
    return (residual - rolling.mean()) / rolling.std()
