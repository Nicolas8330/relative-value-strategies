"""The 3-2-1 crack spread, a proxy for refining margin.

A refinery buys crude and sells products. The 3-2-1 ratio approximates a
typical US refinery's yield: three barrels of crude in, two barrels of
gasoline and one of distillate out.

The one thing that has to be right here is the units. Crude trades in dollars
per barrel; RBOB gasoline and heating oil trade in dollars per *gallon*. There
are 42 gallons in a barrel, so the product legs have to be multiplied by 42
before anything is subtracted. Getting this wrong is the most common error in
published crack spread code, and it produces a number that is off by a factor
of roughly forty, which is large enough to be obvious and small enough that
people still publish it.
"""

import numpy as np
import pandas as pd

GALLONS_PER_BARREL = 42.0


def crack_321(crude, gasoline, distillate):
    """3-2-1 crack spread in dollars per barrel of crude.

    Parameters are aligned price series: `crude` in $/bbl, `gasoline` and
    `distillate` in $/gal.

        crack = (2 * RB * 42 + 1 * HO * 42) / 3 - CL

    The division by three expresses the spread per barrel of crude input,
    which is how it is quoted. Multiply by three for the spread on the
    three-barrel basket actually traded.
    """
    frame = pd.concat([crude, gasoline, distillate], axis=1).dropna()
    frame.columns = ['crude', 'gasoline', 'distillate']

    products = (2.0 * frame['gasoline'] + 1.0 * frame['distillate']) \
        * GALLONS_PER_BARREL
    spread = products / 3.0 - frame['crude']
    spread.name = 'crack_321'
    return spread


def component_cracks(crude, gasoline, distillate):
    """The two single-product cracks, both in $/bbl.

    Worth separating because they are driven by different things: the gasoline
    crack is a summer driving-season story, the distillate crack a winter
    heating and diesel story. A 3-2-1 that looks stable can hide two legs
    moving in opposite directions.
    """
    frame = pd.concat([crude, gasoline, distillate], axis=1).dropna()
    frame.columns = ['crude', 'gasoline', 'distillate']
    return pd.DataFrame({
        'gasoline_crack': frame['gasoline'] * GALLONS_PER_BARREL - frame['crude'],
        'distillate_crack': frame['distillate'] * GALLONS_PER_BARREL - frame['crude'],
    })


def seasonality(spread, by='month'):
    """Average spread by calendar month, with dispersion.

    Seasonality in the crack is a genuine physical effect, not a statistical
    artefact: refineries run maintenance in the shoulder seasons and gasoline
    demand peaks in summer. That makes it one of the few places where a
    calendar effect has a mechanism behind it.
    """
    frame = pd.DataFrame({'spread': spread})
    frame['month'] = frame.index.month
    grouped = frame.groupby('month')['spread']
    return pd.DataFrame({
        'mean': grouped.mean(),
        'median': grouped.median(),
        'std': grouped.std(),
        'n': grouped.count(),
    })


def deseasonalise(spread):
    """Remove the monthly mean, so a signal does not just retrade the calendar.

    Without this, a z-score on the raw crack buys every winter and sells every
    summer and calls it mean reversion.
    """
    monthly = spread.groupby(spread.index.month).transform('mean')
    residual = spread - monthly
    residual.name = 'crack_deseasonalised'
    return residual


def summary(spread):
    """Descriptive statistics, in the units the spread is quoted in."""
    changes = spread.diff().dropna()
    return {
        'start': str(spread.index[0].date()),
        'end': str(spread.index[-1].date()),
        'observations': int(len(spread)),
        'mean': float(spread.mean()),
        'median': float(spread.median()),
        'std': float(spread.std()),
        'min': float(spread.min()),
        'max': float(spread.max()),
        'daily_vol': float(changes.std()),
        'annualised_vol': float(changes.std() * np.sqrt(252)),
    }
