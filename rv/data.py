"""Price and yield history, cached on disk so results are reproducible.

Energy futures come from Yahoo Finance as continuous front-month series, and
Treasury yields from FRED. Both are free and need no key, which is deliberate:
a reader should be able to reproduce every number in this repository without
an account anywhere.
"""

import io
import urllib.request
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parents[1] / 'data'

ENERGY = {'crude': 'CL=F', 'gasoline': 'RB=F', 'distillate': 'HO=F'}
YIELDS = {'2y': 'DGS2', '5y': 'DGS5', '10y': 'DGS10'}


def energy(start='2015-01-01', end=None, refresh=False):
    """Front-month crude, RBOB gasoline and heating oil settles.

    Crude is in dollars per barrel, the two products in dollars per gallon.
    They are returned in their native units rather than converted here, so the
    conversion stays visible in `crack.py` where it can be checked.

    These are continuous contracts: Yahoo rolls them, so the series contains
    roll gaps. That is fine for measuring the spread level and its
    seasonality, and it is not fine for claiming a tradeable P&L to the cent.
    The README says so.
    """
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"energy_{start}_{end or 'latest'}.csv"

    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    import yfinance as yf
    frames = {}
    for name, ticker in ENERGY.items():
        raw = yf.download(ticker, start=start, end=end, progress=False,
                          auto_adjust=False)
        if raw.empty:
            raise RuntimeError(f'no history returned for {ticker}')
        close = raw['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        frames[name] = close

    frame = pd.DataFrame(frames).dropna()
    frame.index.name = 'date'
    frame.to_csv(path)
    return frame


def treasury_yields(start='2015-01-01', end=None, refresh=False):
    """Constant maturity Treasury yields, in percent, from FRED.

    FRED publishes these as par yields on the H.15 curve. Non-trading days
    appear as a dot and are dropped rather than forward filled, so a holiday
    does not turn into a fabricated zero change.
    """
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"yields_{start}_{end or 'latest'}.csv"

    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    frames = {}
    for name, series_id in YIELDS.items():
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read().decode('utf-8')

        frame = pd.read_csv(io.StringIO(raw))
        frame.columns = ['date', 'value']
        frame = frame[frame['value'] != '.']
        frame['date'] = pd.to_datetime(frame['date'])
        frames[name] = frame.set_index('date')['value'].astype(float)

    out = pd.DataFrame(frames).dropna()
    out = out.loc[start:end] if end else out.loc[start:]
    out.index.name = 'date'
    out.to_csv(path)
    return out
