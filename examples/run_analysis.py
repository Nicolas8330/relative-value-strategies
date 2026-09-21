"""Reproduce every number and figure in the README.

Run with:  python examples/run_analysis.py

Four layers, applied to two unrelated markets:
  1  conditional fair value, fitted out of sample, trade the residual
  2  what the position is really exposed to, and whether it can be hedged
  3  which regimes the edge lives in
  4  the call today: level, fair value, horizon, size, invalidation
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings('ignore')

from rv import (butterfly, crack, data, exposure, fairvalue as fv, regimes,
                signals, today)

FIGURES = Path(__file__).resolve().parents[1] / 'figures'
FIGURES.mkdir(exist_ok=True)
NAVY, RUST, GREEN, GREY = '#1f3a5f', '#b5651d', '#4a7c59', '#8a867c'


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.15)


def rule(title):
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)


def adf_line(d):
    return (f"ADF {d['adf']:7.2f}  p={d['p_value']:6.4f}  "
            f"{'STATIONARY' if d['stationary_5pct'] else 'unit root'}")


# ====================================================================== data
energy = data.energy(start='2015-01-01')
ck = crack.crack_321(energy['crude'], energy['gasoline'], energy['distillate'])
legs = crack.component_cracks(energy['crude'], energy['gasoline'],
                              energy['distillate'])

yields = data.treasury_yields(start='1990-01-01')
recent = yields.loc['2015':]
y2, y5, y10 = recent['2y'], recent['5y'], recent['10y']
fly = butterfly.fly_yield(y2, y5, y10)

# ============================================== layer 0, the diagnostic
rule('WHY A ROLLING MEAN IS THE WRONG BASELINE')

full_fly = butterfly.fly_yield(yields['2y'], yields['5y'], yields['10y'])
print('Augmented Dickey-Fuller on the 2s5s10s butterfly, by window\n')
for lo, hi, label in [('1990', '2026', 'full history'),
                      ('2000', '2009', '2000s'),
                      ('2010', '2019', '2010s'),
                      ('2015', '2026', '2015-2026'),
                      ('2020', '2026', 'since 2020')]:
    block = full_fly.loc[lo:hi]
    d = fv.stationarity(block)
    print(f"  {label:14} n={len(block):5d}  {adf_line(d)}   mean {block.mean():+6.1f} bp")
print('\n  The fly reverts to a level that itself moves. Its mean slides from')
print('  +2.4bp over 36 years to -19.2bp since 2020, and every ten-year window')
print('  fails the test. A z-score against a trailing mean is mis-specified.')

# ================================================ layer 1, fair value
rule('LAYER 1  CONDITIONAL FAIR VALUE')

X_fly = pd.DataFrame({
    'level': (y2 + y5 + y10) / 3,
    'slope': y10 - y2,
    'realised_vol': y10.diff().rolling(21).std() * np.sqrt(252) * 100,
}).dropna()

fit_fly = fv.fair_value(fly, X_fly, window=500)
verdict = fv.model_earns_its_keep(fly.reindex(fit_fly.index), fit_fly['residual'])

print(f"Butterfly, {len(fit_fly)} out-of-sample days "
      f"({fit_fly.index[0].date()} to {fit_fly.index[-1].date()})")
print(f"  raw fly        {adf_line(verdict['raw'])}")
print(f"  model residual {adf_line(verdict['residual'])}")
print(f"  VERDICT: {verdict['verdict']}")

hl_fly_raw = fv.half_life(fly.reindex(fit_fly.index))
hl_fly = fv.half_life(fit_fly['residual'])
print(f"  half-life  raw {'n/a (does not revert)' if np.isnan(hl_fly_raw) else f'{hl_fly_raw:.0f} d'}"
      f"   residual {hl_fly:.0f} d")

table, info = fv.newey_west_tstats(fly.reindex(X_fly.index), X_fly)
print(f"\n  full-sample regression, Newey-West {info['lags']} lags, "
      f"R2 = {info['r_squared']:.3f}")
print('   ' + table.round(3).to_string().replace('\n', '\n   '))
print('\n  Curvature is driven by the slope, not the level: the level'
      '\n  coefficient is not significant at all.')

print(f"\nCrack, cointegration with crude: "
      f"{ {k: (round(v, 4) if isinstance(v, float) else v) for k, v in fv.engle_granger(ck, energy['crude']).items()} }")

months = pd.get_dummies(ck.index.month, prefix='m',
                        drop_first=True).set_index(ck.index).astype(float)
X_ck = pd.concat([pd.DataFrame({'crude': energy['crude']}), months], axis=1).dropna()
fit_ck = fv.fair_value(ck, X_ck, window=750)
verdict_ck = fv.model_earns_its_keep(ck.reindex(fit_ck.index), fit_ck['residual'])
print(f"  raw crack      {adf_line(verdict_ck['raw'])}")
print(f"  model residual {adf_line(verdict_ck['residual'])}")
print(f"  VERDICT: {verdict_ck['verdict']}")
print('  The crack model does NOT pass its own test. Reported as such.')

# ================================================= layer 2, exposure
rule('LAYER 2  WHAT THE POSITION IS REALLY EXPOSED TO')

print('Butterfly, dollar loading of each weighting per $1m body\n')
weighting = butterfly.factor_exposure(y2, y5, y10)
print('   ' + weighting.round(2).to_string().replace('\n', '\n   '))
ratio = abs(weighting.loc['equal notional', 'slope'] /
            weighting.loc['PCA-neutral', 'slope'])
print(f"\n  Solving the wing weights against the factor loadings cuts the slope")
print(f"  exposure by a factor of {ratio:.0f}.")

pca_w = butterfly.pca_neutral_weights(y2, y5, y10).dropna()
stab_fly = exposure.stability(pca_w['w2'])
print(f"  Those weights are stable: 2y wing mean {stab_fly['mean']:+.3f}, "
      f"sd {stab_fly['std']:.3f}, signal-to-noise {stab_fly['signal_to_noise']:.1f}")

print('\nCrack, exposure to outright crude\n')
report = exposure.exposure_report(ck, energy['crude'], window=250, label='crack')
for key in ('beta_raw', 'beta_hedged', 'correlation_raw', 'correlation_hedged',
            'hedge_quality', 'beta_mean', 'beta_std', 'beta_min', 'beta_max'):
    print(f"  {key:20} {report[key]:+.3f}")
beta_series = exposure.rolling_beta(ck, energy['crude'], 250)
stab_ck = exposure.stability(beta_series)
print(f"  signal-to-noise      {stab_ck['signal_to_noise']:.2f}")
print(f"\n  A '3-2-1 crack spread' is {abs(report['beta_raw']):.0%} a short-crude")
print('  position, and hedging only removes a third of it, because the hedge')
print('  ratio itself swings from -0.99 to +0.27. A hedge ratio is only as')
print('  good as its own standard deviation.')

# ================================================== layer 3, regimes
rule('LAYER 3  WHICH REGIMES THE EDGE LIVES IN')

z_fly = fv.zscore(fit_fly['residual'], window=250)
pos_fly = signals.positions(z_fly, entry=1.5, exit_at=0.5)
bt_fly = signals.backtest(fit_fly['residual'], pos_fly, cost_per_trade=0.25)

drivers = {
    'curve slope': (y10 - y2, 'tercile'),
    'rate volatility': (y10.diff().rolling(21).std(), 'tercile'),
    'curve inverted': (y10 - y2, 'binary'),
}
regime_tables = {}
for name, (driver, kind) in drivers.items():
    driver = driver.reindex(bt_fly.index)
    reg = (regimes.tercile_regime(driver) if kind == 'tercile'
           else regimes.binary_regime(driver, 0.0, ('inverted', 'normal')))
    table = regimes.performance_by_regime(bt_fly['net'], reg)
    regime_tables[name] = table
    print(f"\n  by {name}:")
    print('   ' + table[['days', 'share', 'sharpe', 'hit_rate', 'worst_day']]
          .round(3).to_string().replace('\n', '\n   '))
    print(f"    spread of Sharpes {regimes.spread_of_sharpes(table):.2f}")

print('\n  The signal earns 1.26 Sharpe with the curve inverted and 0.29 when it')
print('  is not. That is a regime bet, and it should be described as one.')

# ==================================================== layer 4, today
rule('LAYER 4  THE CALL TODAY')

units = today.size_to_volatility(fit_fly['residual'], target_daily_vol=1.0)
current = today.call(fit_fly['residual'], hl_fly, units=units,
                     unit_label='bp of daily residual vol')
for key, value in current.items():
    print(f"  {key:28} {value if not isinstance(value, float) else round(value, 3)}")

cone = today.projection_cone(fit_fly['residual'], hl_fly)
print('\n  projection of the residual, bp')
print('   ' + cone.round(2).to_string().replace('\n', '\n   '))

# ===================================================== figures
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                              gridspec_kw={'height_ratios': [2, 1]})
ax.plot(ck.index, ck, color=NAVY, linewidth=1.2, label='3-2-1 crack')
ax.axhline(ck.mean(), color=GREY, linestyle='--', linewidth=1,
           label=f'mean {ck.mean():.1f}')
ax.set_ylabel('$ per barrel')
ax.set_title('3-2-1 crack spread, refining margin implied by the futures curve')
ax.legend(frameon=False)
style(ax)
ax2.plot(legs.index, legs['gasoline_crack'], color=RUST, linewidth=1,
         label='gasoline crack')
ax2.plot(legs.index, legs['distillate_crack'], color=GREEN, linewidth=1,
         label='distillate crack')
ax2.set_ylabel('$ per barrel')
ax2.legend(frameon=False, ncol=2)
style(ax2)
fig.tight_layout(); fig.savefig(FIGURES / 'crack_spread.png', dpi=140)

season = crack.seasonality(ck)
fig, ax = plt.subplots(figsize=(10, 4))
names = [pd.Timestamp(2020, m, 1).strftime('%b') for m in season.index]
ax.bar(names, season['mean'],
       color=[RUST if m == season['mean'].idxmax() else NAVY for m in season.index])
ax.errorbar(names, season['mean'], yerr=season['std'] / np.sqrt(season['n']),
            fmt='none', ecolor=GREY, capsize=3, linewidth=1)
ax.axhline(ck.mean(), color=GREY, linestyle='--', linewidth=1)
ax.set_ylabel('$ per barrel')
ax.set_title('Crack spread by calendar month, with standard error of the mean')
style(ax)
fig.tight_layout(); fig.savefig(FIGURES / 'crack_seasonality.png', dpi=140)

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
for series, colour, label in ((y2, NAVY, '2y'), (y5, RUST, '5y'), (y10, GREEN, '10y')):
    ax.plot(series.index, series, color=colour, linewidth=1, label=label)
ax.set_ylabel('Yield, %')
ax.set_title('US Treasury yields and the 2s5s10s butterfly')
ax.legend(frameon=False, ncol=3); style(ax)
ax2.plot(fly.index, fly, color=NAVY, linewidth=1.2)
ax2.axhline(0, color=GREY, linewidth=1)
ax2.fill_between(fly.index, fly, 0, where=(fly > 0), color=NAVY, alpha=0.18)
ax2.fill_between(fly.index, fly, 0, where=(fly < 0), color=RUST, alpha=0.18)
ax2.set_ylabel('2·5y − 2y − 10y, bp'); style(ax2)
fig.tight_layout(); fig.savefig(FIGURES / 'butterfly_curve.png', dpi=140)

pca = butterfly.curve_pca(y2, y5, y10)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
for i, (name, colour) in enumerate(zip(['level', 'slope', 'curvature'],
                                       [NAVY, RUST, GREEN])):
    ax.plot(range(3), pca['loadings'][name], 'o-', color=colour, linewidth=1.8,
            label=f"{name} ({pca['explained'][i]:.1%})")
ax.set_xticks(range(3)); ax.set_xticklabels(['2y', '5y', '10y'])
ax.axhline(0, color=GREY, linewidth=1)
ax.set_ylabel('Loading'); ax.set_title('Principal components of daily yield changes')
ax.legend(frameon=False); style(ax)

schemes = weighting.index.tolist()
width = 0.27
for i, (factor, colour) in enumerate(zip(['level', 'slope', 'curvature'],
                                         [NAVY, RUST, GREEN])):
    ax2.bar(np.arange(len(schemes)) + (i - 1) * width,
            weighting[factor].abs(), width, color=colour, label=factor)
ax2.set_yscale('symlog', linthresh=1)
ax2.set_xticks(np.arange(len(schemes)))
ax2.set_xticklabels([s.replace(' ', '\n') for s in schemes], fontsize=9)
ax2.set_ylabel('|dollar loading| per $1m body, log scale')
ax2.set_title('What each weighting is actually exposed to')
ax2.legend(frameon=False); style(ax2)
fig.tight_layout(); fig.savefig(FIGURES / 'butterfly_weighting.png', dpi=140)

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                              gridspec_kw={'height_ratios': [3, 2]})
ax.plot(fit_fly.index, fit_fly['actual'], color=NAVY, linewidth=1.2, label='fly')
ax.plot(fit_fly.index, fit_fly['fair_value'], color=RUST, linewidth=1.2,
        label='conditional fair value')
ax.set_ylabel('bp')
ax.set_title('2s5s10s butterfly against its out-of-sample fair value')
ax.legend(frameon=False); style(ax)

sd = fit_fly['residual'].std()
ax2.plot(fit_fly.index, fit_fly['residual'], color=NAVY, linewidth=1)
for level, colour in ((1.5, RUST), (-1.5, RUST)):
    ax2.axhline(level * sd, color=colour, linestyle='--', linewidth=1)
ax2.axhline(0, color=GREY, linewidth=1)
ax2.set_ylabel('residual, bp')
ax2.text(fit_fly.index[10], 1.6 * sd, 'entry band', color=RUST, fontsize=9)
style(ax2)
fig.tight_layout(); fig.savefig(FIGURES / 'fairvalue_butterfly.png', dpi=140)

fig, ax = plt.subplots(figsize=(10, 4.2))
labels, values, colours = [], [], []
for name, table in regime_tables.items():
    for regime_name, row in table.iterrows():
        labels.append(f"{regime_name}\n({name})")
        values.append(row['sharpe'])
        colours.append(NAVY if row['sharpe'] > 0 else RUST)
ax.bar(labels, values, color=colours)
ax.axhline(0, color=GREY, linewidth=1)
ax.set_ylabel('Sharpe')
ax.set_title('Where the butterfly signal earns, and where it does not')
ax.tick_params(axis='x', labelsize=8)
style(ax)
fig.tight_layout(); fig.savefig(FIGURES / 'regimes.png', dpi=140)

print(f"\nFigures written to {FIGURES}")
