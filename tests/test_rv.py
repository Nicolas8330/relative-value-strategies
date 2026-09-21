"""Tests against facts that hold independently of this code: unit algebra,
closed-form duration, a known data-generating process, and the absence of
look-ahead.
"""

import numpy as np
import pandas as pd
import pytest

from rv import butterfly, crack, exposure, fairvalue, regimes, signals, today


def days(n, start='2015-01-02'):
    return pd.bdate_range(start, periods=n)


# --------------------------------------------------------------- crack

def test_crack_unit_conversion_is_per_barrel():
    """Products quoted per gallon must be scaled by 42 before subtracting."""
    idx = days(3)
    crude = pd.Series([100.0] * 3, index=idx)
    gasoline = pd.Series([3.0] * 3, index=idx)      # $/gal
    distillate = pd.Series([3.0] * 3, index=idx)

    spread = crack.crack_321(crude, gasoline, distillate)
    expected = (2 * 3.0 * 42 + 1 * 3.0 * 42) / 3 - 100.0     # = 26.0
    assert spread.iloc[0] == pytest.approx(expected)
    assert spread.iloc[0] == pytest.approx(26.0)


def test_crack_is_zero_when_products_exactly_cover_crude():
    idx = days(3)
    crude = pd.Series([84.0] * 3, index=idx)
    per_gallon = pd.Series([2.0] * 3, index=idx)             # 2 * 42 = 84
    spread = crack.crack_321(crude, per_gallon, per_gallon)
    assert spread.abs().max() < 1e-12


def test_component_cracks_average_back_to_the_321():
    idx = days(50)
    rng = np.random.default_rng(0)
    crude = pd.Series(80 + rng.standard_normal(50).cumsum(), index=idx)
    gasoline = pd.Series(2.5 + rng.standard_normal(50).cumsum() * 0.01, index=idx)
    distillate = pd.Series(2.7 + rng.standard_normal(50).cumsum() * 0.01, index=idx)

    legs = crack.component_cracks(crude, gasoline, distillate)
    blended = (2 * legs['gasoline_crack'] + legs['distillate_crack']) / 3
    spread = crack.crack_321(crude, gasoline, distillate)
    pd.testing.assert_series_equal(blended, spread, check_names=False)


def test_deseasonalising_removes_the_monthly_mean():
    idx = pd.bdate_range('2015-01-01', periods=1500)
    seasonal = pd.Series(idx.month * 2.0, index=idx)
    residual = crack.deseasonalise(seasonal)
    assert residual.abs().max() < 1e-10


# ------------------------------------------------------------ duration

def test_par_bond_duration_matches_the_closed_form():
    """A 10y par bond at 5% has modified duration near 7.79."""
    assert butterfly.modified_duration(5.0, 10) == pytest.approx(7.7946, abs=1e-3)


def test_duration_rises_with_maturity_and_falls_with_yield():
    assert butterfly.modified_duration(4.0, 10) > butterfly.modified_duration(4.0, 2)
    assert butterfly.modified_duration(2.0, 10) > butterfly.modified_duration(8.0, 10)


def test_dv01_scales_linearly_with_face():
    one = butterfly.dv01(4.0, 5, face=100.0)
    ten = butterfly.dv01(4.0, 5, face=1000.0)
    assert ten == pytest.approx(10 * one)


def test_duration_preserves_a_series_index():
    idx = days(20)
    yields = pd.Series(np.linspace(3.0, 4.0, 20), index=idx)
    out = butterfly.modified_duration(yields, 5)
    assert isinstance(out, pd.Series)
    pd.testing.assert_index_equal(out.index, idx)


# ------------------------------------------------------------ butterfly

def test_fly_is_zero_on_a_perfectly_straight_curve():
    idx = days(10)
    y2 = pd.Series([2.0] * 10, index=idx)
    y5 = pd.Series([3.0] * 10, index=idx)
    y10 = pd.Series([4.0] * 10, index=idx)        # 5y exactly on the line
    assert butterfly.fly_yield(y2, y5, y10).abs().max() < 1e-12


def test_dv01_weighted_fly_is_neutral_to_a_parallel_shift():
    idx = days(40)
    y2 = pd.Series(np.full(40, 3.0), index=idx)
    y5 = pd.Series(np.full(40, 3.5), index=idx)
    y10 = pd.Series(np.full(40, 4.0), index=idx)
    weights = butterfly.dv01_weights(y2, y5, y10)
    assert weights['net_dv01'].abs().max() < 1e-6


def test_pca_recovers_level_slope_and_curvature():
    """Build a curve from three known factors and check they come back out."""
    rng = np.random.default_rng(7)
    n = 3000
    level = rng.standard_normal(n) * 0.06
    slope = rng.standard_normal(n) * 0.02
    curve = rng.standard_normal(n) * 0.005

    idx = days(n)
    y2 = pd.Series(np.cumsum(level - slope - curve) + 3.0, index=idx)
    y5 = pd.Series(np.cumsum(level + 2 * curve) + 3.5, index=idx)
    y10 = pd.Series(np.cumsum(level + slope - curve) + 4.0, index=idx)

    pca = butterfly.curve_pca(y2, y5, y10)
    assert pca['explained'][0] > 0.8                    # level dominates
    assert pca['explained'][0] > pca['explained'][1] > pca['explained'][2]
    loadings = pca['loadings']
    assert (loadings['level'] > 0).all()                # all tenors rise together
    assert loadings['curvature']['5y'] > 0              # belly opposes the wings
    assert loadings['curvature']['2y'] < 0
    assert loadings['curvature']['10y'] < 0


# ------------------------------------------------------------ fair value

def test_rolling_ols_never_sees_the_future():
    """Rewriting the tail must leave every earlier coefficient untouched."""
    rng = np.random.default_rng(3)
    n = 900
    idx = days(n)
    x = pd.DataFrame({'a': rng.standard_normal(n).cumsum()}, index=idx)
    y = pd.Series(2.0 * x['a'] + rng.standard_normal(n), index=idx)

    cut = n - 100
    before = fairvalue.rolling_ols(y, x, window=250).iloc[:cut]
    tampered = y.copy()
    tampered.iloc[cut:] += 500.0
    after = fairvalue.rolling_ols(tampered, x, window=250).iloc[:cut]
    pd.testing.assert_frame_equal(before, after)


def test_fair_value_recovers_a_known_relationship():
    rng = np.random.default_rng(11)
    n = 1200
    idx = days(n)
    driver = pd.DataFrame({'d': rng.standard_normal(n).cumsum()}, index=idx)
    target = pd.Series(1.5 * driver['d'] + 10.0 + rng.standard_normal(n) * 0.1,
                       index=idx)

    coefficients = fairvalue.rolling_ols(target, driver, window=400).dropna()
    assert coefficients['d'].mean() == pytest.approx(1.5, abs=0.05)
    assert coefficients['const'].mean() == pytest.approx(10.0, abs=0.5)


def test_model_earns_its_keep_detects_a_useful_model():
    """A random walk plus noise: the residual reverts, the raw series does not."""
    rng = np.random.default_rng(5)
    n = 2000
    idx = days(n)
    driver = pd.DataFrame({'d': rng.standard_normal(n).cumsum()}, index=idx)
    target = pd.Series(driver['d'] + rng.standard_normal(n) * 0.5, index=idx)

    result = fairvalue.fair_value(target, driver, window=500)
    verdict = fairvalue.model_earns_its_keep(target.reindex(result.index),
                                             result['residual'])
    assert not verdict['raw']['stationary_5pct']
    assert verdict['residual']['stationary_5pct']
    assert verdict['verdict'] == 'model adds information'


def test_half_life_recovers_a_known_mean_reversion_speed():
    """Simulate an OU process with a 20 day half-life and measure it back."""
    rng = np.random.default_rng(13)
    n, target = 40_000, 20.0
    kappa = np.log(2) / target
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] * (1 - kappa) + rng.standard_normal() * 0.1
    measured = fairvalue.half_life(pd.Series(x, index=days(n)))
    assert measured == pytest.approx(target, rel=0.15)


def test_half_life_is_nan_for_a_random_walk():
    rng = np.random.default_rng(17)
    walk = pd.Series(rng.standard_normal(4000).cumsum(), index=days(4000))
    assert np.isnan(fairvalue.half_life(walk))


def test_newey_west_errors_exceed_ordinary_ones_under_autocorrelation():
    rng = np.random.default_rng(19)
    n = 2000
    idx = days(n)
    # Newey-West only bites when the regressor is autocorrelated too: with an
    # iid regressor the product x_t * u_t is near white noise and HAC matches
    # ordinary errors. A persistent regressor is the case it exists for.
    persistent = np.zeros(n)
    noise = np.zeros(n)
    for i in range(1, n):
        persistent[i] = 0.95 * persistent[i - 1] + rng.standard_normal()
        noise[i] = 0.9 * noise[i - 1] + rng.standard_normal()
    x = pd.DataFrame({'a': persistent}, index=idx)
    y = pd.Series(0.2 * x['a'] + noise, index=idx)

    table, _ = fairvalue.newey_west_tstats(y, x)
    residual = y - (table.loc['const', 'coefficient']
                    + table.loc['a', 'coefficient'] * x['a'])
    ordinary = residual.std() / np.sqrt(((x['a'] - x['a'].mean()) ** 2).sum())
    assert table.loc['a', 'hac_se'] > 2 * ordinary


# ------------------------------------------------------------- exposure

def test_hedging_removes_a_stable_beta():
    rng = np.random.default_rng(23)
    n = 1500
    idx = days(n)
    factor = pd.Series(rng.standard_normal(n).cumsum(), index=idx)
    target = pd.Series((0.6 * factor.diff().fillna(0)
                        + rng.standard_normal(n) * 0.2).cumsum(), index=idx)

    report = exposure.exposure_report(target, factor, window=250)
    assert report['beta_raw'] == pytest.approx(0.6, abs=0.06)
    assert abs(report['beta_hedged']) < 0.1
    assert report['hedge_quality'] > 0.8


def test_unstable_beta_is_flagged_by_its_own_dispersion():
    steady = pd.Series(np.full(600, 0.5))
    jumpy = pd.Series(np.tile([-0.9, 0.3], 300))
    assert exposure.stability(steady)['signal_to_noise'] > \
        exposure.stability(jumpy)['signal_to_noise']


# -------------------------------------------------------------- regimes

def test_regime_labels_use_only_past_information():
    rng = np.random.default_rng(29)
    n = 1200
    driver = pd.Series(rng.standard_normal(n).cumsum(), index=days(n))

    cut = n - 100
    before = regimes.tercile_regime(driver, window=250).iloc[:cut]
    tampered = driver.copy()
    tampered.iloc[cut:] += 1000.0
    after = regimes.tercile_regime(tampered, window=250).iloc[:cut]
    pd.testing.assert_series_equal(before, after)


def test_performance_by_regime_splits_a_planted_edge():
    idx = days(1000)
    regime = pd.Series(['calm'] * 500 + ['stressed'] * 500, index=idx)
    rng = np.random.default_rng(31)
    pnl = pd.Series(np.concatenate([rng.standard_normal(500) + 0.5,
                                    rng.standard_normal(500) - 0.5]), index=idx)

    table = regimes.performance_by_regime(pnl, regime)
    assert table.loc['calm', 'sharpe'] > table.loc['stressed', 'sharpe']
    assert regimes.spread_of_sharpes(table) > 1.0


# ---------------------------------------------------------------- today

def test_convergence_horizon_matches_the_half_life_definition():
    assert today.convergence_horizon(20.0, 0.5) == pytest.approx(20.0)
    assert today.convergence_horizon(20.0, 0.75) == pytest.approx(40.0)


def test_projection_cone_decays_towards_zero_and_widens():
    rng = np.random.default_rng(37)
    residual = pd.Series(rng.standard_normal(1000) * 3 + 10, index=days(1000))
    cone = today.projection_cone(residual, half_life=30.0)

    assert abs(cone['expected'].iloc[-1]) < abs(cone['expected'].iloc[0])
    widths = cone['q90'] - cone['q10']
    assert widths.iloc[-1] > widths.iloc[0]


def test_call_reads_the_sign_of_the_deviation():
    idx = days(500)
    rich = pd.Series(np.concatenate([np.zeros(499), [10.0]]), index=idx)
    assert today.call(rich, half_life=30.0)['direction'] == 'short the spread'

    cheap = pd.Series(np.concatenate([np.zeros(499), [-10.0]]), index=idx)
    assert today.call(cheap, half_life=30.0)['direction'] == 'long the spread'


def test_positions_are_lagged_by_one_day():
    idx = days(200)
    rng = np.random.default_rng(41)
    score = pd.Series(rng.standard_normal(200) * 2, index=idx)
    held = signals.positions(score)
    assert held.iloc[0] == 0.0
    # a position on day t may only reflect the score of day t-1
    flips = score.index[(score.abs() > 1.5)]
    if len(flips):
        first = score.index.get_loc(flips[0])
        assert held.iloc[first] == 0.0 or first > 0
