"""Futures signals and portfolio engines (factors/futures_signals.py, backtest/futures_portfolio.py).

The DolphinDB conventions below were probed on the ddb server (v2.00.19) on 2026-10-10 and are
pinned here, because matching them is what lets the two engines reconcile to 1e-16 on the real
panel (all six batch-3 portfolios, research/reconcile_futures.py).
"""

import numpy as np
import pandas as pd
import pytest

from backtest.futures_portfolio import calendar, group_rank, long_short, time_series_portfolio, vol_target
from factors.futures_signals import SIGNALS, rolling_rows, with_returns


@pytest.mark.parametrize("values, expected", [
    (list(range(1, 8)), [0, 0, 1, 1, 2, 3, 4]),
    (list(range(1, 12)), [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
    (list(range(1, 14)), [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4]),
    ([1, 1, 1, 1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 0, 2, 2, 3, 3, 4, 4]),
    ([1, 2, 3, 3, 3, 3, 4, 5, 6, 7], [0, 0, 1, 1, 1, 1, 3, 3, 4, 4]),
    ([np.nan, 1, 2, 3, 4, 5, 6, 7], [np.nan, 0, 0, 1, 1, 2, 3, 4]),
])
def test_group_rank_matches_dolphindb_rank_with_groups(values, expected):
    """DolphinDB rank(x, true, 5): larger groups first, ties take the group of their lowest rank."""
    v = pd.Series(values, dtype=float)
    got = group_rank(v, pd.Series(0, index=v.index), 5)
    np.testing.assert_array_equal(got.to_numpy(), np.array(expected, dtype=float))


def test_group_rank_treats_float_noise_as_a_tie():
    v = pd.Series([0.1 + 0.2, 0.3, 0.5, 0.7, 0.9])            # 0.1 + 0.2 != 0.3 in floating point
    got = group_rank(v, pd.Series(0, index=v.index), 5)
    assert got[0] == got[1]


@pytest.mark.parametrize("values, how, expected", [
    ([np.nan, 1, 2, 4, 8, 16], "std", [np.nan, np.nan, 0.7071067811865476, 1.5275252316519468,
                                        3.0550504633038935, 6.110100926607787]),
    ([np.nan, 1, 2, 4, 8, 16], "mean", [np.nan, np.nan, 1.5, 2.3333333333333335, 4.666666666666667,
                                         9.333333333333334]),
    ([np.nan, 1, 2, 4, 8, 16], "max", [np.nan, np.nan, 2, 4, 8, 16]),
    ([1, np.nan, np.nan, 2, 4, 8, 16], "std", [np.nan, np.nan, np.nan, np.nan, 1.4142135623730951,
                                                3.0550504633038935, 6.110100926607787]),
])
def test_rolling_rows_matches_dolphindb_moving_windows(values, how, expected):
    """mstd/mavg/mmax(x, 3): a value once 3 rows have passed, over the non-missing ones."""
    v = pd.Series(values, dtype=float)
    got = rolling_rows(v, pd.Series(0, index=v.index), 3, how)
    np.testing.assert_allclose(got.to_numpy(), np.array(expected, dtype=float), equal_nan=True)


def _panel(n_days=420, n_syms=12, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2015-01-01", periods=n_days)
    rows = []
    for i in range(n_syms):
        close = 100 * np.exp(np.cumsum(rng.normal(0.0002 * (i - 6), 0.015, n_days)))
        for d, c in zip(days, close):
            rows.append({"date": d, "sym": f"C{i:02d}", "open": c, "high": c * 1.01, "low": c * 0.99,
                         "close": c, "ry": rng.normal(0, 0.1), "uni": True})
    return calendar(with_returns(pd.DataFrame(rows)))


@pytest.mark.parametrize("name", list(SIGNALS))
def test_signals_never_use_future_bars(name):
    px = _panel()
    cutoff = px["date"].sort_values().unique()[300]
    full = SIGNALS[name].compute(px)
    part_px = calendar(with_returns(px[px["date"] <= cutoff].drop(columns=["di", "me", "ret", "fwd", "vol60"])))
    part = SIGNALS[name].compute(part_px)
    a = full[px["date"] <= cutoff].to_numpy()
    b = part.to_numpy()
    assert np.allclose(a, b, equal_nan=True)


def test_time_series_portfolio_is_inverse_vol_and_fully_invested():
    px = _panel()
    p = time_series_portfolio(px, SIGNALS["MA20_60"].compute(px), 1, 3.0)
    assert len(p) > 300 and p["turn"].iloc[0] == pytest.approx(1.0)   # first day buys a gross-1 book
    assert np.isfinite(p["net"]).all()


def test_long_short_is_dollar_neutral_with_half_gross_per_leg():
    px = _panel()
    f = SIGNALS["CARRY"].compute(px)
    p = long_short(px, f, +1, 0, 0.0)
    first_trade = p.loc[p["turn"] > 0, "turn"].iloc[0]                 # monthly: nothing before month-end
    assert first_trade == pytest.approx(1.0)                           # 0.5 long + 0.5 short


def test_vol_target_uses_only_past_volatility():
    rng = np.random.default_rng(1)
    pnl = pd.DataFrame({"gross": rng.normal(0, 0.01, 400), "turn": 0.0},
                       index=pd.bdate_range("2020-01-01", periods=400))
    base = vol_target(pnl, 0.0)
    shocked = pnl.copy()
    shocked.iloc[-1, 0] = 0.5                                          # a huge final-day move
    after = vol_target(shocked, 0.0)
    assert np.allclose(base.iloc[:-1], after.iloc[:-1])
