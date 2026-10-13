"""Curve-study building blocks: the seasonal norm only looks at earlier years, the Newey-West
regression recovers a planted slope, forward volatility starts the day after, and tickers that
mean different things in the two countries stay apart."""

import numpy as np
import pandas as pd
import pytest

from contangolab.markets.curves import TWINS, label, sector
from contangolab.studies.storage import ols_nw, seasonal_norm
from contangolab.studies.volatility import HORIZON, forward_vol


def _weekly(years=6, seed=0):
    idx = pd.date_range("2015-01-02", periods=52 * years, freq="W-FRI")
    season = np.sin(2 * np.pi * idx.isocalendar().week.to_numpy() / 52)
    return pd.Series(season + np.random.default_rng(seed).normal(0, 0.1, len(idx)), index=idx)


def test_seasonal_norm_needs_three_earlier_years_and_ignores_the_future():
    s = _weekly()
    norm = seasonal_norm(s)
    assert norm[s.index.year < 2018].isna().all() and norm[s.index.year >= 2018].notna().all()
    future_shock = s.copy()
    future_shock[future_shock.index.year == 2020] += 100.0
    moved = seasonal_norm(future_shock)
    early = s.index.year <= 2020
    assert np.allclose(norm[early].dropna(), moved[early].dropna())        # 2020 never feeds its own or earlier norms
    assert (moved[s.index.year == 2020] - s[s.index.year == 2020]).abs().max() < 1.0


def test_seasonal_norm_tracks_the_seasonal_shape():
    s = _weekly(years=8)
    dev = (s - seasonal_norm(s)).dropna()
    assert dev.abs().mean() < 0.2 < s.abs().mean()


def test_newey_west_recovers_a_planted_slope():
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    y = 0.5 - 0.3 * x + rng.normal(0, 1, 2000)
    r = ols_nw(y, x)
    assert r["b"] == pytest.approx(-0.3, abs=0.05) and r["t"] < -8 and r["n"] == 2000
    r0 = ols_nw(rng.normal(size=2000), x)
    assert abs(r0["t"]) < 3


def test_forward_volatility_window_starts_the_next_day():
    r = pd.Series(np.random.default_rng(2).normal(0, 0.01, 100))
    fwd = forward_vol(r)
    assert fwd.iloc[10] == pytest.approx(r.iloc[11: 11 + HORIZON].std())   # day 10 sees days 11..31, not itself
    shocked = r.copy()
    shocked.iloc[10] = 5.0                                                  # today's return never enters today's value
    assert forward_vol(shocked).iloc[10] == pytest.approx(fwd.iloc[10])
    assert fwd.iloc[-HORIZON:].isna().all()


def test_tickers_that_collide_across_countries_stay_apart():
    assert sector("cn", "RB") == "Ferrous" and sector("us", "RB") == "Energy"           # rebar vs gasoline
    assert sector("cn", "SI") == "Base metals" and sector("us", "SI") == "Precious metals"
    assert label("cn", "RB") != label("us", "RB")
    assert ("Gold", "AU", "GC") in TWINS and len(TWINS) == 9
