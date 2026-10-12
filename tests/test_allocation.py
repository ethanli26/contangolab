"""P5: allocation across books and the vol dial."""

import numpy as np
import pandas as pd
import pytest

from core.allocation import allocate, bootstrap_max_dd, by_month, carry_costs, run, weights
from core.portfolio import simulate

IDX = pd.date_range("2005-01-31", periods=180, freq="ME")


def _books(seed=0, rho=0.0):
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=len(IDX))
    return pd.DataFrame({"a": 0.006 + 0.025 * z[:, 0], "b": 0.004 + 0.015 * z[:, 1]}, index=IDX)


def test_carry_costs_charge_borrow_on_shorts_and_spread_on_longs():
    d = pd.DataFrame({"net": [0.0], "long": [0.5], "short": [0.5]})
    out = carry_costs(borrow_bps=50, financing_bps=50)(d)
    assert out["net"].iloc[0] == pytest.approx(-(0.5 * 50 + 0.5 * 50) / 1e4 / 252)


def test_simulate_reports_long_and_short_exposure():
    days = pd.bdate_range("2020-01-01", periods=4)
    r = pd.DataFrame({"A": [0.0, 0.10, 0.0, 0.0], "B": [0.0, 0.0, 0.0, 0.0]}, index=days)
    out = simulate({days[0]: pd.Series({"A": 0.5, "B": -0.5})}, r, cost_bps=0.0)
    assert out["long"].iloc[0] == pytest.approx(0.55 / 1.05) and out["short"].iloc[0] == pytest.approx(0.5 / 1.05)


def test_risk_parity_is_inverse_vol_and_max_sharpe_matches_the_uncorrelated_solution():
    books = _books()
    w = weights(books, "risk_parity")
    vol = books.std().to_numpy()
    assert np.allclose(w, (1 / vol) / (1 / vol).sum())
    diag = pd.DataFrame({"a": [0.02, -0.01, 0.03, 0.0], "b": [0.01, 0.01, -0.01, 0.01]})
    w_ms = weights(diag, "max_sharpe")
    cov = diag.cov().to_numpy()
    shrunk = 0.5 * cov + 0.5 * np.diag(np.diag(cov))
    raw = np.clip(np.linalg.solve(shrunk, diag.mean().to_numpy()), 0, None)
    assert np.allclose(w_ms, raw / raw.sum())


def test_allocation_uses_only_earlier_months():
    books = _books()
    base = allocate(books, "max_sharpe", target=0.15)
    k = 60
    changed = books.copy()
    changed.iloc[k:] *= 5
    moved = allocate(changed, "max_sharpe", target=0.15)
    t = books.index[k]
    cols = ["lev", "w_a", "w_b"]
    assert np.allclose(base.loc[:t, cols], moved.loc[:t, cols])         # month k's weights ignore month k
    assert not np.isclose(base.loc[t, "ret"], moved.loc[t, "ret"])


def test_vol_dial_hits_the_target_and_respects_the_cap():
    out = allocate(_books(rho=0.2), "risk_parity", target=0.15)
    realized = out["ret"].std() * np.sqrt(12)
    assert realized == pytest.approx(0.15, rel=0.2) and out["lev"].max() <= 4.0


def test_bootstrap_drawdown_grows_with_leverage_and_is_deterministic():
    r = _books()["a"]
    assert bootstrap_max_dd(r) == bootstrap_max_dd(r)
    assert bootstrap_max_dd(2 * r) > bootstrap_max_dd(r) > 0


def test_by_month_aligns_different_exchange_calendars():
    us = pd.Series([0.01], index=[pd.Timestamp("2024-02-29")])
    cn = pd.Series([0.02], index=[pd.Timestamp("2024-02-28")])
    assert by_month(us).index.equals(by_month(cn).index)


def test_run_applies_both_bars():
    b = _books(rho=0.0)
    out = run({"a": b["a"], "b": b["b"]})
    assert out["adopted"] in ("risk_parity", "max_sharpe") and set(out["dial"]) == {0.10, 0.15, 0.20, 0.25}
    assert isinstance(out["bar1"], bool) and abs(out["corr"]) < 0.3
