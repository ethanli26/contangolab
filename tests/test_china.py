"""Continuous China-futures contracts (data/futures_cn.py) on hand-built bars.

The real-data check is the engine reconciliation in contangolab/studies/reconcile_dolphindb.py (pandas vs the
DolphinDB pipeline: 157,395 / 157,395 rows identical on 2026-10-10). These tests pin the rules
themselves on cases small enough to check by hand.
"""

import numpy as np
import pandas as pd
import pytest

from contangolab.markets.china import build_continuous, to_factor_data

SPECS = pd.DataFrame({"exchange": ["SHFE", "SHFE", "CFFEX"], "variety": ["RB", "CU", "IF"],
                      "multiplier": [10, 5, 300]})


def _bars(rows):
    return pd.DataFrame(rows, columns=["date", "sym", "open", "high", "low", "close", "volume", "oi"])


def _two_contract_roll():
    """RB2401 is main for three days, then RB2405 takes over on open interest."""
    days = pd.bdate_range("2023-11-01", periods=6)
    near = [100, 101, 102, 103, 104, 105]
    far = [110, 112, 111, 113, 115, 116]
    near_oi = [900, 900, 900, 100, 100, 100]
    far_oi = [100, 100, 100, 900, 900, 900]
    rows = []
    for i, d in enumerate(days):
        rows.append([d, "RB2401", near[i], near[i], near[i], near[i], 50, near_oi[i]])
        rows.append([d, "RB2405", far[i], far[i], far[i], far[i], 50, far_oi[i]])
    return days, _bars(rows)


def test_held_contract_is_yesterdays_main_and_returns_never_jump_contracts():
    days, bars = _two_contract_roll()
    px = build_continuous(bars, SPECS).set_index("date")
    # Day 0 has no "yesterday", so the panel starts on day 1.
    assert px.index[0] == days[1]
    # The main switches on day 3 (OI), so day 3 is still HELD in RB2401 and day 4 in RB2405.
    assert px.loc[days[3], "held"] == "RB202401" and px.loc[days[4], "held"] == "RB202405"
    # Day 4's return is RB2405's own move (113 -> 115), not the 103 -> 115 jump across contracts.
    assert np.isclose(px.loc[days[4], "ret"], 115 / 113 - 1)
    assert np.isclose(px.loc[days[3], "ret"], 103 / 102 - 1)


def test_back_adjusted_close_compounds_exactly_the_held_returns():
    _, bars = _two_contract_roll()
    px = build_continuous(bars, SPECS)
    chain = px["close"] / px["close"].shift(1) - 1
    assert np.allclose(chain.iloc[1:], px["ret"].iloc[1:])
    assert np.isclose(px["close"].iloc[-1], px["rawClose"].iloc[-1])   # anchored to the latest raw price


def test_main_never_rolls_back_to_an_earlier_expiry():
    days, bars = _two_contract_roll()
    bars.loc[(bars["sym"] == "RB2401") & (bars["date"] == days[5]), "oi"] = 5000   # near spikes back
    px = build_continuous(bars, SPECS).set_index("date")
    assert px["held"].iloc[-1] == "RB202405"


def test_carry_is_annualized_log_roll_yield_to_the_next_expiry():
    days, bars = _two_contract_roll()
    px = build_continuous(bars, SPECS).set_index("date")
    # Day 1: held RB2401 at 101, next expiry RB2405 at 112, 4 months apart.
    assert np.isclose(px.loc[days[1], "ry"], np.log(101 / 112) * 12 / 4)


def test_czce_three_digit_codes_get_the_right_decade():
    rows = [[pd.Timestamp("2019-12-02"), "SR001", 1, 1, 1, 5000, 10, 100],
            [pd.Timestamp("2019-12-03"), "SR001", 1, 1, 1, 5010, 10, 100],
            [pd.Timestamp("2019-12-04"), "SR001", 1, 1, 1, 5020, 10, 100]]
    specs = pd.DataFrame({"exchange": ["CZCE"], "variety": ["SR"], "multiplier": [10]})
    px = build_continuous(_bars(rows), specs)
    assert set(px["held"]) == {"SR202001"}


def test_financial_futures_and_unknown_listings_are_excluded():
    days, bars = _two_contract_roll()
    extra = []
    for d in days:
        extra.append([d, "IF2312", 1, 1, 1, 4000, 10, 100])
        extra.append([d, "SC_TAS2312", 1, 1, 1, 1.0, 0, 0])
    px = build_continuous(pd.concat([bars, _bars(extra)]), SPECS)
    assert set(px["sym"]) == {"RB"}


def test_universe_requires_sixty_days_of_history():
    days = pd.bdate_range("2020-01-01", periods=80)
    rows = [[d, "CU2012", 1, 1, 1, 50000 + i, 100, 1000] for i, d in enumerate(days)]
    px = build_continuous(_bars(rows), SPECS)
    assert not px["uni"].iloc[:59].any() and px["uni"].iloc[59:].all()


def test_to_factor_data_pivots_panels_and_mask():
    _, bars = _two_contract_roll()
    data, eligible, extra = to_factor_data(build_continuous(bars, SPECS))
    assert list(data.close.columns) == ["RB"] and eligible.dtypes.eq(bool).all()
    assert {"ry", "farRet", "oi", "notional", "ret"} <= set(extra)


@pytest.mark.parametrize("cut", [2, 3, 4])
def test_deleting_the_future_cannot_change_the_past(cut):
    """Rows up to a cutoff are identical whether or not later bars exist (back-adjust aside)."""
    days, bars = _two_contract_roll()
    full = build_continuous(bars, SPECS).set_index("date")
    part = build_continuous(bars[bars["date"] <= days[cut]], SPECS).set_index("date")
    cols = ["held", "ret", "rawClose", "ry", "uni"]
    pd.testing.assert_frame_equal(full.loc[part.index, cols], part[cols])
