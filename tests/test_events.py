"""E1 event signals: builders on hand-made Sharadar-shaped tables, timing, and the calendar-time engine."""

import numpy as np
import pandas as pd
import pytest

from core.dataset import Dataset
from core.events import classify, double_surprise, leg_targets, only_eligible, run_signal, universe_targets
from core.portfolio import simulate
from data.us_events import dividend_initiations, ear, insider_clusters, sue
from factors.base import FactorData

DAYS = pd.bdate_range("2015-01-01", periods=400)


def _ds(close: pd.DataFrame) -> Dataset:
    data = FactorData(open=close, high=close, low=close, close=close, volume=close * 0 + 1e6)
    return Dataset("us_equity", data, pd.DataFrame(True, index=close.index, columns=close.columns), 15.0, 10)


def test_sue_uses_the_same_quarter_last_year_and_only_earlier_changes_for_scale(tmp_path):
    quarters = pd.date_range("2010-03-31", periods=16, freq="QE")
    eps = [1.0, 1.0, 1.0, 1.0, 1.1, 1.0, 1.2, 0.9, 1.2, 1.1, 1.1, 1.1, 1.3, 1.0, 1.4, 2.0]
    pd.DataFrame({"ticker": "X", "dimension": "ARQ", "calendardate": quarters,
                  "datekey": quarters + pd.Timedelta(days=40), "eps": eps}).to_parquet(tmp_path / "SF1.parquet")
    out = sue(bulk_dir=tmp_path).set_index("quarter")
    changes = pd.Series(eps, index=quarters).diff(4)
    expected = changes.iloc[15] / changes.iloc[7:15].std()                # the 8 changes before the last one
    assert out.loc[quarters[15], "value"] == pytest.approx(expected)
    assert out.loc[quarters[15], "avail"] == quarters[15] + pd.Timedelta(days=40)
    assert quarters[9] not in out.index                                   # only 5 earlier changes: < 6


def test_ear_is_stock_minus_universe_over_days_minus_one_to_plus_one():
    close = pd.DataFrame(100.0, index=DAYS[:10], columns=["A", "B"])
    close.loc[DAYS[4]:, "A"] = 110.0                                      # +10% on day 0 (DAYS[4])
    close.loc[DAYS[5]:, "B"] = 105.0                                      # +5% on day +1
    out = ear(pd.DataFrame({"ticker": ["A"], "date": [DAYS[4]]}), close, pd.DataFrame(True, index=DAYS[:10],
                                                                                      columns=["A", "B"]))
    universe = (1.05 * 1.025) - 1                                         # EW: day0 +5%, day+1 +2.5%
    assert out.loc[0, "value"] == pytest.approx(0.10 - universe)
    assert out.loc[0, "avail"] == DAYS[5]


def test_insider_cluster_needs_two_insiders_and_is_not_repeated(tmp_path):
    rows = [("X", "2020-01-02", "P", "Y", "N", "Alice"), ("X", "2020-01-20", "P", "N", "Y", "Bob"),
            ("X", "2020-01-25", "P", "Y", "N", "Carol"),                   # same cluster, no new event
            ("Y", "2020-01-02", "P", "N", "N", "Fund"), ("Y", "2020-01-03", "P", "N", "N", "Fund2"),   # 10% holders
            ("Z", "2020-01-02", "P", "Y", "N", "Dan"), ("Z", "2020-03-15", "P", "Y", "N", "Eve")]     # too far apart
    pd.DataFrame(rows, columns=["ticker", "filingdate", "transactioncode", "isofficer", "isdirector", "ownername"]
                 ).to_parquet(tmp_path / "SF2.parquet")
    out = insider_clusters(bulk_dir=tmp_path)
    assert out[["ticker", "avail"]].values.tolist() == [["X", pd.Timestamp("2020-01-20")]]


def test_dividend_initiation_needs_a_long_gap_and_a_seasoned_stock(tmp_path):
    close = pd.DataFrame(1.0, index=pd.bdate_range("2010-01-01", "2016-12-31"), columns=["A", "B"])
    close.loc[:"2013-12-31", "B"] = np.nan                                # B listed 2014
    rows = [("2012-06-01", "dividend", "A"), ("2012-09-01", "dividend", "A"),   # A initiates 2012
            ("2016-01-04", "dividend", "A"),                              # > 730 days later: initiates again
            ("2014-06-02", "dividend", "B")]                              # B too new
    pd.DataFrame(rows, columns=["date", "action", "ticker"]).to_parquet(tmp_path / "ACTIONS.parquet")
    out = dividend_initiations(close, bulk_dir=tmp_path)
    assert out["avail"].tolist() == [pd.Timestamp("2012-06-01"), pd.Timestamp("2016-01-04")]


def test_breakpoints_come_from_the_previous_quarter():
    ev = pd.DataFrame({"ticker": "A", "avail": list(pd.date_range("2020-01-02", periods=10, freq="D")) +
                       [pd.Timestamp("2020-04-10"), pd.Timestamp("2020-04-11")],
                       "value": list(range(10)) + [8.5, 100.0]})
    out = classify(ev, 0.1)
    assert len(out) == 2                                                  # Q1 has no previous quarter
    hi = np.quantile(range(10), 0.9)
    assert out["top"].tolist() == [8.5 >= hi, True]


def test_positions_enter_strictly_after_the_event_and_last_hold_days():
    close = pd.DataFrame(100.0, index=DAYS[:100], columns=["A", "B"])
    ds = _ds(close)
    ev = only_eligible(pd.DataFrame({"ticker": ["A"], "avail": [DAYS[5]], "value": [1.0]}), ds)
    grid = pd.DatetimeIndex([DAYS[5], DAYS[6], DAYS[68], DAYS[69]])
    t = leg_targets(ev, ds, grid, hold=63)
    assert [len(t[d]) for d in grid] == [0, 1, 1, 0]


def test_double_surprise_takes_the_later_date():
    sue_q = pd.DataFrame({"ticker": ["A", "B"], "avail": pd.to_datetime(["2020-05-10", "2020-05-10"]),
                          "top": [True, True]})
    ear_q = pd.DataFrame({"ticker": ["A", "B"], "event_date": pd.to_datetime(["2020-04-28", "2020-01-01"]),
                          "avail": pd.to_datetime(["2020-04-29", "2020-01-02"]), "top": [True, True]})
    out = double_surprise(sue_q, ear_q)
    assert out[["ticker", "avail"]].values.tolist() == [["A", pd.Timestamp("2020-05-10")]]   # B's EAR too old


def test_a_planted_drift_after_events_shows_up_as_excess_return():
    rng = np.random.default_rng(0)
    names = [f"S{i}" for i in range(60)]
    r = pd.DataFrame(rng.normal(0, 0.01, (len(DAYS), 60)), index=DAYS, columns=names)
    events = []
    for k in range(0, 330, 10):                                           # every 10 days one stock gets news
        name = names[k % 60]
        r.iloc[k + 1: k + 41, names.index(name)] += 0.002                 # +0.2%/day drift for 40 days after
        events.append((name, DAYS[k], 1.0))
    close = 100 * (1 + r).cumprod()
    ds = _ds(close)
    ev = only_eligible(pd.DataFrame(events, columns=["ticker", "avail", "value"]), ds)
    grid = pd.DatetimeIndex(pd.Series(DAYS, index=DAYS).groupby(DAYS.to_period("W-FRI")).last().to_numpy())
    bench = simulate(universe_targets(ds, grid), close.pct_change(fill_method=None), 0.0)["net"]
    out = run_signal("planted", ev, ds, grid, bench, published=None)["result"]
    assert out["ann_return"] > 0.10 and out["t_stat"] > 2


def test_universe_benchmark_holds_cash_before_any_name_is_eligible():
    close = pd.DataFrame(100.0, index=DAYS[:20], columns=["A", "B"])
    ds = _ds(close)
    ds.eligible.iloc[:10] = False                                         # no stock has enough history yet
    targets = universe_targets(ds, pd.DatetimeIndex([DAYS[5], DAYS[15]]))
    assert targets[DAYS[5]].empty and targets[DAYS[15]].sum() == pytest.approx(1.0)
