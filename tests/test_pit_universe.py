"""Point-in-time universe membership: who is investable on date t uses only data <= t.

The old universe ranked every ticker on its FULL-HISTORY median dollar volume, so a stock
that was tiny in 2000 and enormous by 2020 was "in the universe" in 2000 because of what it
later became. These tests pin down the replacement:

  * the bias itself, on a two-stock example where the static ranking picks the future winner,
  * the mechanical no-look-ahead proof (delete the future, membership must not change),
  * delisting and minimum-history rules.
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from backtest.universe import membership_to_daily, point_in_time_membership

HISTORY = 20   # small minimum-history floor so the synthetic panels stay short


def _long(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """{ticker: DataFrame(date, close, volume)} -> long-format SEP-like frame."""
    return pd.concat([f.assign(ticker=t) for t, f in frames.items()], ignore_index=True)


def _bars(dates, close, volume) -> pd.DataFrame:
    return pd.DataFrame({"date": dates, "close": close, "volume": volume})


def test_static_ranking_picks_the_future_winner_and_pit_does_not():
    """A late bloomer must not be in the universe before it became liquid."""
    dates = pd.bdate_range("2000-01-03", "2003-12-31")
    n, half = len(dates), len(dates) // 2
    steady = _bars(dates, np.full(n, 50.0), np.full(n, 1_000_000.0))           # $50M/day always
    bloomer_volume = np.r_[np.full(half, 10_000.0), np.full(n - half, 100_000_000.0)]
    bloomer = _bars(dates, np.full(n, 50.0), bloomer_volume)                    # $0.5M -> $5B/day
    sep = _long({"STEADY": steady, "BLOOM": bloomer})

    # The static rule: rank on whole-life median dollar volume. With top_n=1 it picks BLOOM.
    whole_life = sep.assign(dv=sep.close * sep.volume).groupby("ticker")["dv"].median()
    assert whole_life.idxmax() == "BLOOM"

    members = point_in_time_membership(sep, top_n=1, min_history_bars=HISTORY)
    early = members[members.index < dates[half]]
    late = members[members.index >= dates[half] + pd.DateOffset(months=4)]
    assert early["STEADY"].iloc[1:].all() and not early["BLOOM"].any()
    assert late["BLOOM"].all() and not late["STEADY"].any()


def test_delisted_name_leaves_at_its_last_month_end():
    """No membership on a month-end the stock no longer trades on."""
    dates = pd.bdate_range("2001-01-01", "2002-12-31")
    alive = dates[dates <= "2001-09-14"]                                       # delists mid-September
    sep = _long({
        "DEAD": _bars(alive, np.full(len(alive), 20.0), np.full(len(alive), 9e6)),
        "LIVE": _bars(dates, np.full(len(dates), 20.0), np.full(len(dates), 1e6)),
    })
    members = point_in_time_membership(sep, top_n=1, min_history_bars=HISTORY)
    assert members.loc["2001-08-31", "DEAD"]
    after = members[members.index >= "2001-09-01"]
    assert not after["DEAD"].any() and after["LIVE"].all()


def test_minimum_history_is_counted_as_of_the_date():
    dates = pd.bdate_range("2001-01-01", "2001-12-31")
    sep = _long({"NEW": _bars(dates, np.full(len(dates), 30.0), np.full(len(dates), 1e6))})
    members = point_in_time_membership(sep, top_n=5, min_history_bars=100)
    first = members.index[members["NEW"]][0]
    assert (dates <= first).sum() >= 100
    assert not members.loc[members.index < first, "NEW"].any()


def test_membership_carries_forward_to_daily_until_next_month_end():
    idx = pd.DatetimeIndex(["2001-01-31", "2001-02-28"])
    members = pd.DataFrame({"A": [True, False]}, index=idx)
    daily = membership_to_daily(members, pd.bdate_range("2001-01-29", "2001-03-02"))
    assert not daily.loc["2001-01-30", "A"]          # before the first decision
    assert daily.loc["2001-01-31":"2001-02-27", "A"].all()
    assert not daily.loc["2001-02-28":, "A"].any()


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), n_tickers=st.integers(3, 12), top_n=st.integers(1, 5),
       cut=st.floats(0.3, 0.9))
def test_deleting_the_future_cannot_change_membership(seed, n_tickers, top_n, cut):
    """THE PROOF: membership on or before a cutoff is identical with all later bars deleted."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2005-01-03", periods=400)
    frames = {}
    for i in range(n_tickers):
        start, stop = sorted(rng.integers(0, len(dates), size=2))   # random listing / delisting
        d = dates[start:stop + 1]
        if len(d) == 0:
            continue
        close = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, len(d))))
        volume = rng.lognormal(13, 1.0, len(d)) * rng.uniform(0.2, 5.0)
        frames[f"T{i}"] = _bars(d, close, volume)
    sep = _long(frames)
    cutoff = dates[int(len(dates) * cut)]

    full = point_in_time_membership(sep, top_n=top_n, min_history_bars=HISTORY)
    truncated = point_in_time_membership(sep[sep["date"] <= cutoff], top_n=top_n,
                                         min_history_bars=HISTORY)
    # Month-ends strictly before the cutoff's month are fully decided by then.
    decided = full.index[full.index < cutoff.to_period("M").start_time]
    a = full.reindex(index=decided).fillna(False)
    b = truncated.reindex(index=decided, columns=full.columns).fillna(False)
    pd.testing.assert_frame_equal(a.astype(bool), b.astype(bool), check_freq=False)
