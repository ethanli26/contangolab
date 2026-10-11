"""China commodity futures signals, ported from the DolphinDB research lab (~/ddb/factor_lab).

Each signal takes the long-format continuous panel from ``data.futures_cn.build_continuous``
(one row per date x contract, sorted by sym then date) and returns a Series aligned to it.
Rolling windows follow DolphinDB's convention so the two engines reconcile day by day: a
window of n produces a value once n ROWS have passed, computed over whichever values in it
are present (pandas' default instead waits for n non-missing values, one day later here,
since every contract's first return is empty). See ``rolling_rows``.

Two kinds:
  * time-series (``kind = "time_series"``): each contract against its own history, value in
    {-1, 0, +1}; the portfolio goes long or short each contract on its own.
  * cross-sectional (``kind = "cross_sectional"``): ranked across contracts each day; only
    contracts in the liquid universe get a value (others are NaN), as in ddb's
    ``iif(px.uni, factor, NULL)``.

LOOK-AHEAD GUARD: every signal at row t uses that contract's bars up to t only (rolling,
shift and expanding operations), plus same-day cross-sectional ranks.
``tests/test_futures_signals.py`` deletes all bars after a cutoff and checks nothing before it
changes.

These live in their own registry, not the equity factor registry: they need the futures
panel's carry and universe columns, and the equity harness would run them on stock data.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def rolling_rows(values: pd.Series, groups: pd.Series, window: int, how: str) -> pd.Series:
    """Per-group rolling ``how`` (mean/std/max/min) over the last ``window`` rows, DolphinDB-style:
    empty until ``window`` rows exist, then computed over the non-missing values among them."""
    def one(x: pd.Series) -> pd.Series:
        out = getattr(x.rolling(window, min_periods=1), how)()
        out.iloc[: window - 1] = np.nan
        return out
    return values.groupby(groups).transform(one)


def with_returns(px: pd.DataFrame) -> pd.DataFrame:
    """Add ret (close-to-close of the back-adjusted series), fwd (next row's ret) and vol60."""
    px = px.sort_values(["sym", "date"]).reset_index(drop=True)
    g = px.groupby("sym")["close"]
    px["ret"] = g.pct_change(fill_method=None)
    px["fwd"] = px.groupby("sym")["ret"].shift(-1)
    px["vol60"] = rolling_rows(px["ret"], px["sym"], 60, "std") * np.sqrt(TRADING_DAYS)
    return px


def _by_sym(px: pd.DataFrame, col: str, fn: Callable[[pd.Series], pd.Series]) -> pd.Series:
    return px.groupby("sym")[col].transform(fn)


# --- time-series trend ---------------------------------------------------------------------

def ts12(px: pd.DataFrame) -> pd.Series:
    """12-month time-series momentum (Moskowitz, Ooi & Pedersen 2012): sign of the 252-day return."""
    return np.sign(px["close"] / _by_sym(px, "close", lambda c: c.shift(252)) - 1.0)


def ts1(px: pd.DataFrame) -> pd.Series:
    """1-month time-series momentum (China CTA convention): sign of the 21-day return."""
    return np.sign(px["close"] / _by_sym(px, "close", lambda c: c.shift(21)) - 1.0)


def ma20_60(px: pd.DataFrame) -> pd.Series:
    """Moving-average crossover (天风 CTA): sign of the 20-day minus the 60-day average close."""
    fast = rolling_rows(px["close"], px["sym"], 20, "mean")
    slow = rolling_rows(px["close"], px["sym"], 60, "mean")
    return np.sign(fast - slow)


def donch20(px: pd.DataFrame) -> pd.Series:
    """20-day channel breakout (天风 CTA): long above the prior 20-day high, short below the prior
    20-day low, otherwise keep the last signal."""
    hh = rolling_rows(px.groupby("sym")["high"].shift(1), px["sym"], 20, "max")
    ll = rolling_rows(px.groupby("sym")["low"].shift(1), px["sym"], 20, "min")
    raw = pd.Series(np.where(hh.notna() & (px["close"] > hh), 1.0,
                             np.where(ll.notna() & (px["close"] < ll), -1.0, np.nan)), index=px.index)
    return raw.groupby(px["sym"]).ffill()


# --- cross-sectional -------------------------------------------------------------------------

def _in_universe(px: pd.DataFrame, values: pd.Series) -> pd.Series:
    return values.where(px["uni"].astype(bool))


def rank63(px: pd.DataFrame) -> pd.Series:
    """Robust momentum RANK(6,3) (东方 78): the average standardized daily cross-sectional return
    rank over months t-9 to t-3 (126 days, lagged 63), needing at least 63 valid days."""
    ru = px["ret"].where(px["uni"].astype(bool))
    n = ru.groupby(px["date"]).transform("count")
    rank = ru.groupby(px["date"]).rank(method="min")                     # 1-based, ties share the min
    standardized = (rank - (n + 1) / 2.0) / np.sqrt(((n + 1) * (n - 1) / 12.0).where(n > 1))
    lagged = standardized.groupby(px["sym"]).shift(63)
    value = lagged.groupby(px["sym"]).transform(lambda x: x.rolling(126, min_periods=63).mean())
    return _in_universe(px, value)


def carry(px: pd.DataFrame) -> pd.Series:
    """Carry / term structure (华泰 CTA 七; Koijen et al. 2018): annualized log roll yield,
    positive in backwardation."""
    return _in_universe(px, px["ry"])


@dataclass(frozen=True)
class FuturesSignal:
    name: str
    kind: str          # "time_series" or "cross_sectional"
    sign: int          # +1: high value is good. Fixed from the source, never flipped.
    rebalance: int     # 0 = last session of each month, otherwise every N trading days
    source: str
    compute: Callable[[pd.DataFrame], pd.Series]


SIGNALS: dict[str, FuturesSignal] = {s.name: s for s in [
    FuturesSignal("TS12", "time_series", +1, 0, "Moskowitz, Ooi & Pedersen (2012)", ts12),
    FuturesSignal("TS1", "time_series", +1, 0, "China CTA 1-month trend", ts1),
    FuturesSignal("DONCH20", "time_series", +1, 1, "天风 CTA 通道突破", donch20),
    FuturesSignal("MA20_60", "time_series", +1, 1, "天风 CTA 均线交叉", ma20_60),
    FuturesSignal("RANK63", "cross_sectional", +1, 0, "东方 78 稳健动量 RANK(6,3)", rank63),
    FuturesSignal("CARRY", "cross_sectional", +1, 0, "华泰 CTA 七; Koijen et al. (2018)", carry),
]}
