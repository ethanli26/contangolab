"""Daily portfolio engines for China commodity futures, matching the DolphinDB research lab.

Two portfolio shapes:

  * ``time_series_portfolio``: every liquid contract with a non-zero signal is held in the
    signal's direction, sized by 1/volatility so each position carries similar risk; gross
    exposure 1.
  * ``long_short``: rank contracts into G groups each day; long the top group, short the bottom,
    50% gross per side, equal weight within a leg.

Both rebalance on the last session of each month (``rebalance=0``) or every N trading days, hold
weights in between, charge ``cost_bps`` per side on every unit of weight traded, and earn the
next day's return (a weight set at the close of t earns t -> t+1). ``vol_target`` then scales a
return stream to a constant volatility using only past returns.

Everything runs on the long panel from ``contangolab.signals.library.with_returns``, row-for-row like
the DolphinDB version, so daily P&L can be reconciled exactly.
"""

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252
TIE_DECIMALS = 12   # values equal to 12 decimals are ties (see group_rank)


def calendar(px: pd.DataFrame) -> pd.DataFrame:
    """Add di (global trading-day index) and me (last session of the month) by date."""
    dates = pd.Index(sorted(px["date"].unique()))
    me = pd.Series(dates.month, index=dates)
    flags = pd.DataFrame({"di": np.arange(len(dates)),
                          "me": (me != me.shift(-1)).fillna(True).to_numpy()}, index=dates)
    return px.join(flags, on="date")


def _hold(px: pd.DataFrame, w_raw: pd.Series, active_first: pd.Series, rebalance: int,
          cost_bps: float) -> pd.DataFrame:
    """Set weights on rebalance days, hold them, and aggregate daily gross P&L and turnover."""
    first_di = px.loc[active_first, "di"].min()
    is_reb = px["me"] if rebalance == 0 else ((px["di"] - first_di) % rebalance == 0)
    w = w_raw.where((px["di"] >= first_di) & is_reb)
    w = w.groupby(px["sym"]).ffill().fillna(0.0)
    dw = (w - w.groupby(px["sym"]).shift(1).fillna(0.0)).abs()
    keep = px["di"] >= first_di
    frame = pd.DataFrame({"date": px["date"], "gross": w * px["fwd"].fillna(0.0), "turn": dw})[keep]
    pnl = frame.groupby("date").sum().sort_index()
    pnl["net"] = pnl["gross"] - pnl["turn"] * cost_bps / 1e4
    return pnl.iloc[:-1]                                    # the last day has no next-day return


def time_series_portfolio(px: pd.DataFrame, signal: pd.Series, rebalance: int, cost_bps: float,
                          vol_col: str = "vol60") -> pd.DataFrame:
    """Inverse-volatility time-series portfolio over liquid contracts (see module docstring)."""
    inv_vol = (1.0 / px[vol_col]).where(px[vol_col] > 0)
    active = px["uni"].astype(bool) & signal.notna() & inv_vol.notna() & (signal != 0)
    total = inv_vol.where(active, 0.0).groupby(px["date"]).transform("sum")
    w_raw = (signal * inv_vol / total).where(active & (total > 0), 0.0)
    return _hold(px, w_raw, active, rebalance, cost_bps)


def group_rank(values: pd.Series, dates: pd.Series, groups: int) -> pd.Series:
    """0-based group per date, ascending, matching DolphinDB's ``rank(x, true, groups)``.

    The n valid values are split into ``groups`` near-equal groups with the larger groups first
    (n=7, 5 groups -> sizes 2,2,1,1,1), like ``numpy.array_split``; tied values all take the
    group of their lowest rank.

    Values are rounded to 12 decimals first. Rank-based factors (RANK63) produce genuine exact
    ties, and last-digit floating-point noise would otherwise split them arbitrarily: on the
    real panel that alone moved RANK63's Sharpe from 0.486 to 0.469.
    """
    values = values.round(TIE_DECIMALS)
    rank0 = values.groupby(dates).rank(method="min") - 1.0
    n = values.groupby(dates).transform("count")
    size, extra = n // groups, n % groups
    big = extra * (size + 1)                         # ranks covered by the larger groups
    in_big = rank0 < big
    group = np.where(in_big, rank0 // (size + 1), extra + (rank0 - big) // size.where(size > 0))
    return pd.Series(group, index=values.index).where(values.notna())


def long_short(px: pd.DataFrame, factor: pd.Series, sign: int, rebalance: int, cost_bps: float,
               groups: int = 5) -> pd.DataFrame:
    """Top-minus-bottom group portfolio, 50% gross per side (see module docstring)."""
    f = factor * sign
    q = group_rank(f, px["date"], groups)
    n_long = (q == groups - 1).groupby(px["date"]).transform("sum")
    n_short = (q == 0).groupby(px["date"]).transform("sum")
    ok = (n_long > 0) & (n_short > 0) & q.notna()
    w_raw = pd.Series(0.0, index=px.index)
    w_raw[ok & (q == groups - 1)] = 0.5 / n_long
    w_raw[ok & (q == 0)] = -0.5 / n_short
    return _hold(px, w_raw, f.notna(), rebalance, cost_bps)


def vol_target(pnl: pd.DataFrame, cost_bps: float, target: float = 0.10, window: int = 126,
               cap: float = 2.0) -> pd.Series:
    """Scale to ``target`` annual volatility with leverage from trailing volatility up to t-1,
    capped at ``cap``; changing leverage pays ``cost_bps`` per unit."""
    raw = pnl["gross"] - pnl["turn"] * cost_bps / 1e4
    lev = (target / (raw.rolling(window).std().shift(1) * math.sqrt(TRADING_DAYS))).clip(upper=cap)
    return (lev * raw - lev.diff().abs().fillna(0.0) * cost_bps / 1e4).dropna()


def daily_stats(returns: pd.Series) -> dict:
    """Annualized return, volatility, Sharpe (no risk-free), max drawdown, monthly t-stat."""
    r = returns.dropna()
    nav = (1.0 + r).cumprod()
    monthly = r.groupby(r.index.to_period("M")).apply(lambda x: (1.0 + x).prod() - 1.0)
    return {"start": r.index.min(), "end": r.index.max(), "ann": r.mean() * TRADING_DAYS,
            "vol": r.std() * math.sqrt(TRADING_DAYS), "sharpe": r.mean() / r.std() * math.sqrt(TRADING_DAYS),
            "max_drawdown": float((1.0 - nav / nav.cummax()).max()),
            "t_monthly": monthly.mean() / monthly.std() * math.sqrt(len(monthly)), "months": len(monthly)}
