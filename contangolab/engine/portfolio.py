"""Portfolio construction and simulation (BLUEPRINT 11, P3).

Turns a score (one value per eligible asset at a rebalance date) into target weights, then
simulates the book between rebalances.

Construction (``construct``):
  ls_decile / ls_quintile   long the top fraction, short the bottom, 50% gross per side, equal weight
  topN_ew / topN_iv         long-only, the N best names, equal weight or 1/(60-day vol); fully invested
  ts                        every asset in the direction of its score, sized 1/(60-day vol), gross 1
Options:
  buffer     keep a held name until it leaves a band twice as wide (top 20% instead of top 10%, top
             2N instead of top N), then refill from the best new names. Cuts turnover.
  neutral    rank within groups (e.g. sectors) before selecting, so no group dominates the book.
  vol target scale the book so its ex-ante volatility (the target weights applied to the trailing
             63 days of returns) hits the target, capped; set at each rebalance, so its cost is
             part of the turnover.

Simulation (``simulate``): positions are bought at the rebalance close and held, drifting with
prices, until the next rebalance. Portfolio value 1 + sum_i w_i (G_i - 1), where G_i is the asset's
growth since the rebalance, which is exact for a self-financing book. Cost = cost_bps x the sum of
absolute changes from the drifted weights, charged on the first day of the new block. A missing
return (delisted, suspended) counts as 0, the same limitation as ``contangolab.engine.evaluate``.
"""

import math
import re

import numpy as np
import pandas as pd

VOL_WINDOW = 60
VT_WINDOW = 63
TRADING_DAYS = 252


def week_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("W-FRI")).last().to_numpy())


def neutralize(score: pd.Series, groups: pd.Series) -> pd.Series:
    """Centered percentile rank within each group; assets with no group form their own group."""
    g = groups.reindex(score.index).fillna("__none__")
    return score.groupby(g).rank(pct=True) - 0.5


def buffered_top(score: pd.Series, n: int, keep_n: int, held: set) -> list:
    """The n best names, keeping any held name still within the best keep_n (keep_n = n: no buffer)."""
    if n <= 0:
        return []
    ranked = score.sort_values(ascending=False, kind="stable").index
    band = set(ranked[:keep_n])
    keep = [a for a in ranked if a in held and a in band][:n]
    fill = [a for a in ranked if a not in set(keep)][: n - len(keep)]
    return keep + fill


def construct(score: pd.Series, construction: str, vol: pd.Series, state: dict, buffer: bool = False) -> pd.Series:
    """Target weights for one rebalance. ``state`` carries the held names between calls (for the buffer)."""
    score = score.dropna()
    if construction.startswith("ls_"):
        q = {"ls_decile": 0.10, "ls_quintile": 0.20}[construction]
        n = int(math.floor(len(score) * q))
        if n == 0:
            return pd.Series(dtype=float)
        keep_n = 2 * n if buffer else n
        long = buffered_top(score, n, keep_n, state.get("long", set()))
        short = buffered_top(-score, n, keep_n, state.get("short", set()))
        state["long"], state["short"] = set(long), set(short)
        return pd.concat([pd.Series(0.5 / n, index=long), pd.Series(-0.5 / n, index=short)])
    if construction == "ts":
        iv = (1.0 / vol.reindex(score.index)).replace([np.inf, -np.inf], np.nan)
        active = score[(score != 0) & iv.notna()].index
        if len(active) == 0:
            return pd.Series(dtype=float)
        return np.sign(score[active]) * iv[active] / iv[active].sum()
    m = re.fullmatch(r"top(\d+)_(ew|iv)", construction)
    if not m:
        raise ValueError(f"unknown construction {construction!r}")
    n, how = int(m.group(1)), m.group(2)
    if how == "iv":
        score = score[vol.reindex(score.index).notna() & (vol.reindex(score.index) > 0)]
    n = min(n, len(score))
    names = buffered_top(score, n, 2 * n if buffer else n, state.get("long", set()))
    state["long"] = set(names)
    if how == "ew":
        return pd.Series(1.0 / n, index=names) if n else pd.Series(dtype=float)
    iv = 1.0 / vol.reindex(names)
    return iv / iv.sum()


def ex_ante_vol(w: pd.Series, returns: pd.DataFrame, pos: int, window: int = VT_WINDOW) -> float:
    """Annualized vol of today's weights applied to the trailing ``window`` daily returns ending at row pos."""
    if pos + 1 < window or w.empty:
        return float("nan")
    block = returns.iloc[pos + 1 - window: pos + 1].reindex(columns=w.index).fillna(0.0)
    return float((block.to_numpy() @ w.to_numpy()).std(ddof=1) * math.sqrt(TRADING_DAYS))


def vol_scale(w: pd.Series, returns: pd.DataFrame, pos: int, target: float, cap: float) -> pd.Series:
    v = ex_ante_vol(w, returns, pos)
    if not np.isfinite(v) or v <= 0:
        return w
    return w * min(cap, target / v)


def simulate(targets: dict, returns: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Daily net/gross return, trading cost, turnover and end-of-day long/short exposure of a book
    rebalanced to ``targets`` {date: weights}."""
    idx = returns.index
    dates = sorted(targets)
    prev = pd.Series(dtype=float)
    frames = []
    for k, t in enumerate(dates):
        w = targets[t]
        start = idx.searchsorted(t, side="right")
        stop = idx.searchsorted(dates[k + 1], side="right") if k + 1 < len(dates) else len(idx)
        if start >= stop:
            continue
        turnover = float(w.sub(prev, fill_value=0.0).abs().sum())
        block = returns.iloc[start:stop].reindex(columns=w.index).fillna(0.0)
        growth = (1.0 + block).cumprod()
        value = 1.0 + (growth - 1.0).to_numpy() @ w.to_numpy() if len(w) else np.ones(len(block))
        gross = value / np.concatenate([[1.0], value[:-1]]) - 1.0
        cost = np.zeros(len(block))
        cost[0] = turnover * cost_bps / 1e4
        held = growth.to_numpy() / value[:, None]                     # drifted weight per unit of w
        wv = w.to_numpy() if len(w) else np.zeros(0)
        frames.append(pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost,
                                    "turnover": np.where(np.arange(len(block)) == 0, turnover, 0.0),
                                    "long": held @ np.clip(wv, 0, None), "short": held @ np.clip(-wv, 0, None)},
                                   index=block.index))
        prev = (w * growth.iloc[-1] / value[-1]) if len(w) else w
    return pd.concat(frames) if frames else pd.DataFrame(columns=["gross", "cost", "net", "turnover", "long", "short"])


def to_monthly(daily: pd.Series) -> pd.Series:
    """Compound daily returns into calendar months, indexed by each month's last trading day."""
    g = daily.groupby(daily.index.to_period("M"))
    return pd.Series(g.apply(lambda r: (1.0 + r).prod() - 1.0).to_numpy(),
                     index=pd.DatetimeIndex(g.apply(lambda r: r.index[-1]).to_numpy()))
