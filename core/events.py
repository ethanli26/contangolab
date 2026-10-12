"""Event-driven signals as calendar-time portfolios (BLUEPRINT 11, E1, protocol ``b11-e1-v1``).

Pre-registered in the vault before any code or run. Each week (last trading day), hold equal-weight
every eligible stock whose qualifying event became public within the last ``HOLD_DAYS`` trading
days; enter at the first rebalance strictly after the event's ``avail`` day (an SEC filing accepted
after the close still carries that day's date). Cost and accounting are
``core.portfolio.simulate``'s. The result is the long leg's net return minus the equal-weight
point-in-time universe (weekly, no cost), compounded by month.

Continuous signals (SUE, EAR) qualify by decile; the cut-offs come from the PREVIOUS calendar
quarter's events, so they are known before any event they classify.
"""

import numpy as np
import pandas as pd

from core.dataset import Dataset
from core.design import halves, stats
from core.portfolio import simulate, to_monthly, week_ends

PROTOCOL = "b11-e1-v1"
FAMILY = "us_events"
HOLD_DAYS = 63
CAR_DAYS = 63


def only_eligible(events: pd.DataFrame, ds: Dataset) -> pd.DataFrame:
    """Events whose stock is in the point-in-time universe on the (first trading) day it became public."""
    idx = ds.close.index
    ev = events[events["ticker"].isin(set(ds.close.columns))].copy()
    pos = idx.searchsorted(ev["avail"].to_numpy(), side="left")
    ev = ev[pos < len(idx)].assign(pos=pos[pos < len(idx)])
    elig = ds.eligible.reindex_like(ds.close).fillna(False)
    col = {t: i for i, t in enumerate(ds.close.columns)}
    ok = elig.to_numpy(dtype=bool)[ev["pos"].to_numpy(), ev["ticker"].map(col).to_numpy()]
    return ev[ok].reset_index(drop=True)


def classify(events: pd.DataFrame, q: float) -> pd.DataFrame:
    """Add ``top`` / ``bottom`` flags using the previous calendar quarter's q and 1-q quantiles."""
    ev = events.copy()
    ev["qtr"] = ev["avail"].dt.to_period("Q")
    cuts = ev.groupby("qtr")["value"].quantile([q, 1 - q]).unstack()
    cuts.columns = ["lo", "hi"]
    cuts.index = cuts.index + 1                       # quarter k's cut-offs are applied in quarter k+1
    ev = ev.join(cuts, on="qtr")
    ev["top"] = ev["value"] >= ev["hi"]
    ev["bottom"] = ev["value"] <= ev["lo"]
    return ev[ev["hi"].notna()].reset_index(drop=True)


def leg_targets(events: pd.DataFrame, ds: Dataset, grid: pd.DatetimeIndex, hold: int = HOLD_DAYS) -> dict:
    """{rebalance date: equal weights} over names with an event public within the last ``hold`` trading days."""
    idx = ds.close.index
    ev = events.sort_values("pos")
    pos, tick = ev["pos"].to_numpy(), ev["ticker"].to_numpy()
    elig = ds.eligible.reindex_like(ds.close).fillna(False)
    targets = {}
    for t in grid:
        p = idx.get_loc(t)
        # strictly after the event day (a filing accepted after the close still carries that date),
        # and at most ``hold`` trading days after it
        lo, hi = np.searchsorted(pos, p - hold, side="left"), np.searchsorted(pos, p, side="left")
        names = pd.Index(pd.unique(tick[lo:hi]))
        names = names[elig.loc[t].reindex(names).fillna(False).to_numpy(dtype=bool)]
        targets[t] = pd.Series(1.0 / len(names), index=names) if len(names) else pd.Series(dtype=float)
    return targets


def universe_targets(ds: Dataset, grid: pd.DatetimeIndex) -> dict:
    elig = ds.eligible.reindex_like(ds.close).fillna(False)
    out = {}
    for t in grid:
        names = elig.columns[elig.loc[t].to_numpy(dtype=bool) & ds.close.loc[t].notna().to_numpy()]
        out[t] = pd.Series(1.0 / len(names), index=names) if len(names) else pd.Series(dtype=float)
    return out


def car_paths(events: pd.DataFrame, ds: Dataset, bench_level: pd.Series, days: int = CAR_DAYS) -> np.ndarray:
    """Cumulative abnormal return (stock minus universe) for days 1..days after each event's day."""
    c = ds.close.to_numpy()
    b = bench_level.reindex(ds.close.index).to_numpy()
    col = {t: i for i, t in enumerate(ds.close.columns)}
    p = events["pos"].to_numpy()
    j = events["ticker"].map(col).to_numpy()
    keep = p + days < len(c)
    p, j = p[keep], j[keep]
    steps = np.arange(1, days + 1)
    stock = c[p[:, None] + steps, j[:, None]] / c[p, j][:, None] - 1.0
    bench = b[p[:, None] + steps] / b[p][:, None] - 1.0
    return stock - bench


def run_signal(name: str, long: pd.DataFrame, ds: Dataset, grid: pd.DatetimeIndex, bench_daily: pd.Series,
               published: int | None, short: pd.DataFrame | None = None) -> dict:
    """Simulate the long leg (and short leg if given); excess over the universe; summary dict for the registry."""
    lt = leg_targets(long, ds, grid)
    first = next((t for t, w in lt.items() if len(w)), None)
    lt = {t: w for t, w in lt.items() if first is not None and t >= first}
    daily = simulate(lt, ds.close.pct_change(fill_method=None), ds.cost_bps_per_side)
    excess = (daily["net"] - bench_daily.reindex(daily.index)).dropna()
    monthly = to_monthly(excess)
    window = monthly.index
    h1, h2 = halves(monthly, window)
    res = {"signal": name, "asset_class": "us_equity", "family": FAMILY, "kind": "event_long", "sign": 1,
           "published": published, "cost_bps": ds.cost_bps_per_side, "start": str(window[0].date()),
           "end": str(window[-1].date()), "avg_turnover": float(daily["turnover"].sum() / max(1, len(monthly))),
           "avg_names": float(np.mean([len(w) for w in lt.values()])), "events": int(len(long)),
           **stats(monthly), "h1": h1, "h2": h2,
           "long_abs": stats(to_monthly(daily["net"]))}
    if published is not None:
        cut = pd.Timestamp(f"{published}-12-31")
        res["pre_publication"] = stats(monthly[monthly.index <= cut])
        res["post_publication"] = stats(monthly[monthly.index > cut])
    if short is not None:
        st = {t: w for t, w in leg_targets(short, ds, grid).items() if t >= first}
        sd = simulate(st, ds.close.pct_change(fill_method=None), ds.cost_bps_per_side)
        spread = (daily["net"] - sd["net"].reindex(daily.index).fillna(0.0)) * 0.5
        res["long_short"] = stats(to_monthly(spread))
        res["short_excess"] = stats(to_monthly((sd["net"] - bench_daily.reindex(sd.index)).dropna()))
    return {"result": res, "monthly_excess": monthly}


def double_surprise(sue_q: pd.DataFrame, ear_q: pd.DataFrame, max_gap_days: int = 60) -> pd.DataFrame:
    """Top-quintile SUE whose earnings release (within the prior ``max_gap_days``) was also a top-quintile EAR.

    The 8-K release comes first and the 10-Q/10-K filing carrying the SUE later, so the event becomes
    public at the later of the two dates.
    """
    s = sue_q[sue_q["top"]].sort_values("avail")
    e = ear_q[ear_q["top"]].sort_values("event_date")
    m = pd.merge_asof(s, e[["ticker", "event_date", "avail"]].rename(columns={"avail": "ear_avail"}),
                      left_on="avail", right_on="event_date", by="ticker",
                      tolerance=pd.Timedelta(days=max_gap_days), direction="backward")
    m = m[m["event_date"].notna()]
    return pd.DataFrame({"ticker": m["ticker"], "avail": np.maximum(m["avail"], m["ear_avail"]),
                         "value": 1.0}).reset_index(drop=True)


def weekly_grid(ds: Dataset) -> pd.DatetimeIndex:
    return week_ends(ds.close.index)
