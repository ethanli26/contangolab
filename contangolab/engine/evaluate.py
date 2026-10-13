"""One evaluation for every signal on every dataset (BLUEPRINT 11, P1b protocol).

Monthly, on the last trading day t of each month, for names eligible at t:

  * rank IC: Spearman correlation of (sign x signal at t) with the return from t to the next
    month-end.
  * long-short, net of costs:
      cross-sectional  long the top quantile group, short the bottom (``dataset.groups``),
                       50% gross per side, equal weight within a leg.
      time-series      every eligible asset held in the direction of its own signal, sized
                       by 1/(60-day volatility), gross 1.
    Cost = ``dataset.cost_bps_per_side`` x the sum of absolute weight changes at rebalance.

Reported: mean IC and its t-stat; annualized net return, volatility and Sharpe of the
long-short; the monthly t-stat of net returns (what the multiple-testing correction uses); the
same for each half of the sample and before/after the source's publication year.

LEAKAGE: the signal row at t uses data <= t (each signal's own guarantee); the forward return
t -> next month-end is the label. Known limitation: a stock that delists inside the month has
no next-month price, so it drops out of that month's return (delisting returns are not in the
archive), which flatters short legs of distress-sorted signals slightly.
"""

import math

import numpy as np
import pandas as pd

from contangolab.engine.dataset import Dataset
from contangolab.signals.registry import SignalSpec

MIN_NAMES = 10
VOL_WINDOW = 60
PERIODS = 12


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).last().to_numpy())


def _quantile_groups(values: pd.Series, groups: int) -> pd.Series:
    """Equal-count groups by rank (0 = lowest), ties broken by order for determinism."""
    ranks = values.rank(method="first")
    return np.floor((ranks - 1) * groups / len(values)).astype(int)


def monthly_series(spec: SignalSpec, ds: Dataset, signal: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-month IC, gross, turnover and net long-short return, indexed by the month the return is earned."""
    signal = spec.compute(ds) if signal is None else signal
    signal = signal.replace([np.inf, -np.inf], np.nan) * spec.sign
    close = ds.close
    rebal = month_ends(close.index)
    vol = close.pct_change(fill_method=None).rolling(VOL_WINDOW).std() if spec.kind == "time_series" else None

    rows, prev_w = [], pd.Series(dtype=float)
    for t, nxt in zip(rebal[:-1], rebal[1:]):
        if t not in signal.index:
            continue
        fwd = close.loc[nxt] / close.loc[t] - 1.0
        elig = ds.eligible.loc[t].reindex(signal.columns).fillna(False).astype(bool) if t in ds.eligible.index \
            else pd.Series(False, index=signal.columns)
        f = signal.loc[t][elig].dropna()
        f = f[fwd.reindex(f.index).notna()]
        if len(f) < MIN_NAMES:
            continue
        r = fwd.reindex(f.index)
        ic = f.corr(r, method="spearman")
        if spec.kind == "cross_sectional":
            g = _quantile_groups(f, ds.groups)
            top, bottom = f.index[g == ds.groups - 1], f.index[g == 0]
            w = pd.concat([pd.Series(0.5 / len(top), index=top), pd.Series(-0.5 / len(bottom), index=bottom)])
        else:
            iv = (1.0 / vol.loc[t].reindex(f.index)).replace([np.inf, -np.inf], np.nan)
            active = f[(f != 0) & iv.notna()].index
            if len(active) == 0:
                continue
            w = np.sign(f[active]) * iv[active] / iv[active].sum()
        turnover = float(w.sub(prev_w, fill_value=0.0).abs().sum())
        gross = float((w * r.reindex(w.index)).sum())
        rows.append({"date": nxt, "ic": ic, "gross": gross, "turnover": turnover,
                     "net": gross - turnover * ds.cost_bps_per_side / 1e4, "names": len(f)})
        prev_w = w
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(
        columns=["ic", "gross", "turnover", "net", "names"])


def _block(net: pd.Series, ic: pd.Series) -> dict:
    n = len(net)
    sd = net.std(ddof=1) if n > 1 else np.nan
    ic_sd = ic.std(ddof=1) if len(ic) > 1 else np.nan
    return {
        "months": n,
        "mean_ic": float(ic.mean()) if len(ic) else None,
        "ic_t": float(ic.mean() / ic_sd * math.sqrt(len(ic))) if ic_sd and ic_sd > 0 else None,
        "ann_return": float(net.mean() * PERIODS) if n else None,
        "ann_vol": float(sd * math.sqrt(PERIODS)) if n > 1 else None,
        "sharpe": float(net.mean() / sd * math.sqrt(PERIODS)) if sd and sd > 0 else None,
        "t_stat": float(net.mean() / sd * math.sqrt(n)) if sd and sd > 0 else None,
    }


def summarize(spec: SignalSpec, ds: Dataset, monthly: pd.DataFrame) -> dict:
    net, ic = monthly["net"].astype(float), monthly["ic"].astype(float).dropna()
    out = {"signal": spec.name, "asset_class": ds.asset_class, "family": spec.family, "kind": spec.kind,
           "sign": spec.sign, "published": spec.published, "cost_bps": ds.cost_bps_per_side, "groups": ds.groups,
           "start": str(monthly.index.min().date()) if len(monthly) else None,
           "end": str(monthly.index.max().date()) if len(monthly) else None,
           "avg_turnover": float(monthly["turnover"].mean()) if len(monthly) else None,
           **_block(net, ic)}
    if len(net) >= 4:
        half = net.index[len(net) // 2]
        out["h1"] = _block(net[net.index < half], ic[ic.index < half])
        out["h2"] = _block(net[net.index >= half], ic[ic.index >= half])
    if spec.published is not None and len(net):
        cut = pd.Timestamp(f"{spec.published}-12-31")
        pre, post = net.index <= cut, net.index > cut
        out["pre_publication"] = _block(net[pre], ic[ic.index <= cut])
        out["post_publication"] = _block(net[post], ic[ic.index > cut])
    return out


def evaluate(spec: SignalSpec, ds: Dataset) -> tuple[dict, pd.DataFrame]:
    if ds.asset_class not in spec.asset_classes:
        raise ValueError(f"{spec.name} is not defined for {ds.asset_class}")
    monthly = monthly_series(spec, ds)
    return summarize(spec, ds, monthly), monthly
