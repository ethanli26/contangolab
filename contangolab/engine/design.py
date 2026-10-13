"""Design-space search, judged by an out-of-sample selector (BLUEPRINT 11, P3, protocol ``b11-p3-v1``).

A configuration is one choice per layer: score source x construction x neutralization x turnover
control x vol target. Each track's grid was written down in the vault before any run, every
configuration is run and logged as a trial, and the honest result is the walk-forward
**selector**: every 12 months it switches to the configuration with the best net Sharpe so far
(at least 36 months of that configuration's history), and its out-of-sample months are what count.

Score sources (all walk-forward, built on the P2 features in ``contangolab.engine.combine``):
  best_trailing   the single signal whose own P1b portfolio has the best expanding net Sharpe
  survivors_ew    mean of the signals whose expanding one-sided net t-stat is > 2 (none: best_trailing)
  all_ew          mean of every signal (P2's ew)
  xgb             P2's walk-forward XGBoost score (monthly; carried to weekly dates)

Bar (pre-registered): adopt the selector's process only if it beats the track's simple default
in BOTH halves of its out-of-sample window and the grid's PBO is <= 0.5; otherwise adopt the
simple default. The simple default is all_ew with the plainest construction and nothing added.
Switching configurations is charged a full round trip (2 x cost per side), which is conservative.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from contangolab.engine.combine import COMBO_KIND, Panels, build_panels, walk_forward
from contangolab.engine.dataset import Dataset
from contangolab.engine.evaluate import MIN_NAMES, monthly_series
from contangolab.engine.portfolio import VOL_WINDOW, construct, neutralize, simulate, to_monthly, vol_scale, week_ends
from contangolab.signals.registry import SignalSpec

PROTOCOL = "b11-p3-v1"
SOURCES = ("best_trailing", "survivors_ew", "all_ew", "xgb")
MIN_HISTORY = 36
SURVIVOR_T = 2.0
SELECT_EVERY = 12
PBO_LIMIT = 0.5


@dataclass(frozen=True)
class Track:
    name: str
    asset_class: str
    constructions: tuple[str, ...]
    neutral: tuple[bool, ...]
    turnover: tuple[str, ...]           # "none" / "buffer" (US) or "monthly" / "weekly" rebalance (futures)
    vt_target: float
    vt_cap: float
    simple_construction: str


# The equity tracks of the same protocol (us_ls, us_lo) are in the trial log; their code is in the
# archived pit-backtester repository.
TRACKS = {
    "fut": Track("fut", "cn_futures", ("ts", "ls_quintile"), (False,), ("monthly", "weekly"), 0.10, 3.0, "ts"),
}


@dataclass(frozen=True)
class Config:
    track: str
    source: str
    construction: str
    neutral: bool
    turnover: str
    vt: bool

    @property
    def name(self) -> str:
        return "|".join([self.track, self.source, self.construction, "sector" if self.neutral else "raw",
                         self.turnover, "vt" if self.vt else "novt"])


def grid(track: Track) -> list[Config]:
    return [Config(track.name, s, c, n, t, v) for s in SOURCES for c in track.constructions for n in track.neutral
            for t in track.turnover for v in (False, True)]


def simple_default(track: Track) -> Config:
    return Config(track.name, "all_ew", track.simple_construction, False, track.turnover[0], False)


# --- shared inputs per asset class ------------------------------------------------------------

@dataclass
class Context:
    ds: Dataset
    specs: list[SignalSpec]
    panels: dict[str, Panels]           # "monthly" and, for futures, "weekly"
    xgb: pd.DataFrame                   # month-end x asset
    singles: pd.DataFrame               # month x signal: each signal's own net P1b long-short return
    returns: pd.DataFrame               # daily
    vol: pd.DataFrame                   # daily 60-day vol
    sectors: pd.Series | None


def prepare(ds: Dataset, specs: list[SignalSpec], weekly: bool = False, sectors: pd.Series | None = None,
            min_train: int = MIN_HISTORY) -> Context:
    kind = COMBO_KIND.get(ds.asset_class, "cross_sectional")
    monthly = build_panels(specs, ds, kind)
    panels = {"monthly": monthly}
    if weekly:
        panels["weekly"] = build_panels(specs, ds, kind, rebal=week_ends(ds.close.index))
    xgb = walk_forward(monthly, methods=("xgb",), min_train=min_train)[0]["xgb"]
    singles = pd.DataFrame({s.name: monthly_series(s, ds, signal=monthly.raw[s.name])["net"] for s in specs})
    returns = ds.close.pct_change(fill_method=None)
    return Context(ds, specs, panels, xgb, singles, returns, returns.rolling(VOL_WINDOW).std(), sectors)


def source_weights(source: str, singles: pd.DataFrame, t: pd.Timestamp, min_history: int = MIN_HISTORY):
    """Signal weights at t from realized single-signal results up to t; None if not yet defined."""
    k = singles.shape[1]
    if source == "all_ew":
        return np.full(k, 1.0 / k)
    hist = singles.loc[singles.index <= t]
    n, mean, sd = hist.count(), hist.mean(), hist.std(ddof=1)
    ok = (n >= min_history) & (sd > 0)
    if not ok.any():
        return None
    sharpe = (mean / sd)[ok]
    if source == "survivors_ew":
        tstat = (mean / sd * np.sqrt(n))[ok]
        chosen = tstat[tstat > SURVIVOR_T].index
        if len(chosen):
            return singles.columns.isin(chosen).astype(float) / len(chosen)
    return (singles.columns == sharpe.idxmax()).astype(float)


def scores_on_grid(ctx: Context, source: str, grid_name: str) -> dict:
    """{date: score Series over eligible assets} for one source on the monthly or weekly grid."""
    p = ctx.panels[grid_name]
    out = {}
    for i, t in enumerate(p.dates):
        if source == "xgb":
            rows = ctx.xgb.index[ctx.xgb.index <= t]
            if len(rows) == 0:
                continue
            s = ctx.xgb.loc[rows[-1]].reindex(p.assets[i]).dropna()
        else:
            w = source_weights(source, ctx.singles, t)
            if w is None:
                continue
            s = pd.Series(p.X[i] @ w, index=p.assets[i])
        if len(s) >= MIN_NAMES:
            out[t] = s
    return out


def run_config(ctx: Context, cfg: Config, track: Track, scores: dict) -> pd.DataFrame:
    """Daily gross / cost / net / turnover of one configuration."""
    state, targets = {}, {}
    for t, s in scores.items():
        if cfg.neutral and ctx.sectors is not None:
            s = neutralize(s, ctx.sectors)
        w = construct(s, cfg.construction, ctx.vol.loc[t], state, buffer=cfg.turnover == "buffer")
        if cfg.vt:
            w = vol_scale(w, ctx.returns, ctx.returns.index.get_loc(t), track.vt_target, track.vt_cap)
        targets[t] = w
    return simulate(targets, ctx.returns, ctx.ds.cost_bps_per_side)


# --- statistics, selector, verdict ------------------------------------------------------------

def stats(monthly: pd.Series) -> dict:
    r = monthly.dropna()
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return {"months": n, "sharpe": None, "ann_return": None, "ann_vol": None, "cagr": None, "max_dd": None,
                "t_stat": None}
    eq = (1.0 + r).cumprod()
    return {"months": n, "sharpe": float(r.mean() / r.std(ddof=1) * math.sqrt(12)),
            "ann_return": float(r.mean() * 12), "ann_vol": float(r.std(ddof=1) * math.sqrt(12)),
            "cagr": float(eq.iloc[-1] ** (12 / n) - 1), "max_dd": float((eq / eq.cummax() - 1).min()),
            "t_stat": float(r.mean() / r.std(ddof=1) * math.sqrt(n))}


def halves(monthly: pd.Series, window: pd.DatetimeIndex) -> tuple[dict, dict]:
    cut = window[len(window) // 2]
    r = monthly.reindex(window)
    return stats(r[r.index < cut]), stats(r[r.index >= cut])


def select(matrix: pd.DataFrame, switch_cost: float, min_history: int = MIN_HISTORY,
           every: int = SELECT_EVERY) -> tuple[pd.Series, list]:
    """Walk-forward configuration selector over a month x config return matrix."""
    counts = matrix.notna().cumsum()
    ready = np.flatnonzero((counts >= min_history).any(axis=1).to_numpy())
    if len(ready) == 0:
        return pd.Series(dtype=float), []
    rets, picks, prev = [], [], None
    for k in range(ready[0], len(matrix) - 1, every):
        hist = matrix.iloc[: k + 1]
        ok = hist.count() >= min_history
        sharpe = (hist.mean() / hist.std(ddof=1))[ok].dropna()
        if sharpe.empty:
            continue
        chosen = sharpe.idxmax()
        seg = matrix[chosen].iloc[k + 1: k + 1 + every].fillna(0.0).copy()
        if chosen != prev and len(seg):
            seg.iloc[0] -= switch_cost
        rets.append(seg)
        picks.append((matrix.index[k], chosen))
        prev = chosen
    return pd.concat(rets), picks


def verdict(sel_h: tuple[dict, dict], simple_h: tuple[dict, dict], pbo: float | None) -> dict:
    beats = all((a["sharpe"] or -9) > (b["sharpe"] or -9) for a, b in zip(sel_h, simple_h))
    adopt = "selector" if beats and (pbo is None or pbo <= PBO_LIMIT) else "simple_default"
    reason = ("selector beats the simple default in both halves and PBO <= 0.5" if adopt == "selector" else
              "selector does not beat the simple default in both halves" if not beats else
              f"PBO {pbo:.2f} > {PBO_LIMIT}: the search is overfit")
    return {"adopt": adopt, "beats_simple_both_halves": beats, "reason": reason}


def run_track(ctx: Context, track: Track, benchmark: pd.Series | None = None, adjust=None) -> dict:
    """Run every configuration of a track. ``adjust(daily) -> daily`` (optional) is applied to each
    configuration's daily frame before anything is measured, e.g. P5's carrying costs."""
    from contangolab.engine.overfitting import deflated_sharpe_ratio, probability_of_backtest_overfitting

    cfgs = grid(track)
    cache, daily = {}, {}
    for cfg in cfgs:
        grid_name = "weekly" if cfg.turnover == "weekly" else "monthly"
        key = (cfg.source, grid_name)
        if key not in cache:
            cache[key] = scores_on_grid(ctx, cfg.source, grid_name)
        d = run_config(ctx, cfg, track, cache[key])
        daily[cfg.name] = adjust(d) if adjust is not None and len(d) else d
    matrix = pd.DataFrame({n: to_monthly(d["net"]) for n, d in daily.items() if len(d)})
    switch_cost = 2 * ctx.ds.cost_bps_per_side / 1e4
    sel, picks = select(matrix, switch_cost)
    window = sel.index
    simple = simple_default(track).name
    sel_h, simple_h = halves(sel, window), halves(matrix[simple], window)

    complete = matrix.dropna()
    pbo = probability_of_backtest_overfitting({c: complete[c] for c in complete.columns})["pbo"] \
        if len(complete) >= 16 and complete.shape[1] >= 2 else None
    table = pd.DataFrame({n: {**stats(matrix[n]), "avg_turnover": float(d["turnover"].sum() / max(1, len(matrix[n].dropna())))}
                          for n, d in daily.items() if n in matrix}).T
    best = table["sharpe"].astype(float).idxmax()
    dsr = deflated_sharpe_ratio(matrix[best].dropna(), table["sharpe"].astype(float).tolist())

    out = {"track": track.name, "configs": len(cfgs), "matrix": matrix, "table": table, "daily": daily,
           "selector": sel, "picks": picks, "window": (window[0], window[-1], len(window)) if len(window) else None,
           "selector_stats": stats(sel), "selector_halves": sel_h, "simple": simple,
           "simple_stats": stats(matrix[simple].reindex(window)), "simple_halves": simple_h, "pbo": pbo,
           "hindsight_best": best, "dsr": dsr, "verdict": verdict(sel_h, simple_h, pbo)}
    if benchmark is not None:
        out["benchmark_stats"] = stats(benchmark.reindex(window))
        out["benchmark_halves"] = halves(benchmark, window)
        out["edge_vs_benchmark"] = all((a["sharpe"] or -9) > (b["sharpe"] or -9)
                                       for a, b in zip(sel_h, out["benchmark_halves"]))
    return out


def trial_records(out: dict, track: Track, ds: Dataset) -> list[dict]:
    """One registry record per configuration (family p3_<track>)."""
    recs = []
    for cfg in grid(track):
        if cfg.name not in out["matrix"]:
            continue
        r = out["matrix"][cfg.name].dropna()
        h = (stats(r.iloc[: len(r) // 2]), stats(r.iloc[len(r) // 2:]))
        recs.append({"signal": cfg.name, "asset_class": ds.asset_class, "family": f"p3_{track.name}",
                     "kind": cfg.construction, "sign": 1, "published": None, "cost_bps": ds.cost_bps_per_side,
                     "start": str(r.index.min().date()), "end": str(r.index.max().date()),
                     "avg_turnover": float(out["table"].loc[cfg.name, "avg_turnover"]),
                     **stats(r), "h1": h[0], "h2": h[1],
                     "config": {"source": cfg.source, "construction": cfg.construction, "neutral": cfg.neutral,
                                "turnover": cfg.turnover, "vt": cfg.vt}})
    return recs
