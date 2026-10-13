"""Alpha model: many signals -> one score, fit walk-forward only (BLUEPRINT 11, P2).

Protocol ``b11-p2-v1``, pre-registered in the vault before any combination was run:

  * Pool: every registered signal for the asset class, with the registry's sign. Not the P1b
    survivors, because the survivor flag is computed on the full sample and filtering on it
    would leak the test period into the pool. Dropping dead signals is the combiner's job.
  * Features at month-end t (only eligible names): a cross-sectional input becomes its centered
    rank, scaled to unit variance. In the futures (time-series) combination a time-series input
    is divided by its own pooled standard deviation over month-ends strictly before t and
    clipped at +-3. Missing values become 0.
  * Target: cross-sectional, the unit-variance centered rank of the t -> t+1 return;
    time-series, that return over (60-day vol x sqrt 21), clipped at +-3.
  * Walk-forward: the score at month i uses months j < i only (the label of month j is the
    return to month j+1, realized by month i). The first ``MIN_TRAIN_MONTHS`` are training only.
  * Methods, hyperparameters fixed in advance (no tuning):
        ew     mean of the features
        icw    weights proportional to max(expanding mean Spearman IC, 0); a signal whose IC runs
               against its registered sign gets weight 0, never a flipped sign
        ridge  pooled least squares without intercept, penalty = number of training rows,
               refit every month
        xgb    gradient-boosted trees (``XGB_PARAMS``), refit every ``XGB_REFIT_MONTHS``
  * Each score is evaluated as a new signal by ``contangolab.engine.evaluate.monthly_series`` (same costs and
    portfolios as P1b) on the out-of-sample months only.
  * Bar: a method passes if its OOS net Sharpe beats both the best single signal and ``ew`` in
    each half of the OOS period separately (the best single is chosen after the fact in each
    half, which favors the benchmark). ``ew`` only has to beat the best single.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from contangolab.engine.dataset import Dataset
from contangolab.engine.evaluate import VOL_WINDOW, _block, month_ends, monthly_series, summarize
from contangolab.signals.registry import SignalSpec

PROTOCOL = "b11-p2-v1"
METHODS = ("ew", "icw", "ridge", "xgb")
COMBO_KIND = {"cn_futures": "time_series"}
MIN_TRAIN_MONTHS = 36
XGB_REFIT_MONTHS = 12
XGB_PARAMS = {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8,
              "colsample_bytree": 0.8, "min_child_weight": 50, "random_state": 0, "tree_method": "hist"}
CLIP = 3.0
UNIT = math.sqrt(12)                # 1 / std of a uniform on [-0.5, 0.5]: centered ranks -> unit variance
DAYS_PER_MONTH = 21


def centered_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """Row-wise rank mapped to (-0.5, 0.5) and scaled to unit variance; NaN stays NaN."""
    n = panel.notna().sum(axis=1)
    return ((panel.rank(axis=1) - 0.5).div(n, axis=0) - 0.5) * UNIT


def past_scaled(panel: pd.DataFrame) -> pd.DataFrame:
    """Divide each row by the pooled std of all earlier rows (strictly before), clipped at +-CLIP."""
    v = panel.to_numpy(dtype=float)
    ok = np.isfinite(v)
    cnt = np.cumsum(ok.sum(axis=1))
    s1 = np.cumsum(np.where(ok, v, 0.0).sum(axis=1))
    s2 = np.cumsum(np.where(ok, v * v, 0.0).sum(axis=1))
    var = (s2 - s1 * s1 / np.maximum(cnt, 1)) / np.maximum(cnt - 1, 1)
    sd = pd.Series(np.sqrt(np.where(cnt > 1, var, np.nan)), index=panel.index).shift(1)
    return panel.div(sd, axis=0).clip(-CLIP, CLIP)


@dataclass
class Panels:
    """Month-end inputs for the combination: one feature matrix and label per month."""
    dates: pd.DatetimeIndex                     # grid dates that have a following grid date
    names: list[str]
    X: list[np.ndarray]                         # per month: eligible assets x features
    y: list[np.ndarray]                         # per month: label (NaN if not realized)
    assets: list[pd.Index]
    raw: dict[str, pd.DataFrame] = field(default_factory=dict)   # month-end signal values, unsigned


def build_panels(specs: list[SignalSpec], ds: Dataset, combo_kind: str,
                 rebal: pd.DatetimeIndex | None = None) -> Panels:
    """Features and labels on a rebalance grid (default: month-ends); the label runs to the next grid date."""
    close = ds.close
    rebal = month_ends(close.index) if rebal is None else rebal
    dates = rebal[:-1]
    elig = ds.eligible.reindex(index=dates, columns=close.columns).fillna(False).astype(bool)

    raw, feats = {}, []
    for spec in specs:
        panel = spec.compute(ds).reindex(index=rebal, columns=close.columns).replace([np.inf, -np.inf], np.nan)
        raw[spec.name] = panel
        signed = (panel.loc[dates] * spec.sign).where(elig)
        if combo_kind == "cross_sectional" or spec.kind == "cross_sectional":
            f = centered_rank(signed)
        else:
            f = past_scaled(signed)
        feats.append(f.where(elig, np.nan).fillna(0.0).where(elig))

    fwd = (close.loc[rebal].shift(-1) / close.loc[rebal] - 1.0).loc[dates].where(elig)
    if combo_kind == "cross_sectional":
        label = centered_rank(fwd)
    else:
        vol = close.pct_change(fill_method=None).rolling(VOL_WINDOW).std().reindex(dates)
        label = (fwd / (vol * math.sqrt(DAYS_PER_MONTH))).replace([np.inf, -np.inf], np.nan).clip(-CLIP, CLIP)

    cube = np.stack([f.to_numpy() for f in feats], axis=2)          # dates x assets x features
    X, y, assets = [], [], []
    for i in range(len(dates)):
        row = elig.iloc[i].to_numpy()
        X.append(cube[i, row, :])
        y.append(label.iloc[i].to_numpy()[row])
        assets.append(close.columns[row])
    return Panels(dates, [s.name for s in specs], X, y, assets, raw)


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.full(x.shape[1], np.nan)
    rx = pd.DataFrame(x[ok]).rank().to_numpy()
    ry = pd.Series(y[ok]).rank().to_numpy()
    rx, ry = rx - rx.mean(axis=0), ry - ry.mean()
    den = np.sqrt((rx * rx).sum(axis=0) * (ry * ry).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, rx.T @ ry / den, np.nan)


def walk_forward(p: Panels, methods=METHODS, min_train: int = MIN_TRAIN_MONTHS) -> tuple[dict, dict]:
    """Score panels per method on months min_train.. (indexed by month-end), plus diagnostics."""
    k = len(p.names)
    ics = np.array([_spearman_ic(x, y) for x, y in zip(p.X, p.y)]) if "icw" in methods else None
    xtx, xty, n_rows = np.zeros((k, k)), np.zeros(k), 0
    out = {m: {} for m in methods}
    diag = {"icw_weights": {}, "ridge_beta": {}}
    model = None
    for i in range(len(p.dates)):
        if i >= min_train:
            t, x = p.dates[i], p.X[i]
            if "ew" in methods:
                out["ew"][t] = pd.Series(x.mean(axis=1), index=p.assets[i])
            if "icw" in methods:
                w = np.clip(np.nan_to_num(np.nanmean(ics[:i], axis=0)), 0.0, None)
                w = w / w.sum() if w.sum() > 0 else np.full(k, 1.0 / k)
                diag["icw_weights"][t] = w
                out["icw"][t] = pd.Series(x @ w, index=p.assets[i])
            if "ridge" in methods:
                beta = np.linalg.solve(xtx + n_rows * np.eye(k), xty)
                diag["ridge_beta"][t] = beta
                out["ridge"][t] = pd.Series(x @ beta, index=p.assets[i])
            if "xgb" in methods:
                if (i - min_train) % XGB_REFIT_MONTHS == 0:
                    model = _fit_xgb(p, i)
                out["xgb"][t] = pd.Series(model.predict(x), index=p.assets[i])
        ok = np.isfinite(p.y[i])                   # month i's label joins training for month i+1
        xtx += p.X[i][ok].T @ p.X[i][ok]
        xty += p.X[i][ok].T @ p.y[i][ok]
        n_rows += int(ok.sum())
    scores = {m: pd.DataFrame(v).T.sort_index() for m, v in out.items()}
    return scores, diag


def _fit_xgb(p: Panels, i: int):
    from xgboost import XGBRegressor

    X = np.vstack(p.X[:i])
    y = np.concatenate(p.y[:i])
    ok = np.isfinite(y)
    return XGBRegressor(**XGB_PARAMS).fit(X[ok], y[ok])


def _sharpe(net: pd.Series) -> float | None:
    return _block(net, net.iloc[:0])["sharpe"]


def verdict(sharpes: pd.DataFrame) -> dict:
    """Apply the pre-registered bar to a table with rows = methods + "best_single" and columns h1/h2/oos."""
    passed = {}
    for m in [m for m in METHODS if m in sharpes.index]:
        bars = ["best_single"] + ([] if m == "ew" else ["ew"])
        passed[m] = all(sharpes.loc[m, h] > sharpes.loc[b, h] for h in ("h1", "h2") for b in bars
                        if pd.notna(sharpes.loc[m, h]) and pd.notna(sharpes.loc[b, h])) \
            and sharpes.loc[m, ["h1", "h2"]].notna().all()
    complex_pass = [m for m in ("icw", "ridge", "xgb") if passed.get(m)]
    if complex_pass:
        chosen = max(complex_pass, key=lambda m: sharpes.loc[m, "oos"])
    elif passed.get("ew"):
        chosen = "ew"
    else:
        chosen = None
    return {"passed": passed, "chosen": chosen,
            "text": f"alpha model = {chosen}" if chosen else "no combination beats the best single signal"}


def run(ds: Dataset, specs: list[SignalSpec], methods=METHODS, min_train: int = MIN_TRAIN_MONTHS) -> dict:
    """Fit every method walk-forward, evaluate OOS against the single signals, apply the bar."""
    combo_kind = COMBO_KIND.get(ds.asset_class, "cross_sectional")
    panels = build_panels(specs, ds, combo_kind)
    scores, diag = walk_forward(panels, methods, min_train)

    combos = {}
    for m in methods:
        spec = SignalSpec(name=f"alpha_{m}", asset_classes=(ds.asset_class,), family="alpha_model", kind=combo_kind,
                          sign=1, source=f"contangolab.engine.combine {PROTOCOL}: {m} over {len(specs)} signals", published=None,
                          compute=lambda _ds, s=scores[m]: s)
        combos[m] = (spec, monthly_series(spec, ds, signal=scores[m]))
    oos = sorted(set.intersection(*[set(mo.index) for _, mo in combos.values()]))
    half = oos[len(oos) // 2]

    singles = {s.name: monthly_series(s, ds, signal=panels.raw[s.name]).reindex(oos) for s in specs}

    def blocks(net: pd.Series) -> dict:
        net = net.dropna()
        return {"oos": _sharpe(net), "h1": _sharpe(net[net.index < half]), "h2": _sharpe(net[net.index >= half])}

    rows = {m: blocks(mo["net"].reindex(oos)) for m, (_, mo) in combos.items()}
    single_rows = pd.DataFrame({n: blocks(mo["net"]) for n, mo in singles.items()}).T
    table = pd.DataFrame(rows).T
    best = {h: single_rows[h].max() for h in ("oos", "h1", "h2")}
    best_names = {h: single_rows[h].idxmax() for h in ("oos", "h1", "h2")}
    table.loc["best_single"] = best

    results = {}
    for m, (spec, mo) in combos.items():
        res = summarize(spec, ds, mo.reindex(oos).dropna(how="all"))
        res.update(method=m, pool=panels.names, pool_size=len(specs), oos_start=str(oos[0].date()),
                   min_train_months=min_train)
        results[m] = res
    return {"asset_class": ds.asset_class, "combo_kind": combo_kind, "oos": (oos[0], oos[-1], len(oos)),
            "half": half, "table": table, "singles": single_rows, "best_names": best_names,
            "verdict": verdict(table), "results": results, "scores": scores, "diag": diag, "panels": panels,
            "monthly": {m: mo.reindex(oos) for m, (_, mo) in combos.items()}}
