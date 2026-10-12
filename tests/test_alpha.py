"""Alpha model (P2): walk-forward combination of signals.

Synthetic data with planted signals of known strength: the combiners must find them, must not
see the future (changing prices after a date leaves every earlier score unchanged), must not
flip a signal whose registered sign is wrong, and the pre-registered bar must be applied as written.
"""

import numpy as np
import pandas as pd
import pytest

from core.alpha import Panels, build_panels, centered_rank, past_scaled, run, verdict, walk_forward
from core.dataset import Dataset
from core.signals import SignalSpec
from factors.base import FactorData

N_DAYS, N_ASSETS, MIN_TRAIN = 2200, 80, 24


def _dataset(strengths, seed=0, n_noise=2):
    """Prices driven by independent monthly scores; score k moves next month's drift by strengths[k]."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2010-01-01", periods=N_DAYS)
    assets = [f"A{i:02d}" for i in range(N_ASSETS)]
    months = days.to_period("M")
    uniq = months.unique()
    scores, drift = [], np.zeros((N_DAYS, N_ASSETS))
    for s in list(strengths) + [0.0] * n_noise:
        m = pd.DataFrame(rng.normal(size=(len(uniq), N_ASSETS)), index=uniq, columns=assets)
        scores.append(m.reindex(months).set_axis(days))
        drift += s * 0.002 * m.shift(1).reindex(months).set_axis(days).fillna(0).to_numpy()
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (N_DAYS, N_ASSETS)) + drift, axis=0)),
                         index=days, columns=assets)
    data = FactorData(open=close, high=close, low=close, close=close, volume=close * 0 + 1e6)
    ds = Dataset("synthetic", data, pd.DataFrame(True, index=days, columns=assets), cost_bps_per_side=5.0, groups=5)
    return ds, scores


def _specs(scores, signs=None, kind="cross_sectional"):
    signs = signs or [1] * len(scores)
    return [SignalSpec(name=f"s{k}", asset_classes=("synthetic",), family="test", kind=kind, sign=signs[k],
                       source="test", published=None, compute=lambda d, s=s: s) for k, s in enumerate(scores)]


def test_centered_rank_is_unit_variance_and_past_scaled_uses_only_earlier_rows():
    rng = np.random.default_rng(1)
    panel = pd.DataFrame(rng.normal(size=(50, 400)))
    r = centered_rank(panel)
    assert abs(r.to_numpy().mean()) < 1e-12 and r.to_numpy().std() == pytest.approx(1.0, abs=0.01)
    scaled = past_scaled(panel)
    assert scaled.iloc[0].isna().all()                                    # nothing before the first row
    changed = panel.copy()
    changed.iloc[30:] *= 100                                               # row 30's own value must not set its scale
    sd = np.nanstd(panel.iloc[:30].to_numpy(), ddof=1)
    assert np.allclose(past_scaled(changed).iloc[30], (changed.iloc[30] / sd).clip(-3, 3))
    assert np.allclose(past_scaled(changed).iloc[:30], scaled.iloc[:30], equal_nan=True)


def test_no_look_ahead_scores_ignore_prices_after_the_scoring_date():
    ds, scores = _dataset([3.0, 2.0])
    specs = _specs(scores)
    base, _ = walk_forward(build_panels(specs, ds, "cross_sectional"), min_train=MIN_TRAIN)
    cut = base["ridge"].index[10]
    shocked = ds.close.copy()
    after = shocked.index > cut
    shocked.loc[after] *= np.random.default_rng(5).uniform(0.5, 2.0, size=shocked.loc[after].shape)
    ds2 = Dataset("synthetic", FactorData(open=shocked, high=shocked, low=shocked, close=shocked,
                                          volume=ds.data.volume), ds.eligible, 5.0, 5)
    moved, _ = walk_forward(build_panels(specs, ds2, "cross_sectional"), min_train=MIN_TRAIN)
    for m in base:                         # ew uses no labels, so only the learners can move after the cut
        a, b = base[m].loc[:cut], moved[m].loc[:cut]
        assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True), m
        if m != "ew":
            assert not np.allclose(base[m].iloc[-1].to_numpy(), moved[m].iloc[-1].to_numpy()), m


def test_combination_beats_each_part_and_learners_down_weight_noise():
    ds, scores = _dataset([2.0, 2.0], n_noise=6)
    out = run(ds, _specs(scores), min_train=MIN_TRAIN)
    t, singles = out["table"], out["singles"]
    assert t.loc["ew", "oos"] > 0
    assert t.loc["ridge", "oos"] > max(singles.loc["s0", "oos"], singles.loc["s1", "oos"])
    beta = np.mean(list(out["diag"]["ridge_beta"].values()), axis=0)
    assert beta[:2].min() > 3 * np.abs(beta[2:]).max()                   # planted >> noise
    w = np.mean(list(out["diag"]["icw_weights"].values()), axis=0)
    assert w[:2].sum() > 0.6


def test_icw_never_flips_a_signal_registered_with_the_wrong_sign():
    ds, scores = _dataset([3.0, 2.0], n_noise=0)
    p = build_panels(_specs(scores, signs=[-1, 1]), ds, "cross_sectional")
    _, diag = walk_forward(p, methods=("icw",), min_train=MIN_TRAIN)
    w = np.array(list(diag["icw_weights"].values()))
    assert np.all(w[:, 0] == 0.0) and np.all(w[:, 1] == 1.0)


def test_time_series_combination_trades_the_sign_of_the_score():
    ds, scores = _dataset([3.0], n_noise=1)
    out = run(ds, _specs(scores, kind="time_series"), methods=("ew", "ridge"), min_train=MIN_TRAIN)
    assert out["combo_kind"] == "cross_sectional"                         # "synthetic" defaults to CS
    p = build_panels(_specs(scores, kind="time_series"), ds, "time_series")
    assert isinstance(p, Panels) and np.nanmax(np.abs(np.vstack(p.X))) <= 3.0


@pytest.mark.parametrize("table, chosen", [
    ({"ew": (0.5, 0.6, 0.4), "icw": (0.7, 0.8, 0.6), "ridge": (0.9, 1.0, 0.3), "xgb": (0.2, 0.1, 0.3),
      "best_single": (0.4, 0.5, 0.35)}, "icw"),                         # ridge fails h2 vs ew
    ({"ew": (0.5, 0.6, 0.4), "icw": (0.45, 0.5, 0.4), "ridge": (0.4, 0.4, 0.4), "xgb": (0.3, 0.3, 0.3),
      "best_single": (0.4, 0.5, 0.35)}, "ew"),
    ({"ew": (0.5, 0.6, 0.3), "icw": (0.6, 0.7, 0.34), "ridge": (0.4, 0.4, 0.4), "xgb": (0.3, 0.3, 0.3),
      "best_single": (0.4, 0.5, 0.35)}, None),                          # nobody beats best single in h2
])
def test_verdict_applies_the_pre_registered_bar(table, chosen):
    df = pd.DataFrame(table, index=["oos", "h1", "h2"]).T
    assert verdict(df)["chosen"] == chosen
