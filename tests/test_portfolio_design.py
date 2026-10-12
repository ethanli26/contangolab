"""P3: portfolio construction, simulation, and the design-space selector.

Hand-checkable cases for the accounting (drift, turnover, costs), the buffer and the vol target;
look-ahead checks for a whole configuration run and for the selector; the verdict rule as written.
"""

import numpy as np
import pandas as pd
import pytest

from core.dataset import Dataset
from core.design import TRACKS, Config, grid, prepare, run_config, scores_on_grid, select, simple_default, verdict
from core.portfolio import buffered_top, construct, ex_ante_vol, neutralize, simulate, to_monthly, vol_scale
from core.signals import SignalSpec
from factors.base import FactorData

DAYS = pd.bdate_range("2020-01-01", periods=6)


def test_simulate_drifts_holdings_and_charges_turnover_from_drifted_weights():
    r = pd.DataFrame({"A": [0.0, 0.10, 0.0, 0.0, 0.0, 0.0], "B": [0.0, -0.10, 0.0, 0.0, 0.0, 0.0]}, index=DAYS)
    w = pd.Series({"A": 0.5, "B": 0.5})
    out = simulate({DAYS[0]: w, DAYS[2]: w}, r, cost_bps=10.0)
    assert out["gross"].iloc[0] == pytest.approx(0.0)                    # day 1: (0.5*1.1 + 0.5*0.9) - 1 = 0
    # after day 1, A is 0.55/1.0 and B 0.45/1.0; rebalancing back to 50/50 trades 0.05 + 0.05
    assert out.loc[DAYS[3], "turnover"] == pytest.approx(0.10)
    assert out.loc[DAYS[3], "cost"] == pytest.approx(0.10 * 10 / 1e4)
    assert out.loc[DAYS[1], "turnover"] == pytest.approx(1.0)            # opening the book


def test_simulate_long_short_value_is_exact():
    r = pd.DataFrame({"A": [0.0, 0.10, 0.10, 0.0, 0.0, 0.0], "B": [0.0, 0.05, 0.05, 0.0, 0.0, 0.0]}, index=DAYS)
    out = simulate({DAYS[0]: pd.Series({"A": 0.5, "B": -0.5})}, r, cost_bps=0.0)
    value = (1 + out["net"]).cumprod()
    assert value.iloc[-1] == pytest.approx(1 + 0.5 * (1.1 ** 2 - 1) - 0.5 * (1.05 ** 2 - 1))


def test_buffer_keeps_held_names_inside_the_wider_band():
    score = pd.Series(np.arange(10, 0, -1.0), index=list("abcdefghij"))   # a best ... j worst
    assert buffered_top(score, 3, 3, held=set()) == ["a", "b", "c"]
    assert set(buffered_top(score, 3, 6, held={"e", "f", "j"})) == {"e", "f", "a"}   # j fell out of the top 6
    assert buffered_top(score, 3, 3, held={"e", "f"}) == ["a", "b", "c"]          # no buffer: plain top 3


def test_constructions_have_the_stated_exposures():
    rng = np.random.default_rng(0)
    names = [f"S{i}" for i in range(50)]
    score, vol = pd.Series(rng.normal(size=50), index=names), pd.Series(rng.uniform(0.01, 0.03, 50), index=names)
    ls = construct(score, "ls_decile", vol, {})
    assert ls[ls > 0].sum() == pytest.approx(0.5) and ls[ls < 0].sum() == pytest.approx(-0.5) and len(ls) == 10
    top = construct(score, "top20_iv", vol, {})
    assert top.sum() == pytest.approx(1.0) and len(top) == 20 and (top > 0).all()
    assert set(top.index) == set(score.nlargest(20).index)
    ts = construct(score, "ts", vol, {})
    assert ts.abs().sum() == pytest.approx(1.0) and (np.sign(ts) == np.sign(score[ts.index])).all()


def test_neutralize_ranks_within_groups():
    score = pd.Series([1.0, 2.0, 100.0, 200.0], index=list("abcd"))
    out = neutralize(score, pd.Series({"a": "x", "b": "x", "c": "y", "d": "y"}))
    assert out["a"] == out["c"] and out["b"] == out["d"]


def test_vol_target_hits_the_target_and_respects_the_cap():
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.normal(0, 0.02, (200, 3)), columns=list("abc"))
    w = pd.Series(1 / 3, index=list("abc"))
    scaled = vol_scale(w, r, 150, target=0.10, cap=5.0)
    assert ex_ante_vol(scaled, r, 150) == pytest.approx(0.10)
    assert vol_scale(w, r, 150, target=10.0, cap=2.0).sum() == pytest.approx(2.0)


def test_to_monthly_compounds():
    d = pd.Series([0.01, 0.02, -0.01], index=pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03"]))
    m = to_monthly(d)
    assert m.iloc[0] == pytest.approx(1.01 * 1.02 - 1) and m.index[0] == pd.Timestamp("2020-01-31")


def _synthetic(close=None, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2010-01-01", periods=1500)
    assets = [f"A{i:02d}" for i in range(40)]
    months = days.to_period("M")
    m = pd.DataFrame(rng.normal(size=(months.nunique(), 40)), index=months.unique(), columns=assets)
    skill = m.reindex(months).set_axis(days)
    drift = 0.006 * m.shift(1).reindex(months).set_axis(days).fillna(0).to_numpy()
    if close is None:
        close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (1500, 40)) + drift, axis=0)),
                             index=days, columns=assets)
    data = FactorData(open=close, high=close, low=close, close=close, volume=close * 0 + 1e6)
    ds = Dataset("synthetic", data, pd.DataFrame(True, index=days, columns=assets), cost_bps_per_side=5.0, groups=5)
    noise = pd.DataFrame(rng.normal(size=skill.shape), index=days, columns=assets)
    specs = [SignalSpec(name=n, asset_classes=("synthetic",), family="t", kind="cross_sectional", sign=1, source="t",
                        published=None, compute=lambda d, s=s: s) for n, s in [("skill", skill), ("noise", noise)]]
    return ds, specs


def test_config_run_has_no_look_ahead():
    ds, specs = _synthetic()
    track = TRACKS["us_lo"]
    cfg = Config("us_lo", "best_trailing", "top20_ew", False, "buffer", True)
    ctx = prepare(ds, specs, min_train=24)
    base = run_config(ctx, cfg, track, scores_on_grid(ctx, cfg.source, "monthly"))
    cut = base.index[len(base) // 2]
    shocked = ds.close.copy()
    shocked.loc[shocked.index > cut] *= np.random.default_rng(7).uniform(0.7, 1.3, shocked.loc[shocked.index > cut].shape)
    ds2, specs2 = _synthetic(close=shocked)
    ctx2 = prepare(ds2, specs2, min_train=24)
    moved = run_config(ctx2, cfg, track, scores_on_grid(ctx2, cfg.source, "monthly"))
    assert np.allclose(base.loc[:cut, "net"], moved.loc[:cut, "net"])
    assert base["net"].sum() > 0                                          # the planted signal pays


def test_selector_uses_only_the_past_and_switches_to_the_better_config():
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(3)
    a = pd.Series(np.r_[rng.normal(0.02, 0.01, 60), rng.normal(-0.02, 0.01, 60)], index=idx)
    b = pd.Series(rng.normal(0.005, 0.01, 120), index=idx)
    matrix = pd.DataFrame({"a": a, "b": b})
    rets, picks = select(matrix, switch_cost=0.0)
    assert picks[0][1] == "a" and picks[-1][1] == "b"
    changed = matrix.copy()
    changed.iloc[100:] = 0.5
    _, picks2 = select(changed, switch_cost=0.0)
    assert [p for p in picks2 if p[0] < idx[100]] == [p for p in picks if p[0] < idx[100]]
    assert rets.index[0] == idx[36]                                       # first pick after 36 months


def test_grid_sizes_match_the_pre_registration():
    assert [len(grid(TRACKS[t])) for t in ("us_ls", "us_lo", "fut")] == [32, 96, 32]
    assert simple_default(TRACKS["us_lo"]).name == "us_lo|all_ew|top50_ew|raw|none|novt"


@pytest.mark.parametrize("sel, simple, pbo, adopt", [
    ((0.6, 0.5), (0.4, 0.4), 0.3, "selector"),
    ((0.6, 0.3), (0.4, 0.4), 0.3, "simple_default"),
    ((0.6, 0.5), (0.4, 0.4), 0.7, "simple_default"),
])
def test_verdict_rule(sel, simple, pbo, adopt):
    h = lambda pair: tuple({"sharpe": s} for s in pair)  # noqa: E731
    assert verdict(h(sel), h(simple), pbo)["adopt"] == adopt
