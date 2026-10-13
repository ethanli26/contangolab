"""Core drivers: dataset contract, signal registry, evaluation, trial registry.

The evaluation tests plant a signal with known predictive power in synthetic data and check the
evaluator finds it (and finds nothing in noise), that the sign from the source is applied rather
than inferred, and that the label never leaks into the signal.
"""

import numpy as np
import pandas as pd
import pytest

from contangolab.engine.dataset import Dataset, FactorData
from contangolab.engine.evaluate import evaluate, month_ends, monthly_series
from contangolab.engine.trials import bh_qvalues, correction_table, load_trials, log_trial, one_sided_p, search_summary
from contangolab.signals.registry import SignalSpec, get_signal, signals

N_DAYS, N_ASSETS = 2200, 60


def _dataset(seed=0, strength=0.0):
    """Random-walk prices; each month's score (known during that month) drives next month's drift."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2010-01-01", periods=N_DAYS)
    assets = [f"A{i:02d}" for i in range(N_ASSETS)]
    months = days.to_period("M")
    monthly = pd.DataFrame(rng.normal(size=(months.nunique(), N_ASSETS)), index=months.unique(), columns=assets)
    score = monthly.reindex(months).set_axis(days)                     # month m's score, on every day of m
    drift = monthly.shift(1).reindex(months).set_axis(days).fillna(0)  # month m's drift = month m-1's score
    daily = rng.normal(0, 0.02, (N_DAYS, N_ASSETS)) + strength * 0.002 * drift.to_numpy()
    close = pd.DataFrame(100 * np.exp(np.cumsum(daily, axis=0)), index=days, columns=assets)
    data = FactorData(open=close, high=close, low=close, close=close, volume=close * 0 + 1e6)
    ds = Dataset("synthetic", data, pd.DataFrame(True, index=days, columns=assets), cost_bps_per_side=5.0,
                 groups=5)
    return ds, score, None


def _spec(compute, sign=+1, kind="cross_sectional", published=None, name="planted"):
    return SignalSpec(name=name, asset_classes=("synthetic",), family="test", kind=kind, sign=sign,
                      source="test", published=published, compute=compute)


def test_evaluator_finds_a_planted_signal_and_ignores_noise():
    ds, skill, _ = _dataset(strength=3.0)
    found, _ = evaluate(_spec(lambda d: skill), ds)
    assert found["mean_ic"] > 0.1 and found["t_stat"] > 4
    rng = np.random.default_rng(9)
    noise = pd.DataFrame(rng.normal(size=skill.shape), index=skill.index, columns=skill.columns)
    null, _ = evaluate(_spec(lambda d: noise, name="noise"), ds)
    assert abs(null["t_stat"]) < 3


def test_sign_comes_from_the_source_not_the_data():
    ds, skill, _ = _dataset(strength=3.0)
    flipped, _ = evaluate(_spec(lambda d: skill, sign=-1), ds)
    assert flipped["mean_ic"] < -0.1 and flipped["t_stat"] < -4


def test_signal_at_t_is_paired_with_the_return_after_t():
    """A 'signal' that IS the next month's return must look perfect; lagged by a month it must not."""
    ds, _, _ = _dataset()
    close = ds.close
    me = month_ends(close.index)
    nxt_ret = (close.loc[me].shift(-1) / close.loc[me] - 1).reindex(close.index).ffill()
    leak, _ = evaluate(_spec(lambda d: nxt_ret, name="leak"), ds)
    assert leak["mean_ic"] > 0.99
    lagged = (close.loc[me] / close.loc[me].shift(1) - 1).reindex(close.index).ffill()
    honest, _ = evaluate(_spec(lambda d: lagged, name="lagged"), ds)
    assert abs(honest["mean_ic"]) < 0.1


def test_costs_scale_with_turnover():
    ds, skill, _ = _dataset(strength=3.0)
    m = monthly_series(_spec(lambda d: skill), ds)
    assert np.allclose(m["gross"] - m["net"], m["turnover"] * ds.cost_bps_per_side / 1e4)
    assert m["turnover"].iloc[0] == pytest.approx(1.0)          # opening a 50/50 book trades 1.0


def test_time_series_signal_is_inverse_vol_and_fully_invested():
    ds, skill, _ = _dataset(strength=3.0)
    m = monthly_series(_spec(lambda d: np.sign(skill), kind="time_series"), ds)
    assert m["turnover"].iloc[0] == pytest.approx(1.0) and m["gross"].mean() > 0


def test_publication_split_reports_both_sides():
    ds, skill, _ = _dataset(strength=3.0)
    res, _ = evaluate(_spec(lambda d: skill, published=2013), ds)
    assert res["pre_publication"]["months"] > 0 and res["post_publication"]["months"] > 0
    assert res["pre_publication"]["months"] + res["post_publication"]["months"] == res["months"]


def test_bh_qvalues_match_a_hand_example():
    p = pd.Series([0.01, 0.04, 0.03, 0.20], index=list("abcd"))
    # sorted 0.01, 0.03, 0.04, 0.20 -> 0.04, 0.06, 0.0533, 0.20 -> monotone 0.04, 0.0533, 0.0533, 0.20
    q = bh_qvalues(p)
    assert q["a"] == pytest.approx(0.04) and q["c"] == pytest.approx(0.16 / 3) and q["b"] == pytest.approx(0.16 / 3)
    assert q["d"] == pytest.approx(0.20)


def test_one_sided_p_is_in_the_predicted_direction():
    assert one_sided_p(3.0, 100) < 0.01 and one_sided_p(-3.0, 100) > 0.99 and one_sided_p(None, 10) == 1.0


def test_trial_registry_round_trip_and_correction(tmp_path):
    path = tmp_path / "trials.jsonl"
    ds, skill, _ = _dataset(strength=3.0)
    rng = np.random.default_rng(3)
    for name in ["planted", "noise1", "noise2"]:
        sig = skill if name == "planted" else pd.DataFrame(rng.normal(size=skill.shape), index=skill.index,
                                                           columns=skill.columns)
        res, _ = evaluate(_spec(lambda d, s=sig: s, name=name), ds)
        log_trial(res, path=path)
    log_trial({**res, "t_stat": 9.0}, path=path)                        # a rerun of noise2 supersedes it
    trials = load_trials(path)
    assert len(trials) == 4
    table = correction_table(trials).set_index("signal")
    assert len(table) == 3 and table.loc["noise2", "t_stat"] == 9.0      # latest run per signal wins
    assert bool(table.loc["planted", "survivor"])
    assert search_summary(trials)["n_trials"] == 3


def test_registry_holds_both_markets_with_fixed_signs():
    cn, us = signals("cn_futures"), signals("us_futures")
    assert {"fut_ma20_60", "fut_rank63", "fut_carry"} <= {s.name for s in cn}
    assert {"usf_carry", "usf_basismom", "usf_hedgep"} <= {s.name for s in us}
    assert all(s.sign == +1 for s in cn + us)
    assert get_signal("fut_ts12").kind == "time_series" and get_signal("fut_ts12").published == 2012
