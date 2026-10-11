"""Tests for the results report: the stitched walk-forward path, the universe benchmark,
and an end-to-end render of the markdown + charts on synthetic data (no Sharadar)."""

import numpy as np
import pandas as pd

from backtest.walkforward import make_windows, universe_equal_weight_returns
from factors.base import FactorData
from factors.evaluate import _rebalance_dates
from research import report
from strategies.vol_managed_momentum import VolManagedMomentum


def _data(n_days=2700, n_symbols=60, seed=3):
    """~10 years of random-walk prices, so there are several 3-year walk-forward windows."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n_days)
    close = pd.DataFrame(20 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, (n_days, n_symbols)), axis=0)),
                         index=idx, columns=[f"S{i:02d}" for i in range(n_symbols)])
    data = FactorData(open=close, high=close * 1.01, low=close * 0.99, close=close,
                      volume=close * 0 + 1e6, market=close.mean(axis=1))
    eligible = pd.DataFrame(True, index=idx, columns=close.columns)
    eligible.iloc[:, :5] = False   # a few never-eligible names must not leak into the benchmark
    return data, eligible


def test_walk_forward_path_is_the_windows_chained_without_overlap():
    data, eligible = _data()
    strategy = VolManagedMomentum(long_only=True)
    stitched, windows = report.walk_forward_returns(strategy, data, eligible)
    assert 2 <= len(windows) <= len(make_windows(_rebalance_dates(data.close.index, "M")))
    assert stitched.index.is_monotonic_increasing and not stitched.index.duplicated().any()
    # Every stitched month comes from exactly one window, scaled with that window's own target.
    for start, end, window_returns in windows:
        assert window_returns.index.min() >= start and window_returns.index.max() < end
        pd.testing.assert_series_equal(stitched.loc[window_returns.index], window_returns, check_names=False)
    assert len(stitched) == sum(len(w) for _, _, w in windows)


def test_universe_benchmark_is_the_equal_weight_of_eligible_names_only():
    data, eligible = _data()
    bench = universe_equal_weight_returns(data, eligible)
    rebal = _rebalance_dates(data.close.index, "M")
    current, nxt = rebal[20], rebal[21]
    expected = (data.close.loc[nxt] / data.close.loc[current] - 1.0).iloc[5:].mean()
    assert np.isclose(bench.loc[nxt], expected)


def test_report_renders_markdown_and_charts(tmp_path, monkeypatch):
    """Drive write_markdown + every chart from a fake study result."""
    data, eligible = _data()
    bench = universe_equal_weight_returns(data, eligible)

    def fake_universe(top_n):
        variants = {}
        for long_only in (True, False):
            strategy = VolManagedMomentum(long_only=long_only)
            wf, windows = report.walk_forward_returns(strategy, data, eligible)
            b = bench.reindex(wf.index).dropna()
            variants["long_only" if long_only else "long_short"] = {
                "walk_forward": report._metrics(wf), "universe_ew": report._metrics(b),
                "returns": wf, "benchmark": b,
                "windows": [{"window": f"{s.year}-{e.year}", "strategy": report._metrics(r),
                             "universe_ew": report._metrics(bench.reindex(r.index).dropna())}
                            for s, e, r in windows],
                "cost_sensitivity": {"5": report._metrics(wf), "15": report._metrics(wf), "40": report._metrics(wf)},
                "annual_turnover": 3.0,
            }
        return {"mode": "pit", "top_n": top_n, "median_eligible_names": 55, "variants": variants}

    pit, static, matched = fake_universe(1500), fake_universe(1500), fake_universe(750)
    dsr = {"dsr": 0.4, "sr0_annualized": 0.5}
    pit["overfitting"] = {
        "pbo": {"pbo": 0.1, "n_splits": 70, "n_variants": 5},
        "long_only": {"cpcv_sharpes": [0.1, 0.4, -0.2], "dsr_vs_searched_variants": dsr, "dsr_vs_cpcv_paths": dsr},
        "long_short": {"cpcv_sharpes": [0.3, 0.2], "dsr_vs_searched_variants": dsr, "dsr_vs_cpcv_paths": dsr},
    }
    monkeypatch.setattr(report, "evaluate_universe",
                        lambda mode, full, top_n=None: matched if top_n else (pit if mode == "pit" else static))
    assert report.main(out_dir=tmp_path) == 0
    text = (tmp_path / "RESULTS.md").read_text()
    assert "## Walk-forward windows" in text and "size-matched" in text
    for chart in ("equity_curve", "universe_bias", "drawdown", "walkforward_windows", "cpcv_sharpes"):
        assert (tmp_path / f"{chart}.png").stat().st_size > 10_000
    assert (tmp_path / "results.json").exists()
