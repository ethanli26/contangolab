"""Results report: every number and chart in ``reports/``, regenerated from the local archive.

The README and ARCHITECTURE.md quote results. This module is where those results come
from, so nothing in the docs has to be typed in by hand:

  * a STITCHED WALK-FORWARD equity curve — each non-overlapping 3-year window is run out of
    sample with its vol target fixed from data strictly before the window, then the windows
    are chained. It's the closest thing to "what you would have experienced" this repo can
    produce; a full-history backtest would set the vol target with hindsight.
  * the same curve on the old STATIC universe (full-history liquidity ranking), so the cost
    of that look-ahead is measured, not asserted.
  * per-window Sharpe vs the equal-weight return of the same investable universe,
  * cost sensitivity at the config slippage tiers,
  * CPCV path Sharpes, the Deflated Sharpe Ratio, and PBO.

Writes ``reports/results.json`` (machine-readable), ``reports/RESULTS.md`` and PNG charts.
Offline from the Sharadar cache; takes a while because it loads both universes.

    python main.py report
"""

import gc
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import config
from backtest.walkforward import make_windows, universe_equal_weight_returns
from factors.evaluate import _rebalance_dates
from research.combine_train import _monthly_metrics
from research.cpcv import (
    annualized_sharpe,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    run_cpcv,
)
from strategies.vol_managed_momentum import VolManagedMomentum

log = logging.getLogger("report")

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"
MATCHED_TOP_N = 750   # point-in-time universe sized to the static list's eligible breadth
COST_TIERS_BPS = (config.SLIPPAGE_BPS_LARGE, config.VMM_COST_BPS_PER_SIDE, config.SLIPPAGE_BPS_SMALL)

# Chart styling: one recessive grid, thin 2px lines, text in ink colors (never series colors).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8d8b85"


# --- return streams ----------------------------------------------------------------------

def walk_forward_returns(strategy: VolManagedMomentum, data, eligible: pd.DataFrame) -> tuple[pd.Series, list]:
    """Chain the non-overlapping walk-forward test windows into one OOS monthly series.

    Returns ``(returns, windows)`` where ``windows`` is ``[(start, end, window_returns)]``.
    """
    rebal = _rebalance_dates(data.close.index, "M")
    pieces, windows = [], []
    for start, end in make_windows(rebal):
        ret = strategy.portfolio_returns(data, eligible, train_end=start, start=start, end=end)
        ret = ret[ret.index < end]
        if len(ret) < 6:
            continue
        pieces.append(ret)
        windows.append((start, end, ret))
    stitched = pd.concat(pieces).sort_index()
    return stitched[~stitched.index.duplicated()], windows


def _metrics(returns: pd.Series) -> dict:
    m = _monthly_metrics(returns)
    vol = float(returns.std(ddof=1) * np.sqrt(12)) if len(returns) > 1 else None
    return {**m, "vol": vol, "months": int(len(returns)),
            "start": str(returns.index[0].date()), "end": str(returns.index[-1].date())}


# --- the study ---------------------------------------------------------------------------

def evaluate_universe(mode: str, full: bool, top_n: int | None = None) -> dict:
    """Run both strategy variants on one universe; ``full`` adds costs + CPCV/DSR/PBO."""
    from factors.run_factor_eval import SHARADAR_MAX_SYMBOLS, build_sharadar_factor_data

    top_n = top_n or SHARADAR_MAX_SYMBOLS
    log.info("Building the %s universe (top %d)...", mode, top_n)
    data, eligible = build_sharadar_factor_data(universe_mode=mode, top_n=top_n)
    bench = universe_equal_weight_returns(data, eligible)
    rebal = _rebalance_dates(data.close.index, "M")
    result = {"mode": mode, "top_n": top_n, "variants": {},
              "median_eligible_names": int(eligible.loc[rebal].sum(axis=1).median())}

    for long_only in (True, False):
        strategy = VolManagedMomentum(long_only=long_only)
        key = "long_only" if long_only else "long_short"
        wf, windows = walk_forward_returns(strategy, data, eligible)
        b = bench.reindex(wf.index).dropna()
        variant = {
            "walk_forward": _metrics(wf),
            "universe_ew": _metrics(b),
            "returns": wf,
            "benchmark": b,
            "windows": [{
                "window": f"{s.year}-{e.year}",
                "strategy": _metrics(r),
                "universe_ew": _metrics(bench.reindex(r.index).dropna()),
            } for s, e, r in windows],
        }
        if full:
            variant["cost_sensitivity"] = {
                f"{bps:g}": _metrics(walk_forward_returns(
                    VolManagedMomentum(long_only=long_only, cost_bps_per_side=bps), data, eligible)[0])
                for bps in COST_TIERS_BPS}
            variant["annual_turnover"] = strategy.annual_turnover(data, eligible)
        result["variants"][key] = variant
        log.info("%s top %d / %s: walk-forward Sharpe %.2f vs universe EW %.2f", mode, top_n, key,
                 variant["walk_forward"]["sharpe"], variant["universe_ew"]["sharpe"])

    if full:
        result["overfitting"] = overfitting_checks(data, eligible)
    del data, eligible
    gc.collect()
    return result


def overfitting_checks(data, eligible: pd.DataFrame) -> dict:
    """CPCV distribution, Deflated Sharpe (two trial sets), and PBO for both variants."""
    from research.momentum_variants import build_variant_returns

    rebal = _rebalance_dates(data.close.index, "M")
    variant_returns, _, _, _ = build_variant_returns(data, eligible)
    pbo = probability_of_backtest_overfitting(variant_returns)
    out = {"pbo": {k: pbo[k] for k in ("pbo", "n_splits", "n_variants", "variant_names")}}
    # Trials = the configurations actually searched: the five pre-registered momentum
    # variants, plus the long-only version that was then chosen for paper trading.
    searched = {**variant_returns,
                "long_only_vol_managed": VolManagedMomentum(long_only=True).portfolio_returns(data, eligible)}
    searched_sharpes = [annualized_sharpe(s) for s in searched.values()]

    for long_only in (True, False):
        key = "long_only" if long_only else "long_short"
        strategy = VolManagedMomentum(long_only=long_only)
        cpcv = run_cpcv(strategy, data, eligible, rebal)
        cpcv_sharpes = [r["sharpe"] for r in cpcv if r["sharpe"] is not None]
        wf, _ = walk_forward_returns(strategy, data, eligible)
        out[key] = {
            "cpcv_sharpes": cpcv_sharpes,
            "dsr_vs_searched_variants": _dsr_summary(deflated_sharpe_ratio(wf, searched_sharpes)),
            "dsr_vs_cpcv_paths": _dsr_summary(deflated_sharpe_ratio(wf, cpcv_sharpes)),
        }
    return out


def _dsr_summary(d: dict) -> dict:
    return {k: d[k] for k in ("dsr", "sharpe_annualized", "sr0_annualized", "n_trials", "n_obs", "skew", "kurtosis")}


# --- charts ------------------------------------------------------------------------------

def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="semibold", pad=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(which="both", colors=INK_2, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _growth_chart(path: Path, series: list[tuple[str, pd.Series, str]], title: str, note: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    for label, returns, color in series:
        growth = (1.0 + returns).cumprod()
        ax.plot(growth.index, growth.to_numpy(), color=color, linewidth=2, label=label)
        ax.annotate(f"{label}  {growth.iloc[-1]:.1f}x", (growth.index[-1], growth.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 3.0, 5.0)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    _style(ax, title, "Growth of $1 (log scale)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    fig.text(0.01, 0.01, note, fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 0.88, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _drawdown_chart(path: Path, returns: pd.Series, bench: pd.Series) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor=SURFACE)
    for label, r, color in (("Strategy", returns, BLUE), ("Universe equal-weight", bench, GRAY)):
        eq = (1.0 + r).cumprod()
        dd = eq / eq.cummax() - 1.0
        ax.plot(dd.index, dd.to_numpy() * 100, color=color, linewidth=2 if color == BLUE else 1.5, label=label)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _style(ax, "Drawdown from peak (walk-forward, long-only)", "")
    ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _window_chart(path: Path, windows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [w["window"] for w in windows]
    strat = [w["strategy"]["sharpe"] or 0.0 for w in windows]
    bench = [w["universe_ew"]["sharpe"] or 0.0 for w in windows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURFACE)
    ax.bar(x - 0.19, strat, width=0.36, color=BLUE, label="Strategy (long-only)")
    ax.bar(x + 0.19, bench, width=0.36, color=GRAY, label="Universe equal-weight")
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(x, labels)
    _style(ax, "Sharpe ratio in each 3-year walk-forward window", "Sharpe (annualized)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _cpcv_chart(path: Path, sharpes: list[float], observed: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.4), facecolor=SURFACE)
    ax.hist(sharpes, bins=14, color=BLUE, edgecolor=SURFACE, linewidth=2)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.axvline(observed, color=ORANGE, linewidth=2)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.annotate(f"walk-forward Sharpe {observed:.2f}", (observed, ax.get_ylim()[1] * 0.92),
                xytext=(-6, 0), textcoords="offset points", ha="right", fontsize=8, color=INK)
    _style(ax, f"Out-of-sample Sharpe across {len(sharpes)} CPCV paths (long-only)", "Paths")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- writing ---------------------------------------------------------------------------

def _pct(v):
    return "n/a" if v is None else f"{v * 100:+.1f}%"


def _num(v):
    return "n/a" if v is None else f"{v:.2f}"


def _row(name: str, m: dict) -> str:
    return (f"| {name} | {_pct(m['cagr'])} | {_pct(m['vol']).lstrip('+')} | {_num(m['sharpe'])} | "
            f"{_num(m['sortino'])} | {_pct(-m['max_drawdown'])} |")


def write_markdown(pit: dict, static: dict, matched: dict, path: Path) -> None:
    lo, ls = pit["variants"]["long_only"], pit["variants"]["long_short"]
    of = pit["overfitting"]
    head = "| Series | CAGR | Vol | Sharpe | Sortino | Max DD |\n|---|---|---|---|---|---|"
    wf = lo["walk_forward"]
    lines = [
        "# Results",
        "",
        f"Generated {date.today()} by `python main.py report` from the local Sharadar archive. "
        "Every figure here is written by `research/report.py`; none is typed by hand.",
        "",
        f"Universe: the top {config_universe_size()} US common stocks by trailing 3-month dollar volume, "
        "re-ranked at every month-end using only data available that day (delisted names included), "
        "then filtered to price >= $5 and 20-day ADV >= $5M. Costs: "
        f"{config.VMM_COST_BPS_PER_SIDE:g} bps per side on actual turnover. "
        f"Returns are the stitched walk-forward out-of-sample path, {wf['start']} to {wf['end']}.",
        "",
        "## Headline (walk-forward, after costs)",
        "",
        head,
        _row("Vol-managed momentum, long-only", lo["walk_forward"]),
        _row("Vol-managed momentum, long/short", ls["walk_forward"]),
        _row("Universe equal-weight (no costs)", lo["universe_ew"]),
        "",
        "![growth](equity_curve.png)",
        "",
        "![drawdown](drawdown.png)",
        "",
        "## Walk-forward windows (long-only)",
        "",
        "| Window | Strategy Sharpe | Universe EW Sharpe | Strategy CAGR | Strategy max DD | Beat EW? |",
        "|---|---|---|---|---|---|",
    ]
    beats = 0
    for w in lo["windows"]:
        s, b = w["strategy"], w["universe_ew"]
        beat = (s["sharpe"] or -9) > (b["sharpe"] or -9)
        beats += beat
        lines.append(f"| {w['window']} | {_num(s['sharpe'])} | {_num(b['sharpe'])} | {_pct(s['cagr'])} | "
                     f"{_pct(-s['max_drawdown'])} | {'yes' if beat else 'no'} |")
    positive = sum((w["strategy"]["sharpe"] or 0) > 0 for w in lo["windows"])
    lines += [
        "",
        f"Positive Sharpe in {positive}/{len(lo['windows'])} windows; higher Sharpe than the "
        f"equal-weight universe in {beats}/{len(lo['windows'])}.",
        "",
        "![windows](walkforward_windows.png)",
        "",
        "## The universe look-ahead, measured",
        "",
        "Before this fix the universe was the 1,500 names with the highest *whole-life* median "
        "dollar volume. That quietly selects companies for what they later became. Same strategy, "
        "same code, only the universe rule changed.",
        "",
        "The two rules also give different breadth: on a typical month-end the static list had only "
        f"{static['median_eligible_names']} eligible names (many of its picks had not listed or grown "
        f"liquid yet), the point-in-time top {pit['top_n']:,} had {pit['median_eligible_names']}. So a "
        f"third row uses a point-in-time top {matched['top_n']:,} ({matched['median_eligible_names']} "
        "eligible names), sized to match the static list, to separate the look-ahead from the breadth.",
        "",
        "| Universe | Median eligible names | Long-only Sharpe | Long-only CAGR | Long/short Sharpe | Long/short CAGR |",
        "|---|---|---|---|---|---|",
        *[f"| {label} | {u['median_eligible_names']} | {_num(u['variants']['long_only']['walk_forward']['sharpe'])} | "
          f"{_pct(u['variants']['long_only']['walk_forward']['cagr'])} | "
          f"{_num(u['variants']['long_short']['walk_forward']['sharpe'])} | "
          f"{_pct(u['variants']['long_short']['walk_forward']['cagr'])} |"
          for label, u in (("Static top 1,500 (look-ahead)", static),
                           (f"Point-in-time top {matched['top_n']:,} (size-matched)", matched),
                           (f"Point-in-time top {pit['top_n']:,} (default)", pit))],
        "",
        "![bias](universe_bias.png)",
        "",
        "## Cost sensitivity (walk-forward)",
        "",
        "| Cost per side | Long-only Sharpe | Long-only CAGR | Long/short Sharpe | Long/short CAGR |",
        "|---|---|---|---|---|",
    ]
    for bps in lo["cost_sensitivity"]:
        a, b = lo["cost_sensitivity"][bps], ls["cost_sensitivity"][bps]
        lines.append(f"| {bps} bps | {_num(a['sharpe'])} | {_pct(a['cagr'])} | {_num(b['sharpe'])} | {_pct(b['cagr'])} |")
    lines += [
        "",
        f"Annual one-way turnover of the long book: ~{lo['annual_turnover'] * 100:.0f}%.",
        "",
        "## Is it luck? CPCV, Deflated Sharpe, PBO",
        "",
        "| | Long-only | Long/short |",
        "|---|---|---|",
    ]
    for label, fn in [
        ("CPCV paths with positive Sharpe",
         lambda k: f"{sum(s > 0 for s in of[k]['cpcv_sharpes'])}/{len(of[k]['cpcv_sharpes'])}"),
        ("CPCV median path Sharpe", lambda k: _num(float(np.median(of[k]["cpcv_sharpes"])))),
        ("Deflated Sharpe vs the 6 configurations searched",
         lambda k: f"{of[k]['dsr_vs_searched_variants']['dsr'] * 100:.0f}% "
                   f"(bar {of[k]['dsr_vs_searched_variants']['sr0_annualized']:.2f})"),
        ("Deflated Sharpe vs CPCV path dispersion (stricter)",
         lambda k: f"{of[k]['dsr_vs_cpcv_paths']['dsr'] * 100:.0f}% "
                   f"(bar {of[k]['dsr_vs_cpcv_paths']['sr0_annualized']:.2f})"),
    ]:
        lines.append(f"| {label} | {fn('long_only')} | {fn('long_short')} |")
    lines += [
        "",
        f"Probability of Backtest Overfitting across the {of['pbo']['n_variants']} pre-registered "
        f"momentum variants ({of['pbo']['n_splits']} CSCV splits): **{of['pbo']['pbo'] * 100:.0f}%**.",
        "",
        "The Deflated Sharpe is the probability that the true Sharpe beats the best you would expect "
        "from luck alone given the search. Read 95%+ as convincing, under 50% as \"can't tell from noise.\" "
        "The stricter row treats every CPCV path as a separate attempt, which overstates the search "
        "but shows how much of the result is sampling noise.",
        "",
        "What the high first-row number does *not* say: it tests the Sharpe against a luck bar near "
        "zero, not against the market. Holding the equal-weight universe clears that bar too. The "
        "question that matters for deployment is the headline table: after costs the strategy's "
        "Sharpe is only slightly above the universe's, with deeper drawdowns.",
        "",
        "![cpcv](cpcv_sharpes.png)",
        "",
    ]
    path.write_text("\n".join(lines))


def config_universe_size() -> str:
    from factors.run_factor_eval import SHARADAR_MAX_SYMBOLS
    return f"{SHARADAR_MAX_SYMBOLS:,}"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if not isinstance(v, pd.Series)}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main(out_dir: Path = REPORT_DIR) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    pit = evaluate_universe("pit", full=True)
    static = evaluate_universe("static", full=False)
    matched = evaluate_universe("pit", full=False, top_n=MATCHED_TOP_N)
    lo = pit["variants"]["long_only"]

    _growth_chart(out_dir / "equity_curve.png",
                  [("Strategy (long-only)", lo["returns"], BLUE), ("Universe equal-weight", lo["benchmark"], GRAY)],
                  "Vol-managed momentum vs its own universe, walk-forward out of sample",
                  "Strategy after 15 bps/side costs; benchmark is the equal-weight return of every eligible name, "
                  "no costs. Point-in-time universe, delisted names included.")
    _growth_chart(out_dir / "universe_bias.png",
                  [("Point-in-time universe", lo["returns"], BLUE),
                   ("Static universe (look-ahead)", static["variants"]["long_only"]["returns"], ORANGE)],
                  "Same strategy, two universe rules (long-only, walk-forward)",
                  "Static = top names by whole-life median dollar volume, which uses future data. "
                  "Point-in-time = re-ranked each month-end on trailing volume.")
    _drawdown_chart(out_dir / "drawdown.png", lo["returns"], lo["benchmark"])
    _window_chart(out_dir / "walkforward_windows.png", lo["windows"])
    _cpcv_chart(out_dir / "cpcv_sharpes.png", pit["overfitting"]["long_only"]["cpcv_sharpes"],
                lo["walk_forward"]["sharpe"])

    write_markdown(pit, static, matched, out_dir / "RESULTS.md")
    (out_dir / "results.json").write_text(json.dumps(
        {"generated": str(date.today()), "pit": _jsonable(pit), "static": _jsonable(static),
         "pit_size_matched": _jsonable(matched)}, indent=2))
    log.info("Wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    sys.exit(main())
