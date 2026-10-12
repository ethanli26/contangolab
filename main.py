"""pit-backtester — single integrated entry point.

One CLI over the point-in-time research/validation pipeline. Each subcommand is a
thin wrapper that reuses the existing modules — no strategy logic, guard, or result
is changed here:

  factors   — factor IC evaluation scorecard (price + point-in-time fundamentals,
              large vs broad universe, survivorship-free Sharadar run).
  altdata   — the insider (SF2) / institutional (SF3) factor gauntlet: IC bar,
              walk-forward consistency, and momentum-diversification check.
  strategy  — a registered portfolio strategy's OOS benchmark comparison +
              walk-forward validation (e.g. vol_managed_momentum).
  paper     — one monthly paper rebalance for a portfolio strategy, through the
              existing DU-account guard + DRY_RUN + autonomy gate.
  evaluate  — run every registered signal for an asset class through the shared
              evaluator (core/), logging each trial to research/trials.jsonl.
  alpha     — combine an asset class's signals into one score, walk-forward (ew, IC-weighted,
              ridge, XGBoost), judged against the best single signal (core/alpha.py).
  design    — P3 design space: every pre-registered configuration of score source x
              construction x neutralization x turnover x vol target, plus the walk-forward
              selector that is judged against the simple default (core/design.py).
  allocate  — P5: combine the US long-short book (after borrow/financing) with the China
              futures book, walk-forward weights, and pick the vol dial from the drawdown
              tolerance (core/allocation.py).
  events    — E1: US event-driven signals (earnings surprises, insider clusters, dividend
              initiations) as weekly calendar-time portfolios vs the universe (core/events.py).
  report    — regenerate reports/ (RESULTS.md, results.json, charts): the source of
              every number the README quotes.

config.py is the single config surface (risk params, autonomy mode, DRY_RUN, vol-
target windows). At startup a self-check asserts the paper-safety guards (DU-account
check, DRY_RUN) are intact. Read-only research + the existing paper path; no new
alpha.

    python main.py factors
    python main.py altdata
    python main.py strategy vol_managed_momentum
    python main.py paper vol_managed_momentum --no-broker
    python main.py evaluate cn_futures
    python main.py alpha cn_futures
    python main.py design cn_futures
    python main.py allocate
    python main.py events
    python main.py report
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from backtest.risk_metrics import as_loss  # noqa: E402

log = logging.getLogger("main")


def configure_logging() -> None:
    """Timestamped logs; quiet the chatty libraries."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("yfinance", "ib_async"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _pct(v):
    return f"{v * 100:+.2f}%" if v is not None else "n/a"


def _ratio(v):
    return f"{v:.2f}" if v is not None else "n/a"


def paper_safety_self_check() -> None:
    """Assert the paper-safety guards hold before dispatching any subcommand."""
    from decision.autonomy import is_paper_account  # the DU-account guard

    assert is_paper_account("DU1234567") and not is_paper_account("U1234567") and not is_paper_account(None)
    assert isinstance(config.DRY_RUN, bool)
    print("=== Paper-safety self-check ===")
    print("  DU-account guard   : DU accepted, U/None rejected  OK")
    print(f"  Paper safety       : DRY_RUN={config.DRY_RUN}, AUTONOMY_MODE={config.AUTONOMY_MODE}")
    print()


# --- Subcommands ------------------------------------------------------------------

def cmd_factors(args) -> int:
    """Factor-evaluation harness scorecard (price + point-in-time fundamentals), unchanged."""
    from factors.run_factor_eval import main as run_factor_main
    return run_factor_main()


def cmd_altdata(args) -> int:
    """Insider (SF2) / institutional (SF3) factor gauntlet, unchanged."""
    from factors.run_altdata_eval import main as run_altdata_main
    return run_altdata_main()


def cmd_strategy(args) -> int:
    """Run a portfolio strategy's backtest + benchmark + walk-forward, headline first."""
    from backtest import walkforward
    from factors.evaluate import _rebalance_dates
    from factors.run_factor_eval import build_sharadar_factor_data
    from research.combine_train import OOS_FRACTION, SPY_CACHE, _monthly_metrics, spy_monthly_returns, vol_matched_blend
    from strategies.registry import get_portfolio, portfolio_strategies

    name = args.name
    if name not in portfolio_strategies():
        log.error("Unknown portfolio strategy '%s'. Available: %s", name, ", ".join(portfolio_strategies()) or "none")
        return 1
    strategy = get_portfolio(name)()
    variant = "long-only" if strategy.long_only else "long/short"

    log.info("Loading survivorship-free Sharadar universe (offline cache)...")
    data, eligible = build_sharadar_factor_data()
    close = data.close
    rebal = _rebalance_dates(close.index, "M")
    split = rebal[int(len(rebal) * (1.0 - OOS_FRACTION))]

    # OOS (last 30%) strategy returns + passive, after costs.
    oos_ret = strategy.portfolio_returns(data, eligible, train_end=split, start=split)
    m_oos = _monthly_metrics(oos_ret)
    spy_close = pd.read_parquet(SPY_CACHE)["Close"]
    pairs = [(c, n) for c, n in zip(rebal[:-1], rebal[1:]) if c >= split]
    spy_oos = spy_monthly_returns(pairs, spy_close).reindex(oos_ret.index).dropna()
    m_spy = _monthly_metrics(spy_oos)
    m_blend = _monthly_metrics(vol_matched_blend(spy_oos, target_vol=oos_ret.std(ddof=1)))

    # Walk-forward across non-overlapping windows.
    wf = walkforward.run_walkforward(strategy, data, eligible)
    spy_rows = [r for r in wf["windows"] if r["spy"]]
    beat_spy = sum(1 for r in spy_rows if (r["metrics"]["sharpe"] or -9) > (r["spy"]["sharpe"] or -9))
    beats_oos = (m_oos["sharpe"] or -9) > (m_spy["sharpe"] or -9) and (m_oos["sharpe"] or -9) > (m_blend["sharpe"] or -9)
    consistent = spy_rows and beat_spy * 2 >= len(spy_rows)

    # --- HEADLINE FIRST -----------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"HEADLINE — {name} ({variant}), survivorship-free, after costs")
    print("=" * 78)
    print(f"  OOS (last {int(OOS_FRACTION*100)}%): Sharpe {_ratio(m_oos['sharpe'])} vs SPY {_ratio(m_spy['sharpe'])} "
          f"vs vol-matched blend {_ratio(m_blend['sharpe'])}  -> {'BEATS' if beats_oos else 'TRAILS'} passive risk-adjusted.")
    print(f"  Walk-forward: beat SPY on Sharpe in {beat_spy}/{len(spy_rows)} non-overlapping windows "
          f"(SPY cache covers post-2011 windows).")
    print(f"  Crash control: OOS maxDD {_pct(as_loss(m_oos['max_drawdown']))} "
          f"vs SPY {_pct(as_loss(m_spy['max_drawdown']))}.")
    print(f"  => {'CONSISTENT across walk-forward windows.' if consistent else 'NOT consistent across windows — caution.'}")

    # --- details ------------------------------------------------------------------
    print(f"\n=== OOS benchmark comparison ({oos_ret.index[0].date()} -> {oos_ret.index[-1].date()}, after costs) ===")
    bench = pd.DataFrame([
        {"series": f"{name} ({variant})", "CAGR": _pct(m_oos["cagr"]), "Sharpe": _ratio(m_oos["sharpe"]),
         "Sortino": _ratio(m_oos["sortino"]), "maxDD": _pct(as_loss(m_oos["max_drawdown"]))},
        {"series": "SPY buy-hold", "CAGR": _pct(m_spy["cagr"]), "Sharpe": _ratio(m_spy["sharpe"]),
         "Sortino": _ratio(m_spy["sortino"]), "maxDD": _pct(as_loss(m_spy["max_drawdown"]))},
        {"series": "vol-matched SPY/cash", "CAGR": _pct(m_blend["cagr"]), "Sharpe": _ratio(m_blend["sharpe"]),
         "Sortino": _ratio(m_blend["sortino"]), "maxDD": _pct(as_loss(m_blend["max_drawdown"]))},
    ])
    print(bench.to_string(index=False))
    print("\n=== Walk-forward (non-overlapping OOS windows, after costs) ===")
    walkforward.print_walkforward(wf)
    walkforward.capacity_report(strategy, data, eligible)

    # --- CPCV / Deflated Sharpe / PBO: is this real, or did the search just get lucky? ---
    from research.cpcv import (
        annualized_sharpe,
        deflated_sharpe_ratio,
        print_cpcv_report,
        probability_of_backtest_overfitting,
        run_cpcv,
    )
    from research.momentum_variants import build_variant_returns
    from research.report import walk_forward_returns

    cpcv_results = run_cpcv(strategy, data, eligible, rebal)
    wf_returns, _ = walk_forward_returns(strategy, data, eligible)
    variant_returns, _, _, _ = build_variant_returns(data, eligible)
    # The DSR's "trials" are the configurations that were actually searched: the five
    # pre-registered momentum variants plus the long-only version chosen for paper.
    searched = {**variant_returns,
                "long_only_vol_managed": type(strategy)(long_only=True).portfolio_returns(data, eligible)}
    dsr_result = deflated_sharpe_ratio(wf_returns, [annualized_sharpe(s) for s in searched.values()])
    pbo_result = None
    try:
        pbo_result = probability_of_backtest_overfitting(variant_returns)
    except ValueError as error:
        log.warning("Skipping PBO: %s", error)
    print_cpcv_report(name, cpcv_results, dsr_result, pbo_result)
    return 0


def cmd_evaluate(args) -> int:
    """Evaluate every registered signal for an asset class, log each trial, print the corrected table."""
    from core.dataset import load_dataset
    from core.evaluate import evaluate
    from core.signals import signals
    from core.trials import PROTOCOL, correction_table, load_trials, log_trial

    specs = [s for s in signals(args.asset_class, args.family) if not args.signal or s.name in args.signal]
    if not specs:
        log.error("No registered signals match asset class %s / family %s.", args.asset_class, args.family)
        return 1
    log.info("Loading %s dataset...", args.asset_class)
    ds = load_dataset(args.asset_class)
    for spec in specs:
        result, _ = evaluate(spec, ds)
        if not args.dry_run:
            log_trial(result, protocol=args.protocol, note=args.note)
        log.info("%-28s IC %+.4f  net Sharpe %+.2f  t %+.2f  (%d months)", spec.name, result["mean_ic"] or 0,
                 result["sharpe"] or 0, result["t_stat"] or 0, result["months"])
    if args.dry_run:
        return 0
    table = correction_table(load_trials(), protocol=PROTOCOL if args.protocol == PROTOCOL else None)
    table = table[table["asset_class"] == args.asset_class]
    cols = ["signal", "family", "months", "mean_ic", "ic_t", "ann_return", "sharpe", "t_stat", "q_family", "q_all",
            "survivor"]
    print(f"\n=== {args.asset_class}: latest trial per signal, protocol {args.protocol} "
          f"(q = Benjamini-Hochberg over every logged signal) ===")
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


def cmd_alpha(args) -> int:
    """Fit the P2 combinations walk-forward, compare OOS with the single signals, log each as a trial."""
    from core.alpha import PROTOCOL, run
    from core.dataset import load_dataset
    from core.signals import signals
    from core.trials import log_trial

    specs = signals(args.asset_class)
    log.info("Loading %s dataset (%d signals in the pool)...", args.asset_class, len(specs))
    out = run(load_dataset(args.asset_class), specs)
    if not args.dry_run:
        for res in out["results"].values():
            log_trial(res, protocol=PROTOCOL, note=args.note)
    first, last, n = out["oos"]
    print(f"\n=== {args.asset_class}: alpha model, protocol {PROTOCOL} ({out['combo_kind']} portfolio) ===")
    print(f"OOS {first.date()} .. {last.date()} ({n} months), halves split at {out['half'].date()}")
    best = out["best_names"]
    print(f"best single signal: OOS {best['oos']}, first half {best['h1']}, second half {best['h2']}\n")
    print("net Sharpe")
    print(out["table"].to_string(float_format=lambda v: f"{v:+.2f}"))
    print("\nsingle signals, same months")
    print(out["singles"].sort_values("oos", ascending=False).to_string(float_format=lambda v: f"{v:+.2f}"))
    v = out["verdict"]
    print("\npassed the bar: " + ", ".join(f"{m} {'yes' if ok else 'no'}" for m, ok in v["passed"].items()))
    print(f"VERDICT: {v['text']}")
    dest = Path(__file__).resolve().parent / "research" / "output" / f"alpha_{args.asset_class}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({m: mo["net"] for m, mo in out["monthly"].items()}).to_csv(dest)
    log.info("Monthly net returns per method -> %s", dest)
    return 0


def _sectors() -> pd.Series:
    """Ticker -> Sharadar sector (current classification; a mild look-ahead noted in the P3 protocol)."""
    from data.sharadar_provider import BULK_DIR

    t = pd.read_parquet(BULK_DIR / "TICKERS.parquet", columns=["table", "ticker", "sector"])
    t = t[t["table"] == "SEP"].dropna(subset=["sector"]).drop_duplicates("ticker", keep="last")
    return t.set_index("ticker")["sector"]


def _print_design(out: dict) -> None:
    def fmt(st: dict) -> str:
        return (f"Sharpe {st['sharpe']:+.2f}  CAGR {st['cagr']:+.1%}  vol {st['ann_vol']:.1%}  maxDD {st['max_dd']:.1%}"
                if st.get("sharpe") is not None else "n/a")

    first, last, n = out["window"]
    print(f"\n=== {out['track']}: {out['configs']} configurations, protocol b11-p3-v1 ===")
    print(f"selector OOS window {first.date()} .. {last.date()} ({n} months)")
    cols = ["sharpe", "cagr", "ann_vol", "max_dd", "avg_turnover", "months"]
    print("\ntop 10 configurations by full-history net Sharpe (hindsight, for reference only):")
    print(out["table"][cols].astype(float).sort_values("sharpe", ascending=False).head(10)
          .to_string(float_format=lambda v: f"{v:+.3f}"))
    print(f"\nsimple default  {out['simple']}\n  window: {fmt(out['simple_stats'])}")
    print("  halves: " + " | ".join(fmt(h) for h in out["simple_halves"]))
    print(f"selector\n  window: {fmt(out['selector_stats'])}")
    print("  halves: " + " | ".join(fmt(h) for h in out["selector_halves"]))
    if "benchmark_stats" in out:
        print(f"equal-weight PIT universe\n  window: {fmt(out['benchmark_stats'])}")
        print("  halves: " + " | ".join(fmt(h) for h in out["benchmark_halves"]))
        print(f"  selector beats the universe in both halves: {out['edge_vs_benchmark']}")
    pbo = out["pbo"]
    print(f"PBO over the grid: {pbo:.2f}" if pbo is not None else "PBO: n/a")
    d = out["dsr"]
    print(f"hindsight best {out['hindsight_best']}: Sharpe {d['sharpe_annualized']:+.2f}, luck bar "
          f"{d['sr0_annualized']:.2f} over {d['n_trials']} configs, DSR {d['dsr']:.2f}")
    print("selector picks: " + "; ".join(f"{t.date()} {c.split('|', 1)[1]}" for t, c in out["picks"]))
    v = out["verdict"]
    print(f"VERDICT: adopt {v['adopt']} ({v['reason']})")


def cmd_design(args) -> int:
    """Run every pre-registered P3 configuration for an asset class's tracks; log each; apply the bar."""
    from backtest.walkforward import universe_equal_weight_returns
    from core.dataset import load_dataset
    from core.design import PROTOCOL, TRACKS, prepare, run_track, trial_records
    from core.signals import signals
    from core.trials import log_trial

    tracks = [t for t in TRACKS.values() if t.asset_class == args.asset_class]
    if not tracks:
        log.error("No P3 tracks for %s.", args.asset_class)
        return 1
    log.info("Loading %s and preparing features, XGBoost scores and single-signal histories...", args.asset_class)
    ds = load_dataset(args.asset_class)
    ctx = prepare(ds, signals(args.asset_class), weekly=any("weekly" in t.turnover for t in tracks),
                  sectors=_sectors() if args.asset_class == "us_equity" else None)
    out_dir = Path(__file__).resolve().parent / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    for track in tracks:
        log.info("Running track %s...", track.name)
        bench = universe_equal_weight_returns(ds.data, ds.eligible) if track.name == "us_lo" else None
        out = run_track(ctx, track, bench)
        if not args.dry_run:
            for rec in trial_records(out, track, ds):
                log_trial(rec, protocol=PROTOCOL, note=args.note)
        _print_design(out)
        out["matrix"].to_csv(out_dir / f"design_{track.name}.csv")
        out["selector"].rename("selector").to_csv(out_dir / f"design_{track.name}_selector.csv")
    return 0


def cmd_allocate(args) -> int:
    """P5: build both books, allocate walk-forward, apply the bars, choose the vol dial; log each as a trial."""
    from core import allocation as al
    from core.dataset import load_dataset
    from core.design import TRACKS, halves, prepare, run_config, run_track, scores_on_grid, simple_default, stats
    from core.portfolio import to_monthly
    from core.signals import signals
    from core.trials import log_trial

    log.info("US long-short book: P3 selector with carrying costs (borrow %.0f / %.0f bps, financing %.0f bps)...",
             al.BORROW_BPS, al.BORROW_STRESS_BPS, al.FINANCING_BPS)
    us = load_dataset("us_equity")
    ctx = prepare(us, signals("us_equity"), sectors=_sectors())
    us_ls = {bps: run_track(ctx, TRACKS["us_ls"], adjust=al.carry_costs(bps, al.FINANCING_BPS))
             for bps in (al.BORROW_BPS, al.BORROW_STRESS_BPS)}
    log.info("China futures book: P3 default...")
    fut = load_dataset("cn_futures")
    fctx = prepare(fut, signals("cn_futures"))
    cfg = simple_default(TRACKS["fut"])
    fut_book = to_monthly(run_config(fctx, cfg, TRACKS["fut"], scores_on_grid(fctx, cfg.source, "monthly"))["net"])

    out_dir = Path(__file__).resolve().parent / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    for bps, track_out in us_ls.items():
        stress = bps != al.BORROW_BPS
        res = al.run({"us_ls": track_out["selector"], "fut": fut_book})
        w = res["window"]
        fmt = lambda st: (f"Sharpe {st['sharpe']:+.2f}  CAGR {st['cagr']:+.1%}  vol {st['ann_vol']:.1%}  "  # noqa: E731
                          f"maxDD {st['max_dd']:.1%}")
        print(f"\n=== P5 allocation, protocol {al.PROTOCOL}, borrow {bps:.0f} bps"
              f"{' (STRESS, reported only)' if stress else ''} ===")
        print(f"joint OOS window {w[0].date()} .. {w[-1].date()} ({len(w)} months); correlation of books {res['corr']:+.2f}")
        last = track_out["picks"][-1][1].split("|", 1)[1] if track_out["picks"] else "n/a"
        print(f"US long-short selector (latest pick: {last}), whole history after costs: {fmt(track_out['selector_stats'])}")
        def by_half(hs):
            return " | ".join(f"{x['sharpe']:+.2f}" for x in hs)
        for b, st in res["book_stats"].items():
            print(f"  book {b:7s} window: {fmt(st)}   halves: {by_half(res['book_halves'][b])}")
        for m, st in res["combo_stats"].items():
            print(f"  {m:13s} window: {fmt(st)}   halves: {by_half(res['combo_halves'][m])}")
        print(f"bar 1 (risk parity beats each book in both halves): {res['bar1']}")
        print(f"bar 2 (max_sharpe beats risk parity in both halves): {res['bar2']} -> adopted {res['adopted']}")
        print("vol dial (adopted combination, walk-forward leverage, cap 4x):")
        for t, d in res["dial"].items():
            print(f"  target {t:.0%}: {fmt(d)}  worst year {d['worst_year']:+.1%} ({d['worst_year_at']})  "
                  f"avg lev {d['avg_lev']:.2f}x (max {d['max_lev']:.2f}x)  95th-pct 10y maxDD {d['dd95_10y']:.1%}")
        ch = res["chosen_target"]
        print(f"VERDICT: chosen vol target {ch:.0%}" if ch else "VERDICT: no target keeps the 95th-pct 10y maxDD <= 35%")
        if stress:
            continue
        pd.DataFrame({**{b: res["books"][b] for b in res["books"]},
                      **{m: c["ret"] for m, c in res["combos"].items()},
                      **{f"dial_{t:.0%}": d["series"] for t, d in res["dial"].items()}}).to_csv(out_dir / "allocation.csv")
        if not args.dry_run:
            for name, series in [*((m, c["ret"]) for m, c in res["combos"].items()),
                                 *((f"{res['adopted']}_vt{int(t * 100)}", d["series"]) for t, d in res["dial"].items())]:
                h1, h2 = halves(series, w)
                log_trial({"signal": f"p5|{name}", "asset_class": "multi", "family": "p5_allocation",
                           "kind": "allocation", "sign": 1, "published": None, "start": str(w[0].date()),
                           "end": str(w[-1].date()), **stats(series), "h1": h1, "h2": h2},
                          protocol=al.PROTOCOL, note=args.note)
    return 0


def cmd_events(args) -> int:
    """E1: build the event tables, run each pre-registered event signal, log trials, chart the drift."""
    from core import events as ev
    from core.dataset import load_dataset
    from core.portfolio import simulate
    from core.trials import correction_table, load_trials, log_trial
    from data import us_events as ue
    from research.event_study import BLUE, GRAY, ORANGE, car_chart

    ds = load_dataset("us_equity")
    close, tickers = ds.close, list(ds.close.columns)
    log.info("Building event tables from the Sharadar archive...")
    sue_e = ev.only_eligible(ue.sue(tickers), ds)
    ear_e = ev.only_eligible(ue.ear(ue.earnings_dates(tickers), close, ds.eligible), ds)
    sue10, ear10 = ev.classify(sue_e, 0.10), ev.classify(ear_e, 0.10)
    dbl = ev.only_eligible(ev.double_surprise(ev.classify(sue_e, 0.20), ev.classify(ear_e, 0.20)), ds)
    ins = ev.only_eligible(ue.insider_clusters(tickers), ds)
    div = ev.only_eligible(ue.dividend_initiations(close), ds)
    log.info("Events in the universe: SUE %d, EAR %d, double %d, insider clusters %d, dividend initiations %d",
             len(sue_e), len(ear_e), len(dbl), len(ins), len(div))

    grid = ev.weekly_grid(ds)
    bench = simulate(ev.universe_targets(ds, grid), close.pct_change(fill_method=None), 0.0)["net"]
    specs = [("sue", sue10[sue10["top"]], sue10[sue10["bottom"]], 1989),
             ("ear", ear10[ear10["top"]], ear10[ear10["bottom"]], 1996),
             ("double_surprise", dbl, None, 2008),
             ("insider_cluster", ins, None, 2001),
             ("div_initiation", div, None, 1995)]
    results = []
    for name, long, short, published in specs:
        log.info("Running %s (%d long events)...", name, len(long))
        res = ev.run_signal(name, long, ds, grid, bench, published, short)["result"]
        results.append(res)
        if not args.dry_run:
            log_trial(res, protocol=ev.PROTOCOL, note=args.note)

    def f(v, fmt):
        return format(v, fmt) if v is not None else "n/a"
    print(f"\n=== E1 US event signals, protocol {ev.PROTOCOL}: long leg net of 15 bps/side minus the "
          "equal-weight PIT universe, weekly ===")
    print(f"{'signal':17s} {'from':>7s} {'names':>6s} {'excess/yr':>9s} {'IR':>6s} {'t':>6s} {'h1 IR':>6s} {'h2 IR':>6s} "
          f"{'post-pub IR':>11s} {'L-S IR':>7s} {'turnover/mo':>11s}")
    for r in results:
        ls = r.get("long_short", {})
        print(f"{r['signal']:17s} {r['start'][:7]:>7s} {r['avg_names']:6.0f} {f(r['ann_return'], '+.1%'):>9s} "
              f"{f(r['sharpe'], '+.2f'):>6s} {f(r['t_stat'], '+.2f'):>6s} {f(r['h1']['sharpe'], '+.2f'):>6s} "
              f"{f(r['h2']['sharpe'], '+.2f'):>6s} {f(r.get('post_publication', {}).get('sharpe'), '+.2f'):>11s} "
              f"{f(ls.get('sharpe'), '+.2f'):>7s} {r['avg_turnover']:11.2f}")
    if not args.dry_run:
        table = correction_table(load_trials(), protocol=None)
        table = table[table["family"] == ev.FAMILY]
        print("\nmultiple-testing correction over the whole registry (latest run of every signal/config):")
        print(table[["signal", "months", "t_stat", "p", "q_all", "survivor"]].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))

    level = ue.universe_index(close, ds.eligible)
    paths = {k: ev.car_paths(e, ds, level) for k, e in [
        ("sue_top", sue10[sue10["top"]]), ("sue_bottom", sue10[sue10["bottom"]]),
        ("ear_top", ear10[ear10["top"]]), ("ear_bottom", ear10[ear10["bottom"]]),
        ("double", dbl), ("insider", ins), ("div", div)]}
    out = Path(__file__).resolve().parent / "reports" / "events_car.png"
    car_chart(out, [
        ("Earnings surprise (SUE)", [("top decile", paths["sue_top"], BLUE), ("bottom decile", paths["sue_bottom"], ORANGE)]),
        ("Announcement return (EAR)", [("top decile", paths["ear_top"], BLUE),
                                       ("bottom decile", paths["ear_bottom"], ORANGE)]),
        ("Long-only events", [("double surprise", paths["double"], BLUE), ("insider cluster", paths["insider"], ORANGE),
                              ("dividend initiation", paths["div"], GRAY)]),
    ], "Point-in-time top-1500 US stocks; day 0 = the day the news is public; before costs. Source: Sharadar.")
    log.info("Drift chart -> %s", out)
    return 0


def cmd_report(args) -> int:
    """Regenerate reports/ (RESULTS.md, results.json, charts) from the local archive."""
    from research.report import main as report_main
    return report_main()


def cmd_paper(args) -> int:
    """Monthly paper rebalance for a portfolio strategy, through the existing safety path."""
    from decision.run_paper_strategy import run_paper_cycle
    return run_paper_cycle(args.name, use_broker=not args.no_broker, equity_override=args.equity)


def build_parser() -> argparse.ArgumentParser:
    """CLI with the ten subcommands."""
    parser = argparse.ArgumentParser(prog="main.py", description="pit-backtester — integrated entry point.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, func, help_text in [
        ("factors", cmd_factors, "factor IC evaluation scorecard (price + point-in-time fundamentals)"),
        ("altdata", cmd_altdata, "insider/institutional factor gauntlet (SF2/SF3 vs momentum)"),
        ("report", cmd_report, "regenerate reports/: results tables + charts from the local archive"),
    ]:
        subparsers.add_parser(name, help=help_text).set_defaults(func=func)
    # Portfolio-strategy backtest + walk-forward (e.g. vol_managed_momentum).
    strat_parser = subparsers.add_parser(
        "strategy", help="run a portfolio strategy's backtest + benchmark + walk-forward")
    strat_parser.add_argument("name", help="registered portfolio strategy (e.g. vol_managed_momentum)")
    strat_parser.set_defaults(func=cmd_strategy)
    # Evaluate registered signals on an asset class and log every trial (core/).
    eval_parser = subparsers.add_parser(
        "evaluate", help="evaluate registered signals for an asset class; every run is logged to research/trials.jsonl")
    eval_parser.add_argument("asset_class", help="us_equity, cn_futures or us_futures")
    eval_parser.add_argument("--family", default=None, help="only signals of this family")
    eval_parser.add_argument("--signal", nargs="*", default=None, help="only these signals")
    eval_parser.add_argument("--note", default="", help="free-text note stored with each trial")
    eval_parser.add_argument("--dry-run", action="store_true", help="evaluate without logging trials")
    eval_parser.add_argument("--protocol", default="b11-p1b-v1",
                             help="protocol tag logged with each trial (a non-P1b tag corrects over all protocols)")
    eval_parser.set_defaults(func=cmd_evaluate)
    # Combine signals into one alpha score, walk-forward (core/alpha.py).
    alpha_parser = subparsers.add_parser(
        "alpha", help="walk-forward signal combination (ew / IC-weighted / ridge / XGBoost) vs the best single signal")
    alpha_parser.add_argument("asset_class", help="us_equity or cn_futures")
    alpha_parser.add_argument("--note", default="", help="free-text note stored with each trial")
    alpha_parser.add_argument("--dry-run", action="store_true", help="run without logging trials")
    alpha_parser.set_defaults(func=cmd_alpha)
    # P3 design space + walk-forward selector (core/design.py).
    design_parser = subparsers.add_parser(
        "design", help="run every pre-registered P3 configuration; judge the walk-forward selector vs the simple default")
    design_parser.add_argument("asset_class", help="us_equity (tracks us_ls, us_lo) or cn_futures (track fut)")
    design_parser.add_argument("--note", default="", help="free-text note stored with each trial")
    design_parser.add_argument("--dry-run", action="store_true", help="run without logging trials")
    design_parser.set_defaults(func=cmd_design)
    # P5 allocation across books + vol dial (core/allocation.py).
    alloc_parser = subparsers.add_parser(
        "allocate", help="P5: combine the US long-short and futures books walk-forward and choose the vol dial")
    alloc_parser.add_argument("--note", default="", help="free-text note stored with each trial")
    alloc_parser.add_argument("--dry-run", action="store_true", help="run without logging trials")
    alloc_parser.set_defaults(func=cmd_allocate)
    # E1 US event-driven signals (core/events.py).
    events_parser = subparsers.add_parser(
        "events", help="E1: US event-driven signals as weekly calendar-time portfolios; logs trials, draws the drift chart")
    events_parser.add_argument("--note", default="", help="free-text note stored with each trial")
    events_parser.add_argument("--dry-run", action="store_true", help="run without logging trials")
    events_parser.set_defaults(func=cmd_events)
    # Monthly paper rebalance for a portfolio strategy (existing DU + DRY_RUN + gate path).
    paper_parser = subparsers.add_parser(
        "paper", help="run one monthly paper rebalance for a portfolio strategy")
    paper_parser.add_argument("name", help="registered portfolio strategy (e.g. vol_managed_momentum)")
    paper_parser.add_argument("--no-broker", action="store_true",
                              help="offline preview: compute target/orders, place nothing, no connection")
    paper_parser.add_argument("--equity", type=float, default=1_000_000.0,
                              help="assumed equity for an offline preview (real cycles read broker NAV)")
    paper_parser.set_defaults(func=cmd_paper)
    return parser


def main(argv=None) -> int:
    """Parse args, run the startup self-check, and dispatch the subcommand."""
    configure_logging()
    args = build_parser().parse_args(argv)
    paper_safety_self_check()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
