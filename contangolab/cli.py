"""ContangoLab command line.

    python -m contangolab evaluate cn_futures       # every signal for a market; each run logged as a trial
    python -m contangolab combine cn_futures        # walk-forward signal combination vs the best single signal
    python -m contangolab design cn_futures         # the 32-configuration portfolio grid + walk-forward selector
    python -m contangolab trials                    # every logged trial with Benjamini-Hochberg q-values
    python -m contangolab study cross_market        # regenerate one study's tables and charts in results/

Markets: ``cn_futures`` (71 China commodity futures) and ``us_futures`` (24 CME Group commodities).
Every evaluation appends to ``results/trials.jsonl``, so the multiple-testing correction always
counts every test ever run, including the failures.
"""

import argparse
import importlib
import logging
import sys

import pandas as pd

from contangolab import config

log = logging.getLogger("contangolab")

STUDIES = ("cross_market",)


def cmd_evaluate(args) -> int:
    """Evaluate every registered signal for an asset class, log each trial, print the corrected table."""
    from contangolab.engine.dataset import load_dataset
    from contangolab.engine.evaluate import evaluate
    from contangolab.engine.trials import PROTOCOL, correction_table, load_trials, log_trial
    from contangolab.signals.registry import signals

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


def cmd_combine(args) -> int:
    """Fit the P2 combinations walk-forward, compare OOS with the single signals, log each as a trial."""
    from contangolab.engine.combine import PROTOCOL, run
    from contangolab.engine.dataset import load_dataset
    from contangolab.engine.trials import log_trial
    from contangolab.signals.registry import signals

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
    dest = config.OUTPUT / f"combine_{args.asset_class}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({m: mo["net"] for m, mo in out["monthly"].items()}).to_csv(dest)
    log.info("Monthly net returns per method -> %s", dest)
    return 0


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
    from contangolab.engine.dataset import load_dataset
    from contangolab.engine.design import PROTOCOL, TRACKS, prepare, run_track, trial_records
    from contangolab.engine.trials import log_trial
    from contangolab.signals.registry import signals

    tracks = [t for t in TRACKS.values() if t.asset_class == args.asset_class]
    if not tracks:
        log.error("No P3 tracks for %s.", args.asset_class)
        return 1
    log.info("Loading %s and preparing features, XGBoost scores and single-signal histories...", args.asset_class)
    ds = load_dataset(args.asset_class)
    ctx = prepare(ds, signals(args.asset_class), weekly=any("weekly" in t.turnover for t in tracks))
    out_dir = config.OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)
    for track in tracks:
        log.info("Running track %s...", track.name)
        out = run_track(ctx, track)
        if not args.dry_run:
            for rec in trial_records(out, track, ds):
                log_trial(rec, protocol=PROTOCOL, note=args.note)
        _print_design(out)
        out["matrix"].to_csv(out_dir / f"design_{track.name}.csv")
        out["selector"].rename("selector").to_csv(out_dir / f"design_{track.name}_selector.csv")
    return 0


def cmd_trials(args) -> int:
    """Print the latest trial per signal with one-sided p-values and BH q-values over the whole log."""
    from contangolab.engine.trials import correction_table, load_trials

    t = correction_table(load_trials(), protocol=None)
    if args.market:
        t = t[t["asset_class"] == args.market]
    cols = ["signal", "asset_class", "family", "protocol", "months", "sharpe", "t_stat", "q_all", "survivor"]
    print(f"{len(load_trials())} trials logged; latest per signal, q over all of them:\n")
    print(t[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


def cmd_study(args) -> int:
    """Run a study module's ``main()`` (tables to stdout, charts to results/)."""
    return importlib.import_module(f"contangolab.studies.{args.name}").main() or 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="contangolab", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def logged(name: str, func, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("asset_class", choices=("cn_futures", "us_futures"))
        p.add_argument("--note", default="", help="free-text note stored with each trial")
        p.add_argument("--dry-run", action="store_true", help="run without logging trials")
        p.set_defaults(func=func)
        return p

    ev = logged("evaluate", cmd_evaluate, "evaluate registered signals; every run is logged to results/trials.jsonl")
    ev.add_argument("--family", default=None, help="only signals of this family")
    ev.add_argument("--signal", nargs="*", default=None, help="only these signals")
    ev.add_argument("--protocol", default="b11-p1b-v1",
                    help="protocol tag logged with each trial (a non-P1b tag corrects over all protocols)")
    logged("combine", cmd_combine, "walk-forward signal combination (ew / IC-weighted / ridge / XGBoost)")
    logged("design", cmd_design, "every pre-registered portfolio configuration + the walk-forward selector")
    tr = sub.add_parser("trials", help="the trial log with multiple-testing corrections")
    tr.add_argument("--market", default=None, help="only this market (cn_futures, us_futures)")
    tr.set_defaults(func=cmd_trials)
    st = sub.add_parser("study", help="regenerate one study's tables and charts")
    st.add_argument("name", choices=STUDIES)
    st.set_defaults(func=cmd_study)
    return parser


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = build_parser().parse_args(argv)
    config.OUTPUT.mkdir(parents=True, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
