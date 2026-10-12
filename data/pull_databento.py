"""Download US futures data from Databento (CME Globex, GLBX.MDP3) into ``data/cache/futures_us/``.

Every job quotes its cost first (a free metadata call) and aborts above its cap, so a pull never
spends past the account's free credit. Output is gitignored: Databento data is licensed, not
redistributable.

  daily        ohlcv-1d, every listed contract (outrights and spreads) of the commodity ROOTS
  daily_extra  ohlcv-1d for EXTRA_ROOTS (rough rice, class III milk, lumber)
  financials   ohlcv-1d for FIN_ROOTS (equity index, Treasuries, FX), a cross-asset reference
  stats        statistics schema for the commodity OUTRIGHTS only (settlement price, open
               interest, cleared volume), one request per root, filtered to those stat types
  ng_intraday  1-minute bars of the first two natural-gas contracts (NG.c.0, NG.c.1), 14:00-18:00
               UTC on every Tue-Fri: brackets the 10:30 ET EIA storage release in both EST and
               EDT, including holiday-shifted days and the occasional 12:00 ET release
  lng_bench    ohlcv-1d for the CME-listed overseas gas benchmarks (Dutch TTF, Platts JKM): the
               export arbitrage that decides whether US LNG cargoes lift

    python -m data.pull_databento daily stats      # run jobs
    python -m data.pull_databento --quote stats    # quote only
"""

import argparse
import os
from pathlib import Path

import pandas as pd

DATASET = "GLBX.MDP3"
ROOTS = ["CL", "HO", "RB", "NG", "BZ",            # energy (NYMEX)
         "GC", "SI", "HG", "PL", "PA",            # metals (COMEX / NYMEX)
         "ZC", "ZW", "KE", "ZS", "ZM", "ZL", "ZO",  # grains and oilseeds (CBOT)
         "LE", "HE", "GF"]                        # livestock (CME)
EXTRA_ROOTS = ["ZR", "DC", "LBS", "LBR"]
LNG_ROOTS = ["TTF", "JKM"]
FIN_ROOTS = ["ES", "NQ", "ZT", "ZF", "ZN", "ZB", "6E", "6J", "6B", "6A", "6C"]
START, END = "2010-06-06", "2026-10-10"
OUTRIGHT = r"[A-Z]{2,3}[FGHJKMNQUVXZ]\d{1,2}"
STAT_TYPES = {3: "settlement", 6: "cleared_volume", 9: "open_interest"}
CAPS = {"daily": 60.0, "daily_extra": 5.0, "financials": 5.0, "stats": 15.0, "ng_intraday": 8.0,
        "lng_bench": 1.0}
OUT = Path(__file__).resolve().parent / "cache" / "futures_us"


def _client():
    import databento as db
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    return db.Historical(os.environ["DATABENTO_API_KEY"])


def _parent(roots: list[str], schema: str) -> dict:
    return dict(dataset=DATASET, symbols=[f"{r}.FUT" for r in roots], schema=schema, stype_in="parent",
                start=START, end=END)


def _outrights(root: str) -> list[str]:
    files = ["ohlcv_1d.parquet", "ohlcv_1d_extra.parquet"]
    syms = pd.concat([pd.read_parquet(OUT / f, columns=["symbol"])["symbol"] for f in files if (OUT / f).exists()])
    syms = syms[syms.str.fullmatch(OUTRIGHT) & syms.str.match(rf"{root}[FGHJKMNQUVXZ]\d")]
    return sorted(set(syms))


def _ng_windows() -> list[tuple[str, str]]:
    days = pd.bdate_range(START, END)
    days = days[days.dayofweek.isin([1, 2, 3, 4])]
    return [(f"{d.date()}T14:00", f"{d.date()}T18:00") for d in days]


def _requests(job: str) -> list[dict]:
    if job == "daily":
        return [_parent(ROOTS, "ohlcv-1d")]
    if job == "daily_extra":
        return [_parent(EXTRA_ROOTS, "ohlcv-1d")]
    if job == "financials":
        return [_parent(FIN_ROOTS, "ohlcv-1d")]
    if job == "lng_bench":
        return [_parent(LNG_ROOTS, "ohlcv-1d")]
    if job == "stats":
        return [dict(dataset=DATASET, symbols=_outrights(r), schema="statistics", stype_in="raw_symbol",
                     start=START, end=END) for r in ROOTS + EXTRA_ROOTS if _outrights(r)]
    if job == "ng_intraday":
        return [dict(dataset=DATASET, symbols=["NG.c.0", "NG.c.1"], schema="ohlcv-1m", stype_in="continuous",
                     start=s, end=e) for s, e in _ng_windows()]
    raise ValueError(job)


def quote(client, job: str) -> float:
    reqs = _requests(job)
    if job == "ng_intraday":                        # thousands of windows: quote one year and scale
        sample = [r for r in reqs if r["start"].startswith("2020")]
        return sum(client.metadata.get_cost(**r) for r in sample) * len(reqs) / len(sample)
    return sum(client.metadata.get_cost(**r) for r in reqs)


def run(client, job: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reqs = _requests(job)
    if job in ("daily", "daily_extra", "financials", "lng_bench"):
        name = {"daily": "ohlcv_1d", "daily_extra": "ohlcv_1d_extra", "financials": "ohlcv_1d_fin",
                "lng_bench": "ohlcv_1d_lng"}[job]
        df = client.timeseries.get_range(**reqs[0], path=OUT / f"{name}.dbn.zst").to_df()
        df.to_parquet(OUT / f"{name}.parquet")
        print(f"{name}: {len(df):,} rows, {df['symbol'].nunique():,} symbols")
    elif job == "stats":
        parts = []
        for r in reqs:
            df = client.timeseries.get_range(**r).to_df()
            df = df[df["stat_type"].isin(STAT_TYPES)]
            parts.append(df[["ts_ref", "symbol", "stat_type", "price", "quantity", "stat_flags", "update_action"]])
            print(f"  {r['symbols'][0][:2]}: {len(df):,} rows")
        out = pd.concat(parts)
        out.to_parquet(OUT / "statistics.parquet")
        print(f"statistics: {len(out):,} rows")
    elif job == "ng_intraday":
        from concurrent.futures import ThreadPoolExecutor

        def one(r):
            df = client.timeseries.get_range(**r).to_df()
            return df[["symbol", "instrument_id", "open", "high", "low", "close", "volume"]].rename(
                columns={"symbol": "series"}) if len(df) else None         # series = NG.c.0 / NG.c.1

        part_dir = OUT / "ng_1m_parts"
        part_dir.mkdir(exist_ok=True)
        for year in sorted({r["start"][:4] for r in reqs}):
            path = part_dir / f"{year}.parquet"
            if path.exists():                                        # resumable: a finished year is not re-billed
                continue
            with ThreadPoolExecutor(8) as pool:
                parts = [p for p in pool.map(one, [r for r in reqs if r["start"].startswith(year)]) if p is not None]
            pd.concat(parts).to_parquet(path)
            print(f"  {year}: {sum(len(p) for p in parts):,} rows")
        out = pd.concat(pd.read_parquet(f) for f in sorted(part_dir.glob("*.parquet")))
        out.to_parquet(OUT / "ng_1m_release_windows.parquet")
        print(f"ng_intraday: {len(out):,} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+", choices=list(CAPS))
    ap.add_argument("--quote", action="store_true")
    args = ap.parse_args()
    client = _client()
    for job in args.jobs:
        cost = quote(client, job)
        print(f"{job}: quoted cost ${cost:.2f} (cap ${CAPS[job]:.2f})")
        if args.quote:
            continue
        if cost > CAPS[job]:
            raise SystemExit(f"{job}: quoted cost above the cap, not pulling")
        run(client, job)


if __name__ == "__main__":
    main()
