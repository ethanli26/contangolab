"""Download US commodity futures daily bars (every contract month) from Databento.

One request: CME Globex (GLBX.MDP3) ``ohlcv-1d`` for the parent roots below, June 2010 onward.
The quoted cost is checked first (a free metadata call) and the pull aborts above ``MAX_USD``,
so it never spends past the account's free credit. Output is written raw (DBN) and as parquet
to ``data/cache/futures_us/`` (gitignored; Databento data is licensed, not redistributable).

    python -m data.pull_databento            # quote, then pull if under the cap
    python -m data.pull_databento --quote    # quote only
"""

import argparse
import os
from pathlib import Path

DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1d"
ROOTS = ["CL", "HO", "RB", "NG", "BZ",            # energy (NYMEX)
         "GC", "SI", "HG", "PL", "PA",            # metals (COMEX / NYMEX)
         "ZC", "ZW", "KE", "ZS", "ZM", "ZL", "ZO",  # grains and oilseeds (CBOT)
         "LE", "HE", "GF"]                        # livestock (CME)
START, END = "2010-06-06", "2026-10-10"
MAX_USD = 60.0
OUT = Path(__file__).resolve().parent / "cache" / "futures_us"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quote", action="store_true")
    args = ap.parse_args()

    import databento as db
    from dotenv import load_dotenv
    load_dotenv()
    client = db.Historical(os.environ["DATABENTO_API_KEY"])
    req = dict(dataset=DATASET, symbols=[f"{r}.FUT" for r in ROOTS], schema=SCHEMA, stype_in="parent",
               start=START, end=END)
    cost = client.metadata.get_cost(**req)
    print(f"quoted cost ${cost:.2f} (cap ${MAX_USD:.2f})")
    if args.quote:
        return
    if cost > MAX_USD:
        raise SystemExit("quoted cost above the cap: not pulling")
    OUT.mkdir(parents=True, exist_ok=True)
    data = client.timeseries.get_range(**req, path=OUT / "ohlcv_1d.dbn.zst")
    df = data.to_df()
    df.to_parquet(OUT / "ohlcv_1d.parquet")
    print(f"{len(df):,} rows, {df['symbol'].nunique():,} symbols, {df.index.min()} -> {df.index.max()}")


if __name__ == "__main__":
    main()
