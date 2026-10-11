"""Reconcile the pandas futures panel against the DolphinDB pipeline it was ported from.

Two independent implementations of the same continuous-contract rules should agree to
floating-point noise. If they don't, one of them has a bug, and nothing built on either is
trustworthy until it's found. Pre-registered bar (BLUEPRINT 11, P1a): every column within
1e-9 relative on >= 99.9% of (date, sym) rows, with every mismatch explained.

Export the reference from DolphinDB (~/ddb, server on :8848) first:

    EXCL = array(STRING, 0)
    run 00_load_futures_daily.dos, then save `px` to CSV

then:

    python -m research.reconcile_futures path/to/ddb_px.csv

Result on 2026-10-10: 157,395 / 157,395 rows, all 15 columns identical (max relative
difference 4.4e-16), held contract identical on every row, universe mask identical.
"""

import sys

import numpy as np
import pandas as pd

from data.futures_cn import load_continuous

TOLERANCE = 1e-9
REQUIRED_MATCH = 0.999
NUMERIC = ["ret", "close", "rawClose", "open", "high", "low", "volume", "oi", "totVol", "totOI",
           "mult", "notional", "notionalOI", "ry", "farRet"]


def reconcile(reference: pd.DataFrame, ours: pd.DataFrame) -> pd.DataFrame:
    """One row per column: share of rows matching, NaN mismatches, max relative difference."""
    reference = reference.assign(date=pd.to_datetime(reference["date"]))
    merged = reference.merge(ours, on=["date", "sym"], how="outer", suffixes=("_ref", "_ours"), indicator=True)
    both = merged[merged["_merge"] == "both"]
    rows = [{"column": "(rows)", "match": len(both) / len(merged), "nan_mismatch": int((merged["_merge"] != "both").sum()),
             "max_rel": 0.0}]
    rows.append({"column": "held", "match": float((both["held_ref"] == both["held_ours"]).mean()),
                 "nan_mismatch": 0, "max_rel": 0.0})
    rows.append({"column": "uni", "match": float((both["uni_ref"].astype(bool) == both["uni_ours"].astype(bool)).mean()),
                 "nan_mismatch": 0, "max_rel": 0.0})
    for col in NUMERIC:
        a, b = both[f"{col}_ref"].astype(float), both[f"{col}_ours"].astype(float)
        nan_mismatch = a.isna() ^ b.isna()
        rel = ((a - b).abs() / np.maximum(1.0, a.abs())).fillna(0.0)
        ok = (rel < TOLERANCE) & ~nan_mismatch
        rows.append({"column": col, "match": float(ok.mean()), "nan_mismatch": int(nan_mismatch.sum()),
                     "max_rel": float(rel.max())})
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    table = reconcile(pd.read_csv(argv[0]), load_continuous())
    print(table.to_string(index=False))
    passed = bool((table["match"] >= REQUIRED_MATCH).all())
    print(f"\n{'PASS' if passed else 'FAIL'}: every column must match on >= {REQUIRED_MATCH:.1%} of rows.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
