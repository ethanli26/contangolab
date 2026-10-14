"""Audit: does a market's backwardation share depend on which contract the slope is measured against?

The atlas slope (``markets.curves``) compares the held contract with the *nearest later expiry with
open interest*. In markets where trading concentrates in a few months (Dalian iron ore: January,
May, September), that next contract is often thin and its price can be stale, which could make the
curve look backwardated when the liquid part of the curve is not. This study rebuilds the slope from
every listed contract four ways and compares the shares:

  * ``atlas``:     the published slope (``ry`` from the continuous build), liquid days only;
  * ``next``:      dominant (most open interest) contract vs the nearest later contract with any OI;
  * ``most_held``: dominant contract vs the most-held later contract;
  * ``oi_10pct``:  dominant contract vs the nearest later contract with at least 10% of its OI;

plus ``curve_down``, the share of adjacent steps sloping down along the whole curve, counting only
contracts with at least 5% of the dominant contract's OI.

Descriptive (no bar). Written 2026-10-14 to audit the "Chinese iron ore: 99% of days" figure.

    python -m contangolab study robustness
"""

import numpy as np
import pandas as pd

from contangolab.config import RESULTS
from contangolab.markets.curves import COUNTRY, NAMES, contracts, slopes

MARKETS = {"cn": ["I", "J", "JM", "RB", "HC"], "us": ["CL", "NG", "HG"]}
NEXT_OI_SHARE = 0.10    # oi_10pct: a later contract counts once it holds this share of the dominant OI
CURVE_OI_SHARE = 0.05   # curve_down: contracts below this share of the dominant OI are left off the curve


def _months(far, near):
    return (far // 100 - near // 100) * 12 + (far % 100 - near % 100)


def _ry(front, back) -> float:
    """Annualized log slope, positive in backwardation (the same sign as the atlas slope)."""
    return float(np.log(front.close / back.close) * 12 / _months(back.expKey, front.expKey))


def alt_slopes(c: pd.DataFrame) -> pd.DataFrame:
    """One market's contracts (date, expKey, close, oi) -> per-day slopes under each contract choice."""
    c = c[(c["oi"].fillna(0) > 0) & (c["close"] > 0)]
    rows = []
    for date, g in c.groupby("date"):
        g = g.sort_values("expKey")
        dom = g.loc[g["oi"].idxmax()]
        later = g[g["expKey"] > dom.expKey]
        if later.empty:
            continue
        row = {"date": date, "next": _ry(dom, later.iloc[0]), "most_held": _ry(dom, later.loc[later["oi"].idxmax()]),
               "next_oi_share": float(later.iloc[0].oi / dom.oi)}
        deep = later[later["oi"] >= NEXT_OI_SHARE * dom.oi]
        row["oi_10pct"] = _ry(dom, deep.iloc[0]) if len(deep) else np.nan
        curve = g[g["oi"] >= CURVE_OI_SHARE * dom.oi]
        row["curve_down"] = float((np.diff(curve["close"].to_numpy()) < 0).mean()) if len(curve) >= 3 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def table() -> pd.DataFrame:
    rows = []
    for country, syms in MARKETS.items():
        s = slopes(country)
        c = contracts(country)
        for sym in syms:
            atlas = s[(s["sym"] == sym) & s["liquid"]]
            alt = alt_slopes(c[c["sym"] == sym])
            alt = alt[alt["date"].isin(set(atlas["date"]))]
            if atlas.empty or alt.empty:
                continue
            rows.append({"country": country, "sym": sym, "name": NAMES[country].get(sym, sym),
                         "start": atlas["date"].min().date(), "end": atlas["date"].max().date(),
                         "liquid_days": len(atlas), "atlas": float((atlas["slope"] > 0).mean()),
                         **{k: float((alt[k].dropna() > 0).mean()) for k in ("next", "most_held", "oi_10pct")},
                         "curve_down": float(alt["curve_down"].mean()),
                         "median_slope": float(atlas["slope"].median()),
                         "next_oi_share": float(alt["next_oi_share"].median())})
    return pd.DataFrame(rows)


def main() -> int:
    t = table()
    t.to_csv(RESULTS / "slope_robustness.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print("Share of liquid days in backwardation, by contract choice "
          "(curve_down = share of downward steps on the whole liquid curve)\n")
    for country in MARKETS:
        print(COUNTRY[country])
        print(t[t["country"] == country].drop(columns="country").round(3).to_string(index=False), "\n")
    return 0


if __name__ == "__main__":
    main()
