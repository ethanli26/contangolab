"""CL1 study 3, the theory of storage: do inventories set the shape of the curve?

Kaldor (1939), Working (1949), Brennan (1958): when stocks are ample, holding the commodity is
worth little beyond its carrying cost, and the curve sits in contango; when stocks are scarce, the
"convenience yield" of having it now is high and the curve backwardates. So abnormally high
inventory should go with an abnormally contango curve: b < 0 below.

Per market, at each weekly inventory date d (slope = the last trading day on or before d):
  inv_dev    log(1 + stocks) minus its seasonal norm, z-scored within the market
  slope_dev  slope minus its seasonal norm
  norm       mean over the same ISO week +-2 in all EARLIER years (at least 3 of them)
  slope_dev = a + b inv_dev_z, Newey-West standard errors with 12 lags

H3 bar (pre-registered): China b < 0 in >= 2/3 of qualifying markets with a one-sided sign test
p < 0.01; US b < 0 with t <= -2 for >= 3 of the 4 EIA pairs.

    python -m contangolab study storage
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from contangolab.config import RESULTS
from contangolab.markets.curves import NAMES, slopes
from contangolab.markets.inventory import china_receipts, us_stocks
from contangolab.studies.style import BLUE, INK, INK_2, MUTED, ORANGE, footnote, headline, mpl, save, style

WINDOW_WEEKS = 2
MIN_YEARS = 3
NW_LAGS = 12
MAX_ZERO_SHARE = 0.25
MIN_WEEKS = 156


def seasonal_norm(s: pd.Series) -> pd.Series:
    """For each date: mean of values in earlier years within +-WINDOW_WEEKS ISO weeks (NaN if < MIN_YEARS years)."""
    s = s.dropna()
    week = s.index.isocalendar().week.to_numpy().astype(int)
    year = s.index.year.to_numpy()
    v = s.to_numpy(dtype=float)
    out = np.full(len(s), np.nan)
    for i in range(len(s)):
        dist = np.abs(week - week[i])
        dist = np.minimum(dist, 52 - dist)
        m = (year < year[i]) & (dist <= WINDOW_WEEKS)
        if len(np.unique(year[m])) >= MIN_YEARS:
            out[i] = v[m].mean()
    return pd.Series(out, index=s.index)


def ols_nw(y: np.ndarray, x: np.ndarray, lags: int = NW_LAGS) -> dict:
    """OLS of y on [1, x] with Newey-West (Bartlett) standard errors."""
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ beta
    xu = X * u[:, None]
    S = xu.T @ xu
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        g = xu[lag:].T @ xu[:-lag]
        S += w * (g + g.T)
    inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(inv @ S @ inv))
    return {"b": float(beta[1]), "t": float(beta[1] / se[1]), "r2": float(1 - u.var() / y.var()), "n": len(y)}


def panel(country: str) -> pd.DataFrame:
    """Weekly rows per qualifying market: date, sym, stocks, slope, liquid, inv_dev_z, slope_dev."""
    stocks = china_receipts() if country == "cn" else us_stocks()
    sl = slopes(country).sort_values("date")
    rows, info = [], []
    for sym, st in stocks.groupby("sym"):
        st = st.sort_values("date")
        zero = float((st["value"] == 0).mean())
        m = pd.merge_asof(st, sl[sl["sym"] == sym][["date", "slope", "liquid"]], on="date", direction="backward",
                          tolerance=pd.Timedelta(days=7)).dropna(subset=["slope"])
        liquid = float(m["liquid"].mean()) if len(m) else 0.0
        ok = (country == "us") or (zero <= MAX_ZERO_SHARE and len(m) >= MIN_WEEKS and liquid >= 0.5)
        info.append({"sym": sym, "weeks": len(m), "zero_share": zero, "liquid_share": liquid, "qualifies": ok})
        if not ok:
            continue
        m = m.set_index("date")
        inv = np.log1p(m["value"])
        m["inv_dev"] = inv - seasonal_norm(inv)
        m["slope_dev"] = m["slope"] - seasonal_norm(m["slope"])
        m = m.dropna(subset=["inv_dev", "slope_dev"])
        m["inv_dev_z"] = (m["inv_dev"] - m["inv_dev"].mean()) / m["inv_dev"].std(ddof=1)
        rows.append(m.reset_index().assign(sym=sym))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(info)


def regressions(p: pd.DataFrame, country: str) -> pd.DataFrame:
    out = []
    for sym, g in p.groupby("sym"):
        g = g.sort_values("date")
        r = ols_nw(g["slope_dev"].to_numpy(), g["inv_dev_z"].to_numpy())
        out.append({"country": country, "sym": sym, "name": NAMES[country].get(sym, sym), **r,
                    "start": g["date"].min().date(), "end": g["date"].max().date()})
    return pd.DataFrame(out).sort_values("b")


def verdict(cn: pd.DataFrame, us: pd.DataFrame) -> dict:
    neg = int((cn["b"] < 0).sum())
    p = binomtest(neg, len(cn), 0.5, alternative="greater").pvalue
    us_hits = int(((us["b"] < 0) & (us["t"] <= -2)).sum())
    return {"cn_negative": neg, "cn_markets": len(cn), "cn_share": neg / len(cn), "cn_sign_p": float(p),
            "cn_pass": bool(neg / len(cn) >= 2 / 3 and p < 0.01), "us_hits": us_hits, "us_pairs": len(us),
            "us_pass": bool(us_hits >= 3)}


def deciles(p: pd.DataFrame) -> pd.DataFrame:
    q = pd.qcut(p["inv_dev_z"], 10, labels=False)
    return p.groupby(q).agg(inv_dev_z=("inv_dev_z", "mean"), slope_dev=("slope_dev", "mean"),
                            n=("slope_dev", "size"))


def chart(pc: pd.DataFrame, pu: pd.DataFrame, rc: pd.DataFrame, ru: pd.DataFrame, v: dict,
          path=RESULTS / "storage_curve.png") -> None:
    plt = mpl()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.9), gridspec_kw={"width_ratios": [1.15, 1]})
    style(a1, "", "Curve slope vs its seasonal norm (per year)")
    for p, color, lab in [(pc, BLUE, f"China, {v['cn_markets']} markets (warehouse receipts)"),
                          (pu, ORANGE, "US, 4 markets (EIA weekly stocks)")]:
        d = deciles(p)
        a1.plot(d["inv_dev_z"], d["slope_dev"], color=color, linewidth=1.6, marker="o", markersize=5, label=lab)
    a1.axhline(0, color=MUTED, linewidth=0.7)
    a1.axvline(0, color=MUTED, linewidth=0.7)
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.0%}"))
    a1.set_xlabel("Inventory vs its seasonal norm (z-score, decile means)", color=INK_2, fontsize=9)
    a1.text(0.02, 0.97, "scarce stocks: backwardation", transform=a1.transAxes, fontsize=7.5, color="#a8211f",
            va="top")
    a1.text(0.98, 0.03, "ample stocks: contango", transform=a1.transAxes, fontsize=7.5, color="#104281",
            ha="right", va="bottom")
    a1.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK)
    a1.set_title("The storage curve", loc="left", fontsize=11, fontweight="semibold", color=INK)

    r = pd.concat([rc.assign(color=BLUE), ru.assign(color=ORANGE)]).sort_values("b")
    style(a2, "", "", grid_axis="x")
    y = np.arange(len(r))
    a2.scatter(r["b"] * 100, y, color=r["color"], s=22, zorder=3)
    a2.axvline(0, color=MUTED, linewidth=0.8)
    a2.set_yticks(y, [f"{'CN' if c == BLUE else 'US'} {n}" for c, n in zip(r["color"], r["name"])], fontsize=6.5)
    a2.set_xlabel("Change in slope per 1 s.d. more inventory (percentage points a year)", color=INK_2, fontsize=8.5)
    a2.set_title(f"Each market: {v['cn_negative'] + int((ru['b'] < 0).sum())} of {len(r)} slope down",
                 loc="left", fontsize=11, fontweight="semibold", color=INK)
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.07, right=0.98, wspace=0.32)
    passed = v["cn_pass"] and v["us_pass"]
    headline(fig, "Full warehouses push the curve into contango" if passed else "Do full warehouses flatten the curve?",
             "The theory of storage, tested: does the curve move toward contango when stocks are above their seasonal "
             f"norm? Pre-registered bar: China {'passed' if v['cn_pass'] else 'failed'}, "
             f"US {'passed' if v['us_pass'] else 'failed'}.")
    footnote(fig, "Weekly. China: SHFE / INE / CZCE registered warehouse receipts, 2015-2026. US: EIA Cushing crude (CL), "
             "distillate (HO), gasoline (RB), Lower-48 gas (NG). Both deseasonalized against earlier years only.")
    save(fig, path)


def main() -> int:
    pc, info_cn = panel("cn")
    pu, _ = panel("us")
    rc, ru = regressions(pc, "cn"), regressions(pu, "us")
    pd.concat([rc, ru]).to_csv(RESULTS / "storage_regressions.csv", index=False, float_format="%.4f")
    info_cn.to_csv(RESULTS / "storage_china_markets.csv", index=False, float_format="%.3f")
    print(info_cn.to_string(index=False))
    print(pd.concat([rc, ru]).round(4).to_string(index=False))
    v = verdict(rc, ru)
    print(f"\nChina: b < 0 in {v['cn_negative']}/{v['cn_markets']} ({v['cn_share']:.0%}), sign test p = "
          f"{v['cn_sign_p']:.2g} -> {'PASS' if v['cn_pass'] else 'FAIL'}")
    print(f"US: b < 0 with t <= -2 in {v['us_hits']}/{v['us_pairs']} -> {'PASS' if v['us_pass'] else 'FAIL'}")
    print("\nChina deciles\n", deciles(pc).round(4).to_string())
    print("US deciles\n", deciles(pu).round(4).to_string())
    chart(pc, pu, rc, ru, v)
    return 0


if __name__ == "__main__":
    main()
