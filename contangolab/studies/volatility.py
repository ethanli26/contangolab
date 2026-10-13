"""CL1 study 4: does a backwardated curve warn of volatility?

With little inventory there is no buffer: a supply shock has to be absorbed by price. So a market
whose curve is backwardated should be more volatile over the following weeks than the same market
when its curve is near its usual shape (Kogan, Livdan & Yaron 2009 find a V shape: both ends).

Per market (at least a year of liquid days): slope terciles over its liquid days; forward volatility
= annualized std of the held contract's returns over the next 21 trading days.

H4 bar (pre-registered, each country separately): forward vol in the most-backwardated tercile >
the middle tercile in >= 2/3 of markets, with a one-sided sign test p < 0.01. Contango vs middle is
reported without a bar.

    python -m contangolab study volatility
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from contangolab.config import RESULTS
from contangolab.markets.curves import COUNTRY, NAMES, slopes
from contangolab.studies.style import BLUE, INK, MUTED, ORANGE, footnote, headline, mpl, save, style

HORIZON = 21
MIN_LIQUID_DAYS = 252


def forward_vol(ret: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """Std of returns t+1 .. t+horizon (the trailing window that ends horizon days later), not annualized."""
    return ret.rolling(horizon, min_periods=horizon).std().shift(-horizon)


def table(country: str) -> pd.DataFrame:
    s = slopes(country).sort_values(["sym", "date"])
    fwd = s.groupby("sym")["ret"].transform(forward_vol)
    s = s.assign(fwd_vol=fwd * np.sqrt(252))
    s = s[s["liquid"]].dropna(subset=["fwd_vol"])
    rows = []
    for sym, g in s.groupby("sym"):
        if len(g) < MIN_LIQUID_DAYS:
            continue
        t = pd.qcut(g["slope"].rank(method="first"), 3, labels=["contango", "middle", "backwardation"])
        v = g.groupby(t, observed=True)["fwd_vol"].mean()
        rows.append({"country": country, "sym": sym, "name": NAMES[country].get(sym, sym), "days": len(g), **v.to_dict()})
    out = pd.DataFrame(rows)
    out["back_vs_mid"] = out["backwardation"] / out["middle"] - 1
    out["contango_vs_mid"] = out["contango"] / out["middle"] - 1
    return out


def verdict(t: pd.DataFrame) -> dict:
    k, n = int((t["backwardation"] > t["middle"]).sum()), len(t)
    kc = int((t["contango"] > t["middle"]).sum())
    p = binomtest(k, n, 0.5, alternative="greater").pvalue
    return {"markets": n, "back_gt_mid": k, "share": k / n, "p": float(p), "pass": bool(k / n >= 2 / 3 and p < 0.01),
            "contango_gt_mid": kc, "median_back_vs_mid": float(t["back_vs_mid"].median()),
            "median_contango_vs_mid": float(t["contango_vs_mid"].median())}


def chart(tables: dict, verdicts: dict, path=RESULTS / "curve_volatility.png") -> None:
    plt = mpl()
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    style(ax, "", "Next-month volatility vs the middle tercile")
    cats = ["contango", "middle", "backwardation"]
    x, w = np.arange(3), 0.36
    for off, c, color in [(-w / 2, "cn", BLUE), (w / 2, "us", ORANGE)]:
        t = tables[c]
        med = [float((t[k] / t["middle"]).median() - 1) for k in cats]
        bars = ax.bar(x + off, med, width=w - 0.03, color=color, label=f"{COUNTRY[c]} ({len(t)} markets)")
        for b, v in list(zip(bars, med))[::2]:                         # the middle bar is the 0 reference
            ax.annotate(f"{v:+.0%}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3 if v >= 0 else -10),
                        textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(x, ["Most contango\nthird of days", "Middle third", "Most backwardated\nthird of days"], fontsize=8.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    fig.subplots_adjust(top=0.80, bottom=0.17, left=0.10, right=0.98)
    vc, vu = verdicts["cn"], verdicts["us"]
    headline(fig, "Does the curve warn of volatility?",
             f"Median across markets. Backwardated > middle in {vc['back_gt_mid']}/{vc['markets']} China markets and "
             f"{vu['back_gt_mid']}/{vu['markets']} US markets.")
    footnote(fig, "Forward volatility: annualized std of the held contract's daily returns over the next 21 trading "
             "days. Terciles of each market's own slope history.")
    save(fig, path)


def main() -> int:
    tables = {c: table(c) for c in ("cn", "us")}
    verdicts = {c: verdict(t) for c, t in tables.items()}
    pd.concat(tables.values()).to_csv(RESULTS / "curve_volatility.csv", index=False, float_format="%.4f")
    for c in ("cn", "us"):
        print(tables[c].round(3).to_string(index=False))
        v = verdicts[c]
        print(f"{COUNTRY[c]}: backwardated > middle in {v['back_gt_mid']}/{v['markets']} ({v['share']:.0%}), "
              f"p = {v['p']:.2g} -> {'PASS' if v['pass'] else 'FAIL'}; contango > middle in "
              f"{v['contango_gt_mid']}/{v['markets']}; medians {v['median_back_vs_mid']:+.1%} / "
              f"{v['median_contango_vs_mid']:+.1%}\n")
    chart(tables, verdicts)
    return 0


if __name__ == "__main__":
    main()
