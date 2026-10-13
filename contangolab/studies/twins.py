"""CL1 study 2, twin curves: the same commodity's curve in China and in the US.

For each of the nine commodities listed in both countries (``markets.curves.TWINS``): the monthly
correlation of the two curve slopes, how often they agree on contango vs backwardation, and each
side's mean slope. Months count when both markets are liquid on at least half their days.

H2 (pre-registered, cost of carry): gold is never scarce and cheap to store, so its contango should be
about the local 3-month interest rate. Bar: in each country, monthly -slope on the local rate has a
coefficient in [0.5, 1.5] and R^2 >= 0.5, and the US-minus-China contango gap tracks the rate gap with
correlation >= 0.7. Silver gets the same statistics without a bar.

    python -m contangolab study twins
"""

import numpy as np
import pandas as pd

from contangolab.config import RESULTS
from contangolab.markets.curves import TWINS, monthly_slopes
from contangolab.markets.inventory import rates
from contangolab.studies.style import BLUE, INK, INK_2, MUTED, ORANGE, footnote, headline, mpl, save, style

LIQUID_SHARE = 0.5


def pairs() -> pd.DataFrame:
    cn, us = monthly_slopes("cn", LIQUID_SHARE), monthly_slopes("us", LIQUID_SHARE)
    rows = []
    for name, a, b in TWINS:
        j = pd.concat([cn.get(a), us.get(b)], axis=1, keys=["cn", "us"]).dropna()
        rows.append({"commodity": name, "china": a, "us": b, "months": len(j),
                     "start": str(j.index.min()) if len(j) else None,
                     "corr": j["cn"].corr(j["us"]) if len(j) > 2 else np.nan,
                     "same_sign": float((np.sign(j["cn"]) == np.sign(j["us"])).mean()) if len(j) else np.nan,
                     "cn_mean": j["cn"].mean(), "us_mean": j["us"].mean(),
                     "cn_backwardation": float((j["cn"] > 0).mean()) if len(j) else np.nan,
                     "us_backwardation": float((j["us"] > 0).mean()) if len(j) else np.nan})
    return pd.DataFrame(rows)


def _ols(y: pd.Series, x: pd.Series) -> dict:
    j = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    beta, alpha = np.polyfit(j["x"], j["y"], 1)
    r2 = float(j["y"].corr(j["x"]) ** 2)
    return {"coef": float(beta), "intercept": float(alpha), "r2": r2, "months": len(j),
            "start": str(j.index.min()), "end": str(j.index.max())}


def carry_test(metal: str = "Gold") -> dict:
    """-slope (contango, per year) vs the local 3-month rate, per country, and the gap test."""
    _, a, b = next(t for t in TWINS if t[0] == metal)
    r = rates() / 100.0
    cn = -monthly_slopes("cn", LIQUID_SHARE)[a]
    us = -monthly_slopes("us", LIQUID_SHARE)[b]
    out = {"cn": _ols(cn, r["cn_3m"]), "us": _ols(us, r["us_3m"])}
    gap = pd.concat([us - cn, r["us_3m"] - r["cn_3m"]], axis=1, keys=["contango_gap", "rate_gap"]).dropna()
    out["gap_corr"] = float(gap["contango_gap"].corr(gap["rate_gap"]))
    out["gap_months"] = len(gap)
    ok = lambda d: 0.5 <= d["coef"] <= 1.5 and d["r2"] >= 0.5  # noqa: E731
    out["passed"] = bool(ok(out["cn"]) and ok(out["us"]) and out["gap_corr"] >= 0.7)
    out["series"] = pd.concat([cn, us, r["cn_3m"], r["us_3m"]], axis=1, keys=["cn", "us", "cn_rate", "us_rate"])
    return out


def twin_chart(p: pd.DataFrame, path=RESULTS / "twin_curves.png") -> None:
    plt = mpl()
    cn, us = monthly_slopes("cn", LIQUID_SHARE), monthly_slopes("us", LIQUID_SHARE)
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.4), sharex=True)
    for ax, (name, a, b), row in zip(axes.flat, TWINS, p.itertuples()):
        j = pd.concat([cn.get(a), us.get(b)], axis=1, keys=["cn", "us"])
        j = j.loc[j.dropna(how="all").index.min():] if j.notna().any().any() else j
        lo, hi = float(j.min().min()), float(j.max().max())
        style(ax, "", "")
        ax.axhspan(0, max(hi, 0.02) * 1.15, color="#f6c3bd", alpha=0.18, linewidth=0)
        ax.axhspan(min(lo, -0.02) * 1.15, 0, color="#b7d3f6", alpha=0.18, linewidth=0)
        ax.axhline(0, color=MUTED, linewidth=0.7)
        x = j.index.to_timestamp()
        ax.plot(x, j["cn"], color=BLUE, linewidth=1.3)
        ax.plot(x, j["us"], color=ORANGE, linewidth=1.3)
        ax.set_ylim(min(lo, -0.02) * 1.15, max(hi, 0.02) * 1.15)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:+.0f}%"))
        ax.set_title(f"{name}", loc="left", fontsize=10.5, fontweight="semibold", color=INK, pad=4)
        ax.text(1.0, 1.02, f"r = {row.corr:+.2f}   same shape {row.same_sign:.0%}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, color=INK_2)
        ax.text(0.01, 0.97, "backwardation", transform=ax.transAxes, fontsize=6.5, color="#a8211f", va="top")
        ax.text(0.01, 0.03, "contango", transform=ax.transAxes, fontsize=6.5, color="#104281", va="bottom")
    handles = [plt.Line2D([], [], color=BLUE, lw=2), plt.Line2D([], [], color=ORANGE, lw=2)]
    fig.legend(handles, ["China (SHFE / INE / DCE)", "US (COMEX / NYMEX / CBOT / CME)"], loc="lower right",
               frameon=False, fontsize=9, ncol=2, bbox_to_anchor=(0.995, -0.005))
    fig.subplots_adjust(top=0.86, bottom=0.07, left=0.06, right=0.99, hspace=0.42, wspace=0.18)
    headline(fig, "Same commodity, two curves",
             "Monthly slope of the front of the curve, annualized. r = correlation of the two; same shape = months "
             "both in contango or both in backwardation.")
    footnote(fig, "Months where both markets are liquid. Dalian (soy complex, corn, hogs) data from 2019; INE crude "
             "from 2018; US from 2016.")
    save(fig, path)


def gold_chart(g: dict, path=RESULTS / "gold_carry.png") -> None:
    plt = mpl()
    s = g["series"].sort_index()
    s = s[s.index >= pd.Period("2010-01", "M")].dropna(how="all")
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    style(ax, "", "Percent per year")
    x = s.index.to_timestamp()
    ax.plot(x, s["us"] * 100, color=ORANGE, linewidth=1.5)
    ax.plot(x, s["us_rate"] * 100, color=ORANGE, linewidth=1.2, linestyle=(0, (3, 2)))
    ax.plot(x, s["cn"] * 100, color=BLUE, linewidth=1.5)
    ax.plot(x, s["cn_rate"] * 100, color=BLUE, linewidth=1.2, linestyle=(0, (3, 2)))
    ends = []
    for col, text in [("us", "COMEX gold contango"), ("us_rate", "US 3-month T-bill"),
                      ("cn", "Shanghai gold contango"), ("cn_rate", "China 3-month interbank")]:
        last = s[col].dropna()
        if len(last):
            ends.append([last.iloc[-1] * 100, last.index[-1].to_timestamp(), text])
    ends.sort()
    for i in range(1, len(ends)):                       # keep end labels at least 0.45 points apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.45)
    for y, x, text in ends:
        ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", fontsize=7.5, color=INK_2, va="center")
    ax.axhline(0, color=MUTED, linewidth=0.7)
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.07, right=0.80)
    us, cn = g["us"], g["cn"]
    headline(fig, "Gold's contango is the price of money" if g["passed"] else "Is gold's contango the price of money?",
             f"Solid: gold curve contango. Dashed: local 3-month rate. Slope on rate: US {us['coef']:.2f} "
             f"(R² {us['r2']:.2f}), China {cn['coef']:.2f} (R² {cn['r2']:.2f}).")
    footnote(fig, "Monthly. Rates: FRED TB3MS and IR3TIB01CNM156N (OECD). Contango = minus the annualized slope of "
             "the front of the curve.")
    save(fig, path)


def main() -> int:
    p = pairs()
    p.to_csv(RESULTS / "twin_curves.csv", index=False, float_format="%.4f")
    print(p.round(3).to_string(index=False))
    results = {}
    for metal in ("Gold", "Silver"):
        g = carry_test(metal)
        results[metal] = g
        print(f"\n{metal}: China coef {g['cn']['coef']:.2f} R2 {g['cn']['r2']:.2f} ({g['cn']['months']} months "
              f"{g['cn']['start']}..{g['cn']['end']}); US coef {g['us']['coef']:.2f} R2 {g['us']['r2']:.2f} "
              f"({g['us']['months']} months); gap corr {g['gap_corr']:.2f} ({g['gap_months']} months)"
              + (f"  H2 {'PASS' if g['passed'] else 'FAIL'}" if metal == "Gold" else "  (no bar)"))
    twin_chart(p)
    gold_chart(results["Gold"])
    return 0


if __name__ == "__main__":
    main()
