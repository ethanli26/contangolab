"""CL1 study 5, case studies: whole curves on the days that made the textbooks.

April 2020: lockdowns cut oil demand by about a fifth, storage filled, and on 20 April the May WTI
contract, deliverable only at Cushing, Oklahoma, settled at -$37.63: holders with nowhere to put
the oil paid to be rid of it. The same days are shown for Shanghai's INE crude contract, deliverable
into bonded tanks in China. March 2022 is the opposite: after the invasion of Ukraine the WTI curve
was steeply backwardated.

    python -m contangolab study cases
"""

import pandas as pd

from contangolab.config import RESULTS
from contangolab.markets.curves import curve
from contangolab.studies.style import INK_2, MUTED, RED, footnote, headline, mpl, save, style

APRIL_2020 = ["2020-01-02", "2020-04-17", "2020-04-20", "2020-04-21", "2020-07-01"]
HIGHLIGHT = "2020-04-20"
SHADES = {"2020-01-02": "#c3c2b7", "2020-04-17": "#898781", "2020-04-21": "#52514e", "2020-07-01": "#b7d3f6"}
MAX_MONTHS = 24


def curves(country: str, sym: str, dates: list[str]) -> dict:
    out = {}
    for d in dates:
        c = curve(country, sym, d)
        if len(c):
            c = c[c["expiry"] <= pd.Timestamp(d) + pd.DateOffset(months=MAX_MONTHS)]
            out[d] = c
    return out


def _panel(ax, cs: dict, title: str, unit: str, plt) -> None:
    style(ax, title, unit)
    for d, c in cs.items():
        hi = d == HIGHLIGHT
        ax.plot(c["expiry"], c["price"], color=RED if hi else SHADES.get(d, MUTED), linewidth=2.0 if hi else 1.3,
                marker="o", markersize=3 if hi else 2, zorder=3 if hi else 2)
        ax.lines[-1].set_label(pd.Timestamp(d).strftime("%-d %b %Y") + ("  (May WTI −$37.63)" if hi and title.startswith("WTI")
                                                                      else ""))
    ax.axhline(0, color=MUTED, linewidth=0.7)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b\n%Y"))
    ax.set_xlabel("Delivery month", color=INK_2, fontsize=8.5)


def chart(path=RESULTS / "april_2020.png") -> dict:
    plt = mpl()
    import matplotlib.dates  # noqa: F401  (for the formatter)

    cl, sc = curves("us", "CL", APRIL_2020), curves("cn", "SC", APRIL_2020)
    ukr = curves("us", "CL", ["2022-03-08"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1.15, 0.85]})
    _panel(axes[0], cl, "WTI crude, NYMEX (Cushing, Oklahoma)", "US dollars per barrel", plt)
    _panel(axes[1], sc, "Crude oil, Shanghai INE (bonded tanks, China)", "Yuan per barrel", plt)
    style(axes[2], "WTI, 8 March 2022", "US dollars per barrel")
    for c in ukr.values():
        axes[2].plot(c["expiry"], c["price"], color=RED, linewidth=2.0, marker="o", markersize=3)
    axes[2].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b\n%Y"))
    axes[2].set_xlabel("Delivery month", color=INK_2, fontsize=8.5)
    for ax in axes[:2]:
        ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK_2)
    c22 = ukr["2022-03-08"]
    gap, far = c22["price"].iloc[0] - c22["price"].iloc[-1], c22["expiry"].iloc[-1]
    axes[2].text(0.97, 0.95, f"steep backwardation:\na barrel for April 2022\ncost ${gap:.0f} more than one\n"
                 f"for {far:%B %Y}", transform=axes[2].transAxes,
                 ha="right", va="top", fontsize=8, color=INK_2)
    fig.subplots_adjust(top=0.76, bottom=0.15, left=0.05, right=0.98, wspace=0.28)
    headline(fig, "When storage runs out: the shape of the curve in April 2020",
             "Each line is the whole futures curve on one day. Upward-sloping = contango (paying to store); "
             "downward-sloping = backwardation.")
    footnote(fig, "CME settlements and INE closing prices, contracts with open interest, first 24 months of the curve.")
    save(fig, path)
    return {"CL": cl, "SC": sc, "CL_2022": ukr}


def main() -> int:
    out = chart()
    for k, cs in out.items():
        for d, c in cs.items():
            front = c.iloc[0]
            twelve = c.iloc[min(len(c) - 1, 11)]
            print(f"{k} {d}: front {front['expiry']:%Y-%m} {front['price']:.2f}, "
                  f"+{min(len(c) - 1, 11)} {twelve['expiry']:%Y-%m} {twelve['price']:.2f}, contracts {len(c)}")
    return 0


if __name__ == "__main__":
    main()
