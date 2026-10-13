"""CL1 study 1, the Curve Atlas: how often each market is in contango or backwardation, China vs US.

Descriptive (no bar). For every market: the share of its liquid days with the curve in
backwardation (slope > 0), the median slope, and the month-by-month slope as a heatmap.

    python -m contangolab study atlas
"""

import numpy as np
import pandas as pd

from contangolab.config import RESULTS
from contangolab.markets.curves import COUNTRY, NAMES, SECTOR_ORDER, monthly_slopes, sector, slopes
from contangolab.studies.style import (
    BLUE,
    INK,
    INK_2,
    MUTED,
    ORANGE,
    SURFACE,
    curve_cmap,
    footnote,
    headline,
    mpl,
    save,
    style,
)

MATCHED_START = "2016-02-19"   # first US liquid day: the China vs US comparison uses this window for both
MIN_LIQUID_DAYS = 250          # a market needs a year of liquid days to be shown
CLIP = 0.40                    # heatmap color scale: +-40% a year


def table(start: str | None = None) -> pd.DataFrame:
    """Per market: liquid days, share in backwardation, median slope. ``start`` cuts both countries to one window."""
    rows = []
    for c in ("cn", "us"):
        s = slopes(c)
        s = s[s["liquid"] & (s["date"] >= pd.Timestamp(start or "1900-01-01"))]
        g = s.groupby("sym")
        t = pd.DataFrame({"liquid_days": g.size(), "backwardation_share": g["slope"].apply(lambda x: float((x > 0).mean())),
                          "median_slope": g["slope"].median(), "start": g["date"].min().dt.date,
                          "end": g["date"].max().dt.date}).reset_index()
        t.insert(0, "country", c)
        t["sector"] = t["sym"].map(lambda x, c=c: sector(c, x))
        t["name"] = t["sym"].map(NAMES[c]).fillna(t["sym"])
        rows.append(t[t["liquid_days"] >= MIN_LIQUID_DAYS])
    return pd.concat(rows, ignore_index=True)


def sector_table(t: pd.DataFrame) -> pd.DataFrame:
    """Backwardation share by sector, each market weighted equally ("Other" is non-storable indices: left out)."""
    t = t[t["sector"] != "Other"]
    out = t.groupby(["sector", "country"])["backwardation_share"].agg(["mean", "count"]).unstack("country")
    return out.reindex([s for s in SECTOR_ORDER if s in out.index])


def yearly(country: str) -> pd.DataFrame:
    s = slopes(country)
    s = s[s["liquid"]]
    return s.assign(year=s["date"].dt.year, back=s["slope"] > 0).groupby(["year", "sector"])["back"].mean().unstack()


def heatmap(t: pd.DataFrame, path=RESULTS / "curve_atlas.png") -> None:
    plt = mpl()
    from matplotlib.colors import TwoSlopeNorm

    panels = {}
    for c in ("cn", "us"):
        m = monthly_slopes(c, liquid_share=0.5)
        keep = t[t["country"] == c].copy()
        keep["order"] = keep["sector"].map({s: i for i, s in enumerate(SECTOR_ORDER)})
        keep = keep.sort_values(["order", "backwardation_share"], ascending=[True, False])
        panels[c] = (m.reindex(columns=keep["sym"]), keep)
    months = pd.period_range(min(p[0].index.min() for p in panels.values()),
                             max(p[0].index.max() for p in panels.values()), freq="M")
    heights = [len(panels["cn"][1]), len(panels["us"][1])]
    fig, axes = plt.subplots(2, 1, figsize=(12, 0.17 * sum(heights) + 2.6), sharex=True,
                             gridspec_kw={"height_ratios": heights, "hspace": 0.12})
    norm = TwoSlopeNorm(vmin=-CLIP, vcenter=0.0, vmax=CLIP)
    cmap = curve_cmap()
    cmap.set_bad(SURFACE)
    for ax, c in zip(axes, ("cn", "us")):
        m, keep = panels[c]
        grid = m.reindex(months).T.to_numpy(dtype=float)
        ax.imshow(np.ma.masked_invalid(grid), aspect="auto", interpolation="nearest", cmap=cmap, norm=norm,
                  extent=(0, len(months), len(keep), 0))
        ax.set_yticks(np.arange(len(keep)) + 0.5,
                      [f"{n}" for n in keep["name"]], fontsize=6.3, color=INK_2)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        # sector brackets
        pos = pd.Series(range(len(keep)), index=keep["sector"].to_numpy())
        for sec in keep["sector"].unique():
            lo, hi = pos[[sec]].min(), pos[[sec]].max()
            ax.axhline(lo, color=SURFACE, linewidth=1.6)
            ax.text(len(months) + 1.5, (lo + hi + 1) / 2, sec, va="center", ha="left", fontsize=7.5, color=INK_2)
        ax.set_ylabel(f"{COUNTRY[c]} ({len(keep)} markets)", color=BLUE if c == "cn" else ORANGE, fontsize=10,
                      fontweight="semibold", labelpad=8)
    years = [i for i, p in enumerate(months) if p.month == 1]
    axes[-1].set_xticks(years, [str(months[i].year) for i in years], fontsize=8, color=INK_2)
    fig.subplots_adjust(left=0.14, right=0.88, top=1 - 1.25 / fig.get_figheight(), bottom=0.9 / fig.get_figheight())
    headline(fig, f"The Curve Atlas: {len(t)} commodity futures curves, month by month",
             "Red = backwardation (later delivery cheaper: stocks scarce). Blue = contango (later delivery dearer: "
             "stocks ample). Annualized slope of the front of the curve.")
    cax = fig.add_axes([0.14, 0.45 / fig.get_figheight(), 0.3, 0.12 / fig.get_figheight()])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([-CLIP, -0.2, 0, 0.2, CLIP], labels=[f"{-CLIP:.0%} contango", "-20%", "0", "+20%",
                                                       f"+{CLIP:.0%} backwardation"])
    cb.ax.tick_params(labelsize=7, colors=INK_2, length=0)
    cb.outline.set_visible(False)
    footnote(fig, "Liquid months only (blank = not listed or not liquid). China: SHFE, DCE, CZCE, INE, GFEX "
             "closes; US: CME Group settlements from 2015-11. Color clipped at ±40% a year.")
    save(fig, path)


def share_chart(st: pd.DataFrame, path=RESULTS / "backwardation_share.png") -> None:
    plt = mpl()
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    style(ax, "", "Share of liquid days in backwardation")
    sectors = list(st.index)
    x, w = np.arange(len(sectors)), 0.38
    for off, c, color in [(-w / 2, "cn", BLUE), (w / 2, "us", ORANGE)]:
        vals = st[("mean", c)] if ("mean", c) in st.columns else pd.Series(np.nan, index=sectors)
        bars = ax.bar(x + off, vals.fillna(0), width=w - 0.03, color=color, label=f"{COUNTRY[c]}")
        for b, v in zip(bars, vals):
            if pd.notna(v):
                ax.annotate(f"{v:.0%}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=7.5, color=INK)
    ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_xticks(x, [s.replace(" & ", " &\n").replace(" metals", "\nmetals") for s in sectors], fontsize=8)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right", ncol=2)
    fig.subplots_adjust(top=0.84, bottom=0.17, left=0.08, right=0.98)
    headline(fig, "Which commodities run tight?", "Average share of days each sector's curves spend in backwardation "
             "(markets weighted equally; dashed line = half the time)")
    footnote(fig, f"Both countries over the same window, {MATCHED_START[:7]} to 2026-10 (US open-interest data starts "
             "2015-11). Liquid days only. Count of markets per bar in results/curve_atlas_matched.csv.")
    save(fig, path)


def main() -> int:
    t = table()
    t.to_csv(RESULTS / "curve_atlas.csv", index=False, float_format="%.4f")
    matched = table(MATCHED_START)
    matched.to_csv(RESULTS / "curve_atlas_matched.csv", index=False, float_format="%.4f")
    st = sector_table(matched)
    pd.set_option("display.width", 160)
    print(st.round(3).to_string())
    for c in ("cn", "us"):
        print(f"\n{COUNTRY[c]}: backwardation share by year and sector")
        print(yearly(c).round(2).to_string())
    print("\nmost backwardated / most contango markets:")
    print(t.sort_values("backwardation_share")[["country", "name", "backwardation_share", "median_slope"]]
          .round(3).to_string(index=False))
    heatmap(t)
    share_chart(st)
    return 0


if __name__ == "__main__":
    main()
