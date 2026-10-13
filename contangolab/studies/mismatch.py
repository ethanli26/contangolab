"""CL2: trading the gap between the curve and inventories (protocol docs/protocols/cl2-mismatch.md).

The theory of storage gives a "fair" slope for any level of inventory (CL1 study 3). When the
curve is more backwardated than inventories justify, bet that it relaxes; when it is more contango
than justified, bet that it tightens. The bet is a calendar spread (long the front contract, short
the next one, or the reverse), so it is a view on the shape of the curve, not on the price.

  MISMATCH  sign = -sign(slope_dev - fitted(inv_dev)), the fit on earlier weeks only
  REVERT    sign = -sign(slope_dev), the control: same trade with no inventory information

Weekly (last trading day of the week), inverse-volatility weights, gross 1 per country, 3 bps per
side per leg. Inventories are used only once published (US: 7 days after the as-of Friday; China:
the day after). Seasonal norms use earlier years only.

    python -m contangolab study mismatch            # tables + chart (does not log)
    python -m contangolab.studies.mismatch --log    # also append the four trials to results/trials.jsonl
"""

import sys

import numpy as np
import pandas as pd

from contangolab.config import RESULTS
from contangolab.engine.evaluate import _block
from contangolab.engine.portfolio import week_ends
from contangolab.engine.trials import log_trial
from contangolab.markets.curves import COUNTRY, continuous
from contangolab.markets.inventory import china_receipts, us_stocks
from contangolab.studies.storage import seasonal_norm
from contangolab.studies.style import BLUE, GRAY, INK, MUTED, ORANGE, footnote, headline, mpl, save, style

PROTOCOL = "cl2-v1"
PUBLISH_LAG_DAYS = {"us": 7, "cn": 1}
MIN_FIT_WEEKS = 104
VOL_WINDOW = 60
COST_BPS_PER_SIDE = 3.0
LEGS = 2
DATASET = {"cn": "cn_futures", "us": "us_futures"}


def markets(country: str) -> list[str]:
    if country == "us":
        return ["CL", "HO", "RB", "NG"]
    info = pd.read_csv(RESULTS / "storage_china_markets.csv")
    return info.loc[info["qualifies"], "sym"].tolist()


def _panel(px: pd.DataFrame, col: str) -> pd.DataFrame:
    return px.pivot(index="date", columns="sym", values=col).sort_index()


def signals(country: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Grid-date x market panels: MISMATCH sign, REVERT sign, liquid flag; plus daily spread returns."""
    px = continuous(country)
    syms = [s for s in markets(country) if s in set(px["sym"])]
    px = px[px["sym"].isin(syms)]
    slope, liquid = _panel(px, "ry"), _panel(px, "uni").fillna(False).astype(bool)
    spread = _panel(px, "ret") - _panel(px, "farRet")
    grid = week_ends(slope.index)
    stocks = china_receipts() if country == "cn" else us_stocks()

    mis, rev = pd.DataFrame(np.nan, index=grid, columns=syms), pd.DataFrame(np.nan, index=grid, columns=syms)
    for sym in syms:
        st = stocks[stocks["sym"] == sym].dropna().sort_values("date").set_index("date")["value"]
        inv = np.log1p(st)
        inv_dev = (inv - seasonal_norm(inv)).dropna()
        usable = pd.DataFrame({"usable": inv_dev.index + pd.Timedelta(days=PUBLISH_LAG_DAYS[country]),
                               "inv_dev": inv_dev.to_numpy()})
        g = pd.merge_asof(pd.DataFrame({"t": grid}), usable, left_on="t", right_on="usable", direction="backward")
        g.index = grid
        s = slope[sym].reindex(grid)
        sdev = s - seasonal_norm(s)
        x, y = g["inv_dev"].to_numpy(), sdev.reindex(grid).to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        m = np.full(len(grid), np.nan)
        for i in range(len(grid)):
            past = ok[:i]
            if past.sum() < MIN_FIT_WEEKS or not ok[i]:
                continue
            b, a = np.polyfit(x[:i][past], y[:i][past], 1)
            m[i] = y[i] - (a + b * x[i])
        mis[sym] = -np.sign(m)
        rev[sym] = -np.sign(sdev.reindex(grid))
    return mis, rev, liquid.reindex(grid).fillna(False), spread


def book(sign: pd.DataFrame, liquid: pd.DataFrame, spread: pd.DataFrame) -> pd.DataFrame:
    """Daily gross / cost / net / turnover of the weekly inverse-vol spread book."""
    vol = spread.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()
    days = spread.index
    rows, prev = [], pd.Series(0.0, index=sign.columns)
    grid = sign.index
    for k, t in enumerate(grid):
        sgn = sign.loc[t].where(liquid.loc[t]).dropna()
        sgn = sgn[sgn != 0]
        iv = (1.0 / vol.loc[t].reindex(sgn.index)).replace([np.inf, -np.inf], np.nan).dropna()
        w = (sgn[iv.index] * iv / iv.sum()).reindex(sign.columns).fillna(0.0) if len(iv) else prev * 0.0
        start = days.searchsorted(t, side="right")
        stop = days.searchsorted(grid[k + 1], side="right") if k + 1 < len(grid) else len(days)
        if start >= stop:
            prev = w
            continue
        block = spread.iloc[start:stop].fillna(0.0)
        gross = block.to_numpy() @ w.to_numpy()
        turnover = float((w - prev).abs().sum())
        cost = np.zeros(len(block))
        cost[0] = turnover * LEGS * COST_BPS_PER_SIDE / 1e4
        rows.append(pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost,
                                  "turnover": np.where(np.arange(len(block)) == 0, turnover, 0.0)}, index=block.index))
        prev = w
    return pd.concat(rows)


def monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Compound daily net into calendar months; drop the warm-up months before the first position."""
    live = daily.index[daily["turnover"] > 0]
    d = daily.loc[live.min():] if len(live) else daily.iloc[:0]
    g = d.groupby(d.index.to_period("M"))
    out = pd.DataFrame({"net": g["net"].apply(lambda r: (1 + r).prod() - 1), "turnover": g["turnover"].sum()})
    out.index = out.index.to_timestamp("M")
    return out


def record(name: str, family: str, country: str, mo: pd.DataFrame) -> dict:
    net = mo["net"].astype(float)
    empty = net.iloc[:0]
    half = net.index[len(net) // 2]
    return {"signal": name, "asset_class": DATASET[country], "family": family, "kind": "spread_ts", "sign": 1,
            "published": None, "cost_bps": COST_BPS_PER_SIDE, "groups": None,
            "start": str(net.index.min().date()), "end": str(net.index.max().date()),
            "avg_turnover": float(mo["turnover"].mean()), **_block(net, empty),
            "h1": _block(net[net.index < half], empty), "h2": _block(net[net.index >= half], empty)}


def run() -> dict:
    out = {}
    for c in ("cn", "us"):
        mis, rev, liquid, spread = signals(c)
        for name, sgn in (("mismatch", mis), ("revert", rev)):
            daily = book(sgn, liquid, spread)
            mo = monthly(daily)
            out[(c, name)] = {"daily": daily, "monthly": mo,
                              "record": record(f"cl2_{name}", "cl2_curve_mismatch" if name == "mismatch" else "cl2_control",
                                               c, mo)}
        # the control is judged on the mismatch book's months so the comparison is like for like
        m_idx = out[(c, "mismatch")]["monthly"].index
        out[(c, "revert_same")] = record("cl2_revert", "cl2_control", c, out[(c, "revert")]["monthly"].reindex(m_idx).dropna())
    return out


def chart(out: dict, verdicts: dict, path=RESULTS / "curve_mismatch.png") -> None:
    plt = mpl()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, c in zip(axes, ("cn", "us")):
        style(ax, f"{COUNTRY[c]}: {len(markets(c))} markets", "Growth of 1 (net of costs)")
        m_idx = out[(c, "mismatch")]["monthly"].index
        for name, color, lab in (("mismatch", BLUE if c == "cn" else ORANGE, "Curve vs inventory mismatch"),
                                 ("revert", GRAY, "Control: curve vs its own norm")):
            mo = out[(c, name)]["monthly"]["net"].reindex(m_idx).fillna(0.0)
            r = out[(c, name)]["record"] if name == "mismatch" else out[(c, "revert_same")]
            ax.plot(m_idx, (1 + mo).cumprod(), color=color, linewidth=1.6,
                    label=f"{lab}  (Sharpe {r['sharpe']:+.2f})")
        ax.axhline(1, color=MUTED, linewidth=0.7)
        ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK)
    fig.subplots_adjust(top=0.78, bottom=0.10, left=0.07, right=0.98, wspace=0.22)
    passed = [COUNTRY[c] for c in ("cn", "us") if verdicts[c]["survivor"]]
    headline(fig, "Trading the gap between the curve and the warehouse"
             + ("" if passed else ": no edge after correction"),
             "Weekly calendar spreads: short the front spread when the curve is tighter than inventories justify, long when "
             f"looser. Survives the pre-registered bar: {', '.join(passed) if passed else 'neither country'}.")
    footnote(fig, "Inverse-volatility weights, gross 1, 3 bps per side per leg. Inventories used only after publication; "
             "fair slope fitted on earlier weeks only.")
    save(fig, path)


def verdicts(out: dict, table: pd.DataFrame | None = None) -> dict:
    """Survivor status from the corrected trial log if logged; else the uncorrected stats only."""
    v = {}
    for c in ("cn", "us"):
        m, r = out[(c, "mismatch")]["record"], out[(c, "revert_same")]
        beats = all((m[h]["sharpe"] or -9) > (r[h]["sharpe"] or -9) for h in ("h1", "h2"))
        row = None
        if table is not None:
            hit = table[(table["signal"] == "cl2_mismatch") & (table["asset_class"] == DATASET[c])]
            row = hit.iloc[0] if len(hit) else None
        v[c] = {"sharpe": m["sharpe"], "t": m["t_stat"], "h1": m["h1"]["sharpe"], "h2": m["h2"]["sharpe"],
                "control_sharpe": r["sharpe"], "beats_control_both_halves": beats,
                "q_all": float(row["q_all"]) if row is not None else None,
                "survivor": bool(row["survivor"]) if row is not None else False}
    return v


def main(log: bool = False) -> int:
    from contangolab.engine.trials import correction_table, load_trials

    out = run()
    if log:
        for c in ("cn", "us"):
            for name in ("mismatch", "revert"):
                log_trial(out[(c, name)]["record"], protocol=PROTOCOL, note="CL2 pre-registered run")
    table = correction_table(load_trials(), protocol=None)
    v = verdicts(out, table if (table["protocol"] == PROTOCOL).any() else None)
    rows = []
    for c in ("cn", "us"):
        for name, rec in (("mismatch", out[(c, "mismatch")]["record"]), ("revert (all months)", out[(c, "revert")]["record"]),
                          ("revert (mismatch months)", out[(c, "revert_same")])):
            rows.append({"country": COUNTRY[c], "signal": name, "months": rec["months"], "start": rec["start"],
                         "sharpe": rec["sharpe"], "t": rec["t_stat"], "ann_return": rec["ann_return"],
                         "h1": rec["h1"]["sharpe"], "h2": rec["h2"]["sharpe"], "turnover": rec["avg_turnover"]})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "curve_mismatch.csv", index=False, float_format="%.4f")
    print(res.round(3).to_string(index=False))
    for c in ("cn", "us"):
        x = v[c]
        q = f"q_all {x['q_all']:.3f}" if x["q_all"] is not None else "not logged"
        print(f"{COUNTRY[c]}: MISMATCH Sharpe {x['sharpe']:+.2f} (t {x['t']:+.2f}; halves {x['h1']:+.2f} / {x['h2']:+.2f}), "
              f"{q}, survivor {x['survivor']}; beats control in both halves: {x['beats_control_both_halves']}")
    chart(out, v)
    return 0


if __name__ == "__main__":
    main(log="--log" in sys.argv[1:])
