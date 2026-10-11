"""China commodity futures as a second asset class: continuous contracts from raw daily bars.

A futures contract expires, so "the price of rebar" over 16 years is a chain of contracts.
This module builds that chain the way the DolphinDB research pipeline does
(``~/ddb/factor_lab/00_load_futures_daily.dos``), as an independent pandas implementation,
so the platform runs without Docker and the two engines can be reconciled against each other
(``tests/test_futures_cn.py`` on synthetic data; ``research/reconcile_futures.py`` on the real
panel).

Rules, fixed before any backtest:

  * one row per (date, variety); ``sym`` is the variety code (RB, CU, ...).
  * the MAIN contract is the one with the highest open interest that day, and the main never
    rolls back to an earlier expiry (cumulative max of expiry).
  * the contract HELD on day t is the main chosen at the close of t-1, so a return never uses
    information from the day it is earned: ``ret_t`` is the held contract's own
    close_t / close_{t-1} - 1, and it never jumps across contracts on a roll.
  * prices are back-adjusted multiplicatively so close/prev(close) - 1 == ret.
  * carry: annualized log roll yield between the held contract and the next listed expiry
    (positive = backwardation).
  * investable universe on day t: at least 60 days of history and 60-day average traded
    notional not in the bottom 20% that day.

Inputs (CSV, as written by ``~/ddb/pull_futures.py`` / ``pull_specs.py``), in
``config.FUTURES_CN_DIR``:
  futures_daily.csv   date, sym, open, high, low, close, volume, oi   (one row per contract-day)
  contract_specs.csv  exchange, variety, multiplier, ...

Financial futures (CFFEX) are excluded: this asset class is commodities.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import config
from factors.base import FactorData

MIN_AGE_DAYS = 60
NOTIONAL_WINDOW = 60
NOTIONAL_FLOOR_PCT = 20.0


def _contract_identity(raw: pd.DataFrame) -> pd.DataFrame:
    """Expiry key (yyyymm) and a decade-safe contract id from the exchange symbol.

    CZCE writes three digits (AP401 = 2024-01), so the decade comes from the trade date,
    bumped by ten years when that would put expiry in the past (SR001 traded in 2019).
    """
    sym = raw["sym"].astype(str)
    digits = sym.str.replace(r"^[A-Za-z]+", "", regex=True)
    year = raw["date"].dt.year.to_numpy()
    n = digits.str.len().to_numpy()
    mm = pd.to_numeric(digits.str[-2:]).to_numpy()
    yy = np.where(n == 4, 2000 + pd.to_numeric(digits.str[:2]).to_numpy(),
                  year - year % 10 + pd.to_numeric(digits.str[:1]).to_numpy())
    yy = np.where((n == 3) & (yy < year - 1), yy + 10, yy)
    out = raw.assign(expKey=yy * 100 + mm)
    out["cid"] = out["variety"] + out["expKey"].astype(str)
    return out


def build_continuous(daily: pd.DataFrame, specs: pd.DataFrame) -> pd.DataFrame:
    """Long-format continuous panel, one row per (date, sym). See the module docstring."""
    raw = daily.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[raw["close"].notna() & (raw["close"] > 0)]
    raw = raw.drop_duplicates(["date", "sym"], keep="last")
    # Only varieties with a contract spec survive; that also drops trade-at-settlement
    # listings like SC_TAS2306, whose "variety" (SC_TAS) has no spec.
    raw["variety"] = raw["sym"].astype(str).str.replace(r"[0-9]+$", "", regex=True).str.upper()
    spec = specs.loc[specs["exchange"] != "CFFEX", ["variety", "multiplier"]].drop_duplicates("variety")
    raw = raw.merge(spec, on="variety", how="inner")
    raw = _contract_identity(raw).sort_values(["cid", "date"], kind="stable")
    raw["cret"] = raw.groupby("cid")["close"].pct_change(fill_method=None)

    # Main contract per (date, variety): highest OI, never rolling back; held = yesterday's main.
    mx = (raw[raw["oi"] > 0].sort_values(["date", "variety", "oi"], ascending=[True, True, False], kind="stable")
          .drop_duplicates(["date", "variety"]).sort_values(["variety", "date"])[["date", "variety", "expKey"]])
    mx["mainExp"] = mx.groupby("variety")["expKey"].cummax()
    mx["heldExp"] = mx.groupby("variety")["mainExp"].shift(1)
    held = mx.dropna(subset=["heldExp"])[["date", "variety", "heldExp"]].astype({"heldExp": int})

    tot = raw.groupby(["date", "variety"], as_index=False).agg(totVol=("volume", "sum"), totOI=("oi", "sum"))
    px = held.merge(raw, left_on=["date", "variety", "heldExp"], right_on=["date", "variety", "expKey"], how="inner")
    px = px.rename(columns={"variety": "sym_v"})
    px = pd.DataFrame({
        "date": px["date"], "sym": px["sym_v"], "held": px["cid"], "heldExp": px["heldExp"],
        "open": px["open"], "high": px["high"], "low": px["low"], "rawClose": px["close"], "ret": px["cret"],
        "volume": px["volume"], "oi": px["oi"], "mult": px["multiplier"],
    })
    px = px.merge(tot.rename(columns={"variety": "sym"}), on=["date", "sym"], how="left")

    # Term structure: the next listed expiry after the held one (oi > 0), and its close.
    nxt = raw.loc[raw["oi"] > 0, ["date", "variety", "expKey", "close"]].rename(
        columns={"variety": "sym", "expKey": "nExp", "close": "nClose"})
    nxt = nxt.merge(px[["date", "sym", "heldExp"]], on=["date", "sym"], how="inner")
    nxt = (nxt[nxt["nExp"] > nxt["heldExp"]].sort_values(["date", "sym", "nExp"])
           .drop_duplicates(["date", "sym"])[["date", "sym", "nExp", "nClose"]])
    px = px.merge(nxt, on=["date", "sym"], how="left").sort_values(["sym", "date"])

    # Far-leg return (basis momentum input): the next expiry chosen at t-1, its own return at t.
    px["pNExp"] = px.groupby("sym")["nExp"].shift(1)
    far = raw[["date", "variety", "expKey", "cret"]].rename(
        columns={"variety": "sym", "expKey": "pNExp", "cret": "farRet"})
    px = px.merge(far.astype({"pNExp": float}), on=["date", "sym", "pNExp"], how="left")
    months = lambda e: (e // 100) * 12 + e % 100  # noqa: E731
    px["ry"] = np.log(px["rawClose"] / px["nClose"]) * 12.0 / (months(px["nExp"]) - months(px["heldExp"]))

    # Back-adjusted prices: close/prev(close) - 1 == ret, anchored to the latest raw close.
    px = px.sort_values(["sym", "date"]).reset_index(drop=True)
    g = px.groupby("sym")
    adj = (1.0 + px["ret"].fillna(0.0)).groupby(px["sym"]).cumprod()
    px["close"] = adj * (g["rawClose"].transform("last") / adj.groupby(px["sym"]).transform("last"))
    factor = px["close"] / px["rawClose"]
    for col in ("open", "high", "low"):
        px[col] = px[col] * factor

    # Liquidity universe.
    px["notional"] = px["totVol"] * px["rawClose"] * px["mult"]
    px["notionalOI"] = px["totOI"] * px["rawClose"] * px["mult"]
    px["notional60"] = g["notional"].transform(lambda x: x.rolling(NOTIONAL_WINDOW).mean())
    px["age"] = g.cumcount() + 1
    seasoned = px["notional60"].where(px["age"] >= MIN_AGE_DAYS)
    floor = seasoned.groupby(px["date"]).transform(lambda x: np.nanpercentile(x, NOTIONAL_FLOOR_PCT)
                                                   if x.notna().any() else np.nan)
    px["uni"] = (px["age"] >= MIN_AGE_DAYS) & (px["notional60"] >= floor)
    cols = ["date", "sym", "held", "open", "high", "low", "close", "rawClose", "ret", "volume", "oi",
            "totVol", "totOI", "mult", "notional", "notionalOI", "ry", "farRet", "uni"]
    return px[cols].sort_values(["sym", "date"]).reset_index(drop=True)


def load_continuous(source_dir: str | Path | None = None) -> pd.DataFrame:
    """Read the raw CSVs from ``source_dir`` (default ``config.FUTURES_CN_DIR``) and build the panel."""
    source = Path(source_dir or config.FUTURES_CN_DIR)
    daily = pd.read_csv(source / "futures_daily.csv")
    specs = pd.read_csv(source / "contract_specs.csv")
    return build_continuous(daily, specs)


def to_factor_data(px: pd.DataFrame) -> tuple[FactorData, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Long panel -> (FactorData of date x sym panels, eligible mask, extra panels).

    ``extra`` holds the futures-only inputs signals need: ``ry`` (carry), ``farRet``,
    ``oi`` and ``notional``. ``market`` is the equal-weight return index of the investable
    universe (used by factors that need a market reference).
    """
    def panel(col: str) -> pd.DataFrame:
        return px.pivot(index="date", columns="sym", values=col).sort_index()

    eligible = panel("uni").fillna(False).astype(bool)
    ret = panel("ret")
    market_ret = ret.where(eligible).mean(axis=1).fillna(0.0)
    data = FactorData(open=panel("open"), high=panel("high"), low=panel("low"), close=panel("close"),
                      volume=panel("volume"), market=(1.0 + market_ret).cumprod().rename("market"))
    extra = {name: panel(name) for name in ("ry", "farRet", "oi", "notional", "ret")}
    return data, eligible, extra
