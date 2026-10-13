"""US commodity futures (CME Group: NYMEX, COMEX, CBOT, CME) as continuous contracts.

The continuous series is built by the SAME code as China's (``contangolab.markets.china.build_continuous``),
so both markets follow identical, pre-fixed rules: main = highest open interest, never rolling
back; the contract held on t is the main chosen at t-1; returns never jump across contracts;
carry = annualized log roll yield to the next listed expiry; the same liquidity universe.
This module only translates Databento's contract records into that loader's input.

Inputs (``data/cache/futures_us/``, written by ``contangolab.markets.databento``):
  statistics.parquet  per contract and trade date: final settlement price, open interest,
                      cleared volume (CME's official end-of-day figures)
  ohlcv_1d*.parquet   per contract and UTC day: open/high/low (signals that need a range)

US-specific choices, fixed before any backtest:
  * close = the FINAL settlement price of the trade date. All contracts settle at the same time,
    so carry compares like with like (last-trade prices of thin back months can be stale).
    CME's feed only flags settlements final from ~2016-17; before that (flags 0 or 4) the close
    is the last settlement published for the trade date. Where both exist (1.61M contract-days,
    2017-06 on) the last published equals the final one 100.0% of the time.
  * open interest for trade date t arrives that evening (preliminary) and is revised the next
    morning, so the loader sees each contract's OI one trading day late (the main chosen at the
    close of t uses OI of t-1).
  * symbols: CME codes ``root + month letter + year`` (CLZ0, CLZ30). One-digit years are
    resolved to the decade that is not in the past at the trade date (as for CZCE).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from contangolab import config
from contangolab.markets.china import build_continuous

DIR = Path(config.FUTURES_US_DIR)
MONTHS = {c: i + 1 for i, c in enumerate("FGHJKMNQUVXZ")}
SYMBOL = r"(?P<root>[A-Z]{2,3}?)(?P<month>[FGHJKMNQUVXZ])(?P<year>\d{1,2})"
SETTLEMENT, CLEARED_VOLUME, OPEN_INTEREST = 3, 6, 9
FINAL_FLAG = 1                                   # stat_flags bit 0: final (not preliminary) settlement
NEW = 1                                          # update_action: 1 = new value, 2 = delete

# exchange, variety, multiplier ($ per one unit of the quoted price)
SPECS = pd.DataFrame([
    ("NYMEX", "CL", 1000), ("NYMEX", "HO", 42000), ("NYMEX", "RB", 42000), ("NYMEX", "NG", 10000),
    ("NYMEX", "BZ", 1000), ("COMEX", "GC", 100), ("COMEX", "SI", 5000), ("COMEX", "HG", 25000),
    ("NYMEX", "PL", 50), ("NYMEX", "PA", 100),
    ("CBOT", "ZC", 50), ("CBOT", "ZW", 50), ("CBOT", "KE", 50), ("CBOT", "ZS", 50), ("CBOT", "ZM", 100),
    ("CBOT", "ZL", 600), ("CBOT", "ZO", 50), ("CBOT", "ZR", 2000),
    ("CME", "LE", 400), ("CME", "HE", 400), ("CME", "GF", 500), ("CME", "DC", 2000),
    ("CME", "LBS", 110), ("CME", "LBR", 27.5),
], columns=["exchange", "variety", "multiplier"])


def expiry_key(symbol: pd.Series, trade_date: pd.Series) -> pd.DataFrame:
    """Root and yyyymm expiry of each CME symbol; one-digit years take the first decade >= trade year - 1."""
    parts = symbol.str.extract(f"^{SYMBOL}$")
    year = pd.to_datetime(trade_date).dt.year.to_numpy()
    digits = parts["year"].fillna("")
    d = pd.to_numeric(digits, errors="coerce").to_numpy()
    yy = np.where(digits.str.len() == 2, 2000 + d, year - year % 10 + d)
    yy = np.where((digits.str.len() == 1) & (yy < year - 1), yy + 10, yy)
    mm = parts["month"].map(MONTHS).to_numpy()
    return pd.DataFrame({"root": parts["root"].to_numpy(), "expKey": yy * 100 + mm}, index=symbol.index)


def contract_bars(stats: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Databento records -> ``build_continuous`` input: date, sym (root + yymm), open/high/low/close, volume, oi."""
    st = stats[(stats["update_action"] == NEW) & (pd.to_datetime(stats["ts_ref"]) > pd.Timestamp(0, tz="UTC"))]
    st = st.sort_index(kind="stable")                                    # ts_recv: arrival order
    st = st.assign(date=pd.to_datetime(st["ts_ref"]).dt.tz_localize(None).dt.normalize())
    settles = st[st["stat_type"] == SETTLEMENT]
    final = settles[(settles["stat_flags"] & FINAL_FLAG) > 0]
    final = final.drop_duplicates(["date", "symbol"], keep="last").set_index(["date", "symbol"])["price"]
    latest = settles.drop_duplicates(["date", "symbol"], keep="last").set_index(["date", "symbol"])["price"]
    settle = final.combine_first(latest)
    oi = st[st["stat_type"] == OPEN_INTEREST].drop_duplicates(["date", "symbol"], keep="last")
    oi = oi.set_index(["date", "symbol"])["quantity"]
    vol = st[st["stat_type"] == CLEARED_VOLUME].drop_duplicates(["date", "symbol"], keep="last")
    vol = vol.set_index(["date", "symbol"])["quantity"]
    bars = pd.DataFrame({"close": settle}).join(oi.rename("oi_t")).join(vol.rename("volume")).reset_index()

    o = ohlcv.assign(date=ohlcv.index.tz_convert(None).normalize()).reset_index(drop=True)
    o = o[o["date"].dt.dayofweek < 5].drop_duplicates(["date", "symbol"], keep="last")
    bars = bars.merge(o[["date", "symbol", "open", "high", "low"]], on=["date", "symbol"], how="left")
    for col in ("open", "high", "low"):          # no trade that UTC day: the range collapses to the settlement
        bars[col] = bars[col].fillna(bars["close"])
    bars["high"] = bars[["high", "close"]].max(axis=1)
    bars["low"] = bars[["low", "close"]].min(axis=1)

    key = expiry_key(bars["symbol"], bars["date"])
    bars = bars[key["root"].isin(set(SPECS["variety"]))].assign(root=key["root"], expKey=key["expKey"])
    bars["sym"] = bars["root"] + (bars["expKey"] // 100 % 100).map("{:02d}".format) + \
        (bars["expKey"] % 100).map("{:02d}".format)
    bars = bars.sort_values(["sym", "date"])
    bars["oi"] = bars.groupby("sym")["oi_t"].shift(1)                    # OI is known one day late
    bars["volume"] = bars["volume"].fillna(0.0)
    return bars[["date", "sym", "open", "high", "low", "close", "volume", "oi"]].reset_index(drop=True)


def load_continuous(source_dir: str | Path | None = None) -> pd.DataFrame:
    source = Path(source_dir or DIR)
    stats = pd.read_parquet(source / "statistics.parquet")
    ohlcv = pd.concat([pd.read_parquet(source / f, columns=["open", "high", "low", "symbol"])
                       for f in ("ohlcv_1d.parquet", "ohlcv_1d_extra.parquet") if (source / f).exists()])
    return build_continuous(contract_bars(stats, ohlcv), SPECS)
