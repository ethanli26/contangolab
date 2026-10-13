"""CFTC Commitments of Traders (disaggregated, futures only) for the US commodity roots.

Hedging pressure (Basu & Miffre 2013): the share of producer/merchant/processor/user positions
that are short, short / (long + short). Hedgers are mostly short; when they are MORE short than
usual for that market, they pay speculators a larger premium to take the other side.

Timing: positions are as of Tuesday and published the following Friday at 15:30 ET, after the
day's settlement. A report is treated as usable at the close of the Tuesday one week after its
as-of date (``LAG_DAYS``). During government shutdowns the CFTC published late and caught up
over weeks; reports dated inside those windows are only usable from a conservative date after
the catch-up finished (``SHUTDOWNS``; dates chosen generously late, not taken from the
publication log).

Files: ``data/cache/cftc/fut_disagg_txt_*.zip`` from https://www.cftc.gov/files/dea/history/.
"""

import glob
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from contangolab import config

DIR = Path(config.CFTC_DIR)
LAG_DAYS = 7
HP_WEEKS = 26
SHUTDOWNS = [("2013-09-24", "2013-10-29", "2013-11-15"),        # (first as-of, last as-of, usable from)
             ("2018-12-18", "2019-02-26", "2019-03-15"),
             ("2025-09-23", "2025-12-30", "2026-02-06")]
CODES = {"067651": "CL", "022651": "HO", "111659": "RB", "023651": "NG", "06765T": "BZ",
         "088691": "GC", "084691": "SI", "085692": "HG", "076651": "PL", "075651": "PA",
         "002602": "ZC", "001602": "ZW", "001612": "KE", "005602": "ZS", "026603": "ZM", "007601": "ZL",
         "004603": "ZO", "039601": "ZR", "057642": "LE", "054642": "HE", "061641": "GF", "052641": "DC",
         "058643": "LBS", "058644": "LBR"}


def load_cot(source_dir: str | Path | None = None) -> pd.DataFrame:
    """One row per (root, as-of date): hedger long/short, hedging pressure, and the date it is usable."""
    frames = []
    for f in sorted(glob.glob(str(Path(source_dir or DIR) / "fut_disagg_txt_*.zip"))):
        with zipfile.ZipFile(f) as z:
            frames.append(pd.read_csv(z.open(z.namelist()[0]), low_memory=False, dtype={"CFTC_Contract_Market_Code": str},
                                      usecols=["Report_Date_as_YYYY-MM-DD", "CFTC_Contract_Market_Code",
                                               "Prod_Merc_Positions_Long_All", "Prod_Merc_Positions_Short_All"]))
    c = pd.concat(frames)
    c["root"] = c["CFTC_Contract_Market_Code"].str.strip().map(CODES)
    c = c.dropna(subset=["root"])
    out = pd.DataFrame({"root": c["root"], "asof": pd.to_datetime(c["Report_Date_as_YYYY-MM-DD"]),
                        "long": c["Prod_Merc_Positions_Long_All"].astype(float),
                        "short": c["Prod_Merc_Positions_Short_All"].astype(float)})
    out = out.drop_duplicates(["root", "asof"], keep="last").sort_values(["root", "asof"])
    out["hp"] = out["short"] / (out["long"] + out["short"]).replace(0.0, np.nan)
    out["avail"] = out["asof"] + pd.Timedelta(days=LAG_DAYS)
    for first, last, usable in SHUTDOWNS:
        inside = out["asof"].between(pd.Timestamp(first), pd.Timestamp(last))
        out.loc[inside, "avail"] = out.loc[inside, "avail"].clip(lower=pd.Timestamp(usable))
    out["avail"] = out.groupby("root")["avail"].cummax()       # a later report is never usable before an earlier one
    out["hp26"] = out.groupby("root")["hp"].transform(lambda x: x.rolling(HP_WEEKS, min_periods=HP_WEEKS).mean())
    return out.reset_index(drop=True)


def attach(px: pd.DataFrame, cot: pd.DataFrame, col: str = "hp26") -> pd.DataFrame:
    """Add ``col`` to the continuous panel: per sym, the latest report usable on or before each date.

    A shutdown catch-up makes several reports usable on the same day; the latest as-of wins.
    """
    usable = (cot.dropna(subset=[col]).sort_values(["avail", "asof"])
              .drop_duplicates(["root", "avail"], keep="last")[["root", "avail", col]])
    px = px.sort_values("date")
    out = pd.merge_asof(px, usable.rename(columns={"root": "sym", "avail": "date"}).sort_values("date"),
                        on="date", by="sym", direction="backward")
    return out.sort_values(["sym", "date"]).reset_index(drop=True)
