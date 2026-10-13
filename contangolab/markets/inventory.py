"""Inventories and interest rates: what the theory of storage says should shape the curve.

  * China: exchange warehouse receipts (仓单), weekly, SHFE / INE / CZCE, from 2015-10
    (``receipts.csv`` in ``config.FUTURES_CN_DIR``). Receipts are only the stock registered for
    delivery against the exchange, not all inventory in the country, but they are what a short
    can deliver, which is the part of inventory that sets the spread between months.
  * US: EIA weekly stocks (public domain), each paired with the futures market it supplies:
        Cushing, Oklahoma crude  -> CL  (the WTI contract's delivery point)
        distillate fuel oil      -> HO
        motor gasoline           -> RB
        Lower-48 working gas     -> NG
    Dated by the week-ending Friday the stocks are measured on (published the next Wednesday or
    Thursday).
  * 3-month interest rates (FRED, monthly): US Treasury bill (TB3MS) and China interbank
    (IR3TIB01CNM156N, OECD). Gold's contango should be about the cost of financing it.

    python -m contangolab.markets.inventory      # download the EIA and FRED series into data/cache/eia/

Both sources are public and need no key (EIA's full-history spreadsheets need ``xlrd`` to read).
"""

import io
import time
from pathlib import Path

import pandas as pd

from contangolab import config

EIA_SERIES = {"cushing": "pet/hist_xls/W_EPC0_SAX_YCUOK_MBBLw", "distillate": "pet/hist_xls/WDISTUS1w",
              "gasoline": "pet/hist_xls/WGTSTUS1w", "natgas": "ng/hist_xls/NW2_EPG0_SWO_R48_BCFw"}
US_PAIRS = {"CL": "cushing", "HO": "distillate", "RB": "gasoline", "NG": "natgas"}
FRED_SERIES = {"us_3m": "TB3MS", "cn_3m": "IR3TIB01CNM156N"}
DIR = Path(config.EIA_DIR)


def _get(url: str, tries: int = 4) -> bytes:
    import requests

    for attempt in range(tries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def download() -> None:
    """EIA's full-history spreadsheets (no API key needed) and FRED's CSVs, into data/cache/eia/."""
    DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, path in EIA_SERIES.items():
        sheet = pd.read_excel(io.BytesIO(_get(f"https://www.eia.gov/dnav/{path}.xls")), sheet_name="Data 1", skiprows=2)
        frames.append(pd.DataFrame({"date": pd.to_datetime(sheet.iloc[:, 0]), "series": name,
                                    "value": pd.to_numeric(sheet.iloc[:, 1], errors="coerce")}))
        print(f"{name}: {len(sheet)} weeks")
    pd.concat(frames).sort_values(["series", "date"]).to_csv(DIR / "eia_weekly.csv", index=False)
    rates = []
    for name, sid in FRED_SERIES.items():
        df = pd.read_csv(io.BytesIO(_get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")))
        df.columns = ["date", "value"]
        rates.append(df.assign(series=name, value=pd.to_numeric(df["value"], errors="coerce")))
        print(f"{name}: {len(df)} months")
    pd.concat(rates).to_csv(DIR / "rates_monthly.csv", index=False)


def us_stocks() -> pd.DataFrame:
    """Weekly EIA stocks, long: date, sym (the paired futures market), value."""
    df = pd.read_csv(DIR / "eia_weekly.csv", parse_dates=["date"])
    back = {v: k for k, v in US_PAIRS.items()}
    return df.assign(sym=df["series"].map(back)).dropna(subset=["sym", "value"])[["date", "sym", "value"]]


def china_receipts() -> pd.DataFrame:
    """Weekly warehouse receipts, long: date, sym, value."""
    df = pd.read_csv(Path(config.FUTURES_CN_DIR) / "receipts.csv")
    return pd.DataFrame({"date": pd.to_datetime(df["date"].astype(str)), "sym": df["variety"],
                         "value": pd.to_numeric(df["receipt"], errors="coerce")}).dropna()


def rates() -> pd.DataFrame:
    """Monthly 3-month rates in percent, columns us_3m and cn_3m, indexed by month."""
    df = pd.read_csv(DIR / "rates_monthly.csv", parse_dates=["date"])
    return df.pivot(index="date", columns="series", values="value").to_period("M")


if __name__ == "__main__":
    download()
