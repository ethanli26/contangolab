"""Futures curves: every listed contract per market and day, and the slope of the front of the curve.

A futures curve is the set of prices for one commodity at different delivery dates. Its shape is
the market's statement about storage:

  * contango (later months dearer, slope < 0): stocks are ample, and the spread between months
    pays someone to store. It cannot get steeper than the full cost of carrying the commodity
    (storage + financing), or buying now and selling forward would be a riskless profit.
  * backwardation (later months cheaper, slope > 0): stocks are scarce and the market pays for
    delivery now. There is no arbitrage bound on how steep it can get.

The slope used everywhere is the roll yield ``ry`` from the continuous-contract build
(``markets.china.build_continuous``): ln(F_held / F_next) x 12 / months between the expiries,
annualized, positive in backwardation. ``F_held`` is the contract held that day and ``F_next`` the
nearest later expiry with open interest. It is the same number the carry signal trades, so the
curve studies and the trading results describe one quantity.

Tickers collide across the two countries (RB is rebar in Shanghai and gasoline in New York), so
anything that mixes them uses ``label`` ("CN RB", "US RB") or the ``country`` column.
"""

from functools import cache
from pathlib import Path

import pandas as pd

from contangolab import config

SECTORS = {
    "cn": {
        "Energy": ["SC", "FU", "LU", "BU", "PG", "ZC"],
        "Ferrous": ["I", "J", "JM", "RB", "HC", "WR", "SS", "SF", "SM"],
        "Base metals": ["CU", "BC", "AL", "AO", "AD", "ZN", "PB", "NI", "SN", "SI", "LC", "PS"],
        "Precious metals": ["AU", "AG", "PT", "PD"],
        "Chemicals": ["MA", "TA", "PX", "PF", "PR", "EG", "EB", "L", "L_F", "PP", "PP_F", "V", "V_F", "BZ", "UR",
                      "SA", "SH", "FG", "BR", "PL"],
        "Grains & oilseeds": ["A", "B", "C", "CS", "M", "Y", "P", "OI", "RM", "RS", "PK", "WH", "PM", "RI", "JR", "RR"],
        "Softs": ["CF", "CY", "SR", "AP", "CJ", "RU", "NR"],
        "Livestock": ["LH", "JD"],
        "Other": ["SP", "OP", "LG", "BB", "FB", "EC"],
    },
    "us": {
        "Energy": ["CL", "HO", "RB", "NG", "BZ"],
        "Base metals": ["HG"],
        "Precious metals": ["GC", "SI", "PL", "PA"],
        "Grains & oilseeds": ["ZC", "ZW", "KE", "ZS", "ZM", "ZL", "ZO", "ZR"],
        "Livestock": ["LE", "HE", "GF", "DC"],
        "Other": ["LBS", "LBR"],
    },
}
SECTOR_ORDER = ["Energy", "Ferrous", "Base metals", "Precious metals", "Chemicals", "Grains & oilseeds", "Softs",
                "Livestock", "Other"]
NAMES = {
    "us": {"CL": "WTI crude", "HO": "Heating oil", "RB": "Gasoline", "NG": "Natural gas", "BZ": "Brent crude",
           "GC": "Gold", "SI": "Silver", "HG": "Copper", "PL": "Platinum", "PA": "Palladium", "ZC": "Corn",
           "ZW": "Wheat", "KE": "KC wheat", "ZS": "Soybeans", "ZM": "Soymeal", "ZL": "Soyoil", "ZO": "Oats",
           "ZR": "Rough rice", "LE": "Live cattle", "HE": "Lean hogs", "GF": "Feeder cattle", "DC": "Class III milk",
           "LBS": "Lumber", "LBR": "Lumber"},
    "cn": {"SC": "Crude oil", "FU": "Fuel oil", "LU": "Low-sulfur fuel oil", "BU": "Bitumen", "PG": "LPG",
           "ZC": "Thermal coal", "I": "Iron ore", "J": "Coke", "JM": "Coking coal", "RB": "Rebar", "HC": "Hot-rolled coil",
           "WR": "Wire rod", "SS": "Stainless steel", "SF": "Ferrosilicon", "SM": "Silicomanganese", "CU": "Copper",
           "BC": "Copper (intl)", "AL": "Aluminium", "AO": "Alumina", "AD": "Aluminium alloy", "ZN": "Zinc", "PB": "Lead",
           "NI": "Nickel", "SN": "Tin", "SI": "Industrial silicon", "LC": "Lithium carbonate", "PS": "Polysilicon",
           "AU": "Gold", "AG": "Silver", "PT": "Platinum", "PD": "Palladium", "MA": "Methanol", "TA": "PTA",
           "PX": "Paraxylene", "PF": "Polyester fiber", "PR": "PET bottle chip", "EG": "Ethylene glycol",
           "EB": "Styrene", "L": "Polyethylene", "PP": "Polypropylene", "V": "PVC", "BZ": "Benzene", "UR": "Urea",
           "SA": "Soda ash", "SH": "Caustic soda", "FG": "Glass", "BR": "Butadiene rubber", "PL": "Propylene",
           "A": "Soybeans (domestic)", "B": "Soybeans (imported)", "C": "Corn", "CS": "Corn starch", "M": "Soymeal",
           "Y": "Soyoil", "P": "Palm oil", "OI": "Rapeseed oil", "RM": "Rapeseed meal", "RS": "Rapeseed",
           "PK": "Peanuts", "WH": "Wheat", "PM": "Wheat (common)", "RI": "Rice", "JR": "Japonica rice",
           "RR": "Japonica rice", "CF": "Cotton", "CY": "Cotton yarn", "SR": "Sugar", "AP": "Apples",
           "CJ": "Red dates", "RU": "Natural rubber", "NR": "TSR 20 rubber", "LH": "Live hogs", "JD": "Eggs",
           "SP": "Pulp", "OP": "Offset paper", "LG": "Logs", "BB": "Plywood", "FB": "Fiberboard",
           "EC": "Container freight (Europe)"},
}
# The same commodity listed in both countries (CL1 protocol, study 2).
TWINS = [("Copper", "CU", "HG"), ("Gold", "AU", "GC"), ("Silver", "AG", "SI"), ("Crude oil", "SC", "CL"),
         ("Soybeans", "B", "ZS"), ("Soymeal", "M", "ZM"), ("Soyoil", "Y", "ZL"), ("Corn", "C", "ZC"),
         ("Hogs", "LH", "HE")]
COUNTRY = {"cn": "China", "us": "US"}
DATASET = {"cn": "cn_futures", "us": "us_futures"}


def sector(country: str, sym: str) -> str:
    return next((s for s, syms in SECTORS[country].items() if sym in syms), "Other")


def label(country: str, sym: str) -> str:
    return f"{country.upper()} {sym}"


@cache
def continuous(country: str) -> pd.DataFrame:
    """The continuous-contract panel for one country (one row per date x market), loaded once per process."""
    if country == "cn":
        from contangolab.markets.china import load_continuous
    else:
        from contangolab.markets.cme import load_continuous
    return load_continuous()


def slopes(country: str) -> pd.DataFrame:
    """Long table: date, sym, slope (= ry), liquid (bool), ret, sector, country."""
    px = continuous(country)
    out = px[["date", "sym", "ry", "uni", "ret"]].rename(columns={"ry": "slope", "uni": "liquid"})
    out = out.assign(liquid=out["liquid"].astype(bool), country=country,
                     sector=out["sym"].map(lambda s: sector(country, s)))
    return out.dropna(subset=["slope"]).reset_index(drop=True)


def monthly_slopes(country: str, liquid_share: float = 0.0) -> pd.DataFrame:
    """Month x market mean slope over liquid days; months where the market is liquid on fewer than
    ``liquid_share`` of its days are dropped."""
    s = slopes(country)
    s = s.assign(month=s["date"].dt.to_period("M"))
    share = s.groupby(["sym", "month"])["liquid"].mean()
    mean = s[s["liquid"]].groupby(["sym", "month"])["slope"].mean()
    keep = share[share >= liquid_share].index if liquid_share > 0 else share[share > 0].index
    return mean.reindex(keep).dropna().unstack("sym").sort_index()


@cache
def contracts(country: str) -> pd.DataFrame:
    """Every listed contract per day: date, sym (market), expKey (yyyymm), close, oi."""
    if country == "cn":
        from contangolab.markets.china import _contract_identity

        source = Path(config.FUTURES_CN_DIR)
        raw = pd.read_csv(source / "futures_daily.csv", usecols=["date", "sym", "close", "oi"])
        raw["date"] = pd.to_datetime(raw["date"])
        raw["variety"] = raw["sym"].astype(str).str.replace(r"[0-9]+$", "", regex=True).str.upper()
        raw = _contract_identity(raw[raw["sym"].astype(str).str.fullmatch(r"[A-Za-z]+\d{3,4}")])
        return raw[["date", "variety", "expKey", "close", "oi"]].rename(columns={"variety": "sym"})
    from contangolab.markets.cme import DIR, contract_bars

    stats = pd.read_parquet(Path(DIR) / "statistics.parquet")
    bars = contract_bars(stats, pd.DataFrame(columns=["open", "high", "low", "symbol"],
                                             index=pd.DatetimeIndex([], tz="UTC")))
    bars["expKey"] = ("20" + bars["sym"].str[-4:-2]).astype(int) * 100 + bars["sym"].str[-2:].astype(int)
    bars["sym"] = bars["sym"].str[:-4]
    return bars[["date", "sym", "expKey", "close", "oi"]]


def curve(country: str, sym: str, date: str, min_oi: float = 1.0) -> pd.DataFrame:
    """The curve on one day: expiry (as a Timestamp), price and open interest of every contract with OI."""
    c = contracts(country)
    day = c[(c["sym"] == sym) & (c["date"] == pd.Timestamp(date)) & (c["oi"].fillna(0) >= min_oi)]
    day = day.sort_values("expKey")
    expiry = pd.to_datetime((day["expKey"] * 100 + 1).astype(str), format="%Y%m%d")
    return pd.DataFrame({"expiry": expiry.to_numpy(), "price": day["close"].to_numpy(), "oi": day["oi"].to_numpy()})
