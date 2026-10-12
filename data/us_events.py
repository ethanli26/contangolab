"""US corporate events from the local Sharadar archive (BLUEPRINT 11, E1).

Every event carries the date it became public (``avail``): a strategy may act on it at that day's
close at the earliest. Builders return DataFrames with at least ``ticker``, ``avail`` and ``value``.

  earnings_dates   8-K Item 2.02 ("Results of Operations"), EVENTS code 22, Aug 2004 ->
  sue              standardized unexpected earnings from as-reported quarterly EPS (SF1 ARQ),
                   available on the 10-Q/10-K filing date (datekey), 1998 ->
  ear              earnings-announcement return: stock minus equal-weight universe over trading
                   days -1..+1 around the 8-K 2.02, available at the close of day +1
  insider_clusters >= 2 distinct officers/directors buying in the open market (Form 4 code P)
                   within 30 calendar days, available on the filing date of the 2nd buyer, 2008 ->
  dividend_initiations  first dividend after >= 730 days without one (ACTIONS), 1998 ->
"""

import numpy as np
import pandas as pd

from data.sharadar_provider import BULK_DIR

SUE_WINDOW, SUE_MIN = 8, 6
EAR_DEDUP_DAYS = 30
CLUSTER_DAYS, CLUSTER_MIN = 30, 2
DIV_GAP_DAYS = 730


def earnings_dates(tickers=None, bulk_dir=BULK_DIR) -> pd.DataFrame:
    """One row per 8-K Item 2.02 filing; repeats within 30 days (amendments) are dropped."""
    e = pd.read_parquet(bulk_dir / "EVENTS.parquet")
    if tickers is not None:
        e = e[e["ticker"].isin(set(tickers))]
    e = e[e["eventcodes"].astype(str).str.split("|").apply(lambda codes: "22" in codes)]
    e = e.assign(date=pd.to_datetime(e["date"])).sort_values(["ticker", "date"])
    gap = e.groupby("ticker")["date"].diff().dt.days
    return e[gap.isna() | (gap > EAR_DEDUP_DAYS)][["ticker", "date"]].reset_index(drop=True)


def sue(tickers=None, bulk_dir=BULK_DIR) -> pd.DataFrame:
    """SUE = (EPS_q - EPS_{q-4}) / std of the previous 8 such changes (at least 6)."""
    s = pd.read_parquet(bulk_dir / "SF1.parquet", columns=["ticker", "dimension", "calendardate", "datekey", "eps"],
                        filters=[("dimension", "==", "ARQ")])
    if tickers is not None:
        s = s[s["ticker"].isin(set(tickers))]
    s = s.assign(calendardate=pd.to_datetime(s["calendardate"]), datekey=pd.to_datetime(s["datekey"]))
    s = s.dropna(subset=["eps"]).sort_values(["ticker", "calendardate", "datekey"])
    s = s.drop_duplicates(["ticker", "calendardate"], keep="first")             # the original report
    prior = s[["ticker", "calendardate", "eps"]].assign(calendardate=s["calendardate"] + pd.DateOffset(years=1))
    s = s.merge(prior.rename(columns={"eps": "eps_lag4"}), on=["ticker", "calendardate"], how="left")
    s["change"] = s["eps"] - s["eps_lag4"]
    s = s.sort_values(["ticker", "calendardate"])
    s["scale"] = s.groupby("ticker")["change"].transform(
        lambda c: c.shift(1).rolling(SUE_WINDOW, min_periods=SUE_MIN).std())
    s = s[(s["scale"] > 0) & s["change"].notna()]
    return pd.DataFrame({"ticker": s["ticker"], "avail": s["datekey"], "quarter": s["calendardate"],
                         "value": s["change"] / s["scale"]}).reset_index(drop=True)


def universe_index(close: pd.DataFrame, eligible: pd.DataFrame) -> pd.Series:
    """Cumulative equal-weight return of the names eligible each day (the benchmark level)."""
    r = close.pct_change(fill_method=None).where(eligible.reindex_like(close).fillna(False).astype(bool))
    return (1.0 + r.mean(axis=1).fillna(0.0)).cumprod()


def ear(dates: pd.DataFrame, close: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """Abnormal return over trading days -1..+1 around each 8-K 2.02 (day 0 = first trading day >= filing)."""
    idx = close.index
    level = universe_index(close, eligible).to_numpy()
    col = {t: i for i, t in enumerate(close.columns)}
    d = dates[dates["ticker"].isin(col)].copy()
    d["pos"] = idx.searchsorted(d["date"].to_numpy(), side="left")
    d = d[(d["pos"] >= 2) & (d["pos"] + 1 < len(idx))]
    c = close.to_numpy()
    j = d["ticker"].map(col).to_numpy()
    p = d["pos"].to_numpy()
    stock = c[p + 1, j] / c[p - 2, j] - 1.0
    bench = level[p + 1] / level[p - 2] - 1.0
    elig = eligible.reindex_like(close).fillna(False).to_numpy(dtype=bool)[p, j]
    out = pd.DataFrame({"ticker": d["ticker"].to_numpy(), "event_date": idx[p], "avail": idx[p + 1],
                        "value": stock - bench})
    return out[elig & np.isfinite(out["value"].to_numpy())].reset_index(drop=True)


def insider_clusters(tickers=None, bulk_dir=BULK_DIR) -> pd.DataFrame:
    """Filing dates on which >= 2 distinct officers/directors have bought in the trailing 30 days."""
    i = pd.read_parquet(bulk_dir / "SF2.parquet",
                        columns=["ticker", "filingdate", "transactioncode", "isofficer", "isdirector", "ownername"],
                        filters=[("transactioncode", "==", "P")])
    if tickers is not None:
        i = i[i["ticker"].isin(set(tickers))]
    i = i[(i["isofficer"] == "Y") | (i["isdirector"] == "Y")]
    i = i.assign(filingdate=pd.to_datetime(i["filingdate"])).drop_duplicates(["ticker", "filingdate", "ownername"])
    rows = []
    for ticker, g in i.sort_values("filingdate").groupby("ticker", sort=False):
        dates, owners = g["filingdate"].to_numpy(), g["ownername"].to_numpy()
        last_event = None
        for k in range(len(g)):
            start = dates[k] - np.timedelta64(CLUSTER_DAYS, "D")
            window = owners[(dates > start) & (dates <= dates[k])]
            n = len(set(window))
            if n >= CLUSTER_MIN and (last_event is None or dates[k] - last_event > np.timedelta64(CLUSTER_DAYS, "D")):
                rows.append((ticker, dates[k], float(n)))
                last_event = dates[k]
    return pd.DataFrame(rows, columns=["ticker", "avail", "value"])


def dividend_initiations(close: pd.DataFrame, bulk_dir=BULK_DIR) -> pd.DataFrame:
    """First dividend after >= 730 days without one, for a stock with >= 730 days of price history."""
    a = pd.read_parquet(bulk_dir / "ACTIONS.parquet", columns=["date", "action", "ticker"],
                        filters=[("action", "==", "dividend")])
    a = a[a["ticker"].isin(set(close.columns))].assign(date=lambda x: pd.to_datetime(x["date"]))
    a = a.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])
    gap = a.groupby("ticker")["date"].diff()
    first_price = close.apply(pd.Series.first_valid_index)
    seasoned = a["date"] - a["ticker"].map(first_price) >= pd.Timedelta(days=DIV_GAP_DAYS)
    new = (gap.isna() | (gap >= pd.Timedelta(days=DIV_GAP_DAYS))) & seasoned
    return pd.DataFrame({"ticker": a.loc[new, "ticker"], "avail": a.loc[new, "date"], "value": 1.0}).reset_index(drop=True)
