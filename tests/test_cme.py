"""US futures: CME symbols, Databento records -> continuous-loader input, CFTC timing, the two new signals."""

import zipfile

import numpy as np
import pandas as pd
import pytest

from contangolab.markets.cftc import attach, load_cot
from contangolab.markets.cme import contract_bars, expiry_key
from contangolab.signals.library import basis_momentum, hedging_pressure


def test_cme_symbols_get_the_right_decade():
    sym = pd.Series(["CLZ0", "CLZ0", "CLF1", "NGZ30", "LBRF3", "ZCH9"])
    when = pd.Series(pd.to_datetime(["2010-06-01", "2019-03-01", "2020-12-15", "2021-01-04", "2022-09-01",
                                     "2010-07-01"]))
    out = expiry_key(sym, when)
    assert out["expKey"].tolist() == [201012, 202012, 202101, 203012, 202301, 201903]
    assert out["root"].tolist() == ["CL", "CL", "CL", "NG", "LBR", "ZC"]


def _stats(rows):
    return pd.DataFrame(rows, columns=["ts_ref", "symbol", "stat_type", "price", "quantity", "stat_flags",
                                       "update_action"]).assign(ts_ref=lambda d: pd.to_datetime(d["ts_ref"], utc=True))


def test_contract_bars_use_final_settlements_and_see_open_interest_one_day_late():
    d1, d2 = "2020-03-02", "2020-03-03"
    stats = _stats([
        (d1, "CLK0", 3, 50.0, 0, 0, 1), (d1, "CLK0", 3, 51.0, 0, 1, 1),        # preliminary, then final
        (d1, "CLK0", 9, 0.0, 1000, 0, 1), (d1, "CLK0", 6, 0.0, 300, 0, 1),
        (d2, "CLK0", 3, 52.0, 0, 1, 1), (d2, "CLK0", 9, 0.0, 1200, 0, 1),
    ])
    ohlcv = pd.DataFrame({"open": [50.5], "high": [53.0], "low": [49.0], "symbol": ["CLK0"]},
                         index=pd.DatetimeIndex(["2020-03-02"], tz="UTC"))
    bars = contract_bars(stats, ohlcv).set_index("date")
    assert bars["sym"].unique().tolist() == ["CL2005"]
    assert bars.loc[d1, "close"] == 51.0                                   # the final settlement, not preliminary
    assert np.isnan(bars.loc[d1, "oi"]) and bars.loc[d2, "oi"] == 1000     # day-1 OI only usable on day 2
    assert bars.loc[d1, "high"] == 53.0 and bars.loc[d2, "high"] == 52.0   # no trades: range = settlement


def test_settlements_without_a_final_flag_fall_back_to_the_last_published():
    stats = _stats([("2012-06-14", "CLZ2", 3, 85.30, 0, 4, 1), ("2012-06-14", "CLZ2", 3, 85.39, 0, 0, 1),
                    ("1970-01-01", "CLZ2", 3, 1.0, 0, 3, 1)])                # epoch ts_ref = undefined: dropped
    bars = contract_bars(stats, pd.DataFrame(columns=["open", "high", "low", "symbol"],
                                             index=pd.DatetimeIndex([], tz="UTC")))
    assert bars["close"].tolist() == [85.39] and bars["sym"].tolist() == ["CL1212"]


def _cot_zip(tmp_path, rows):
    df = pd.DataFrame(rows, columns=["Report_Date_as_YYYY-MM-DD", "CFTC_Contract_Market_Code",
                                     "Prod_Merc_Positions_Long_All", "Prod_Merc_Positions_Short_All"])
    with zipfile.ZipFile(tmp_path / "fut_disagg_txt_2019.zip", "w") as z:
        z.writestr("f.txt", df.to_csv(index=False))


def test_cot_is_usable_a_week_later_and_shutdown_reports_wait_for_the_catch_up(tmp_path):
    _cot_zip(tmp_path, [("2018-12-11", "067651", 100, 300), ("2019-01-08", "067651", 100, 100),
                        ("2019-03-12", "067651", 300, 100), ("2019-03-12", "999999", 1, 1)])
    c = load_cot(tmp_path)
    assert c["root"].tolist() == ["CL"] * 3
    assert c["hp"].tolist() == [0.75, 0.5, 0.25]
    assert c["avail"].tolist() == [pd.Timestamp("2018-12-18"), pd.Timestamp("2019-03-15"), pd.Timestamp("2019-03-19")]


def test_attach_never_uses_a_report_before_it_is_usable():
    cot = pd.DataFrame({"root": "CL", "asof": pd.to_datetime(["2020-01-07", "2020-01-14"]),
                        "avail": pd.to_datetime(["2020-01-14", "2020-01-21"]), "hp26": [0.4, 0.6]})
    px = pd.DataFrame({"sym": "CL", "date": pd.to_datetime(["2020-01-13", "2020-01-14", "2020-01-20", "2020-01-21"])})
    out = attach(px, cot)
    assert out["hp26"].tolist()[1:] == [0.4, 0.4, 0.6] and np.isnan(out["hp26"].iloc[0])


def test_basis_momentum_and_hedging_pressure_only_score_the_universe():
    n = 300
    px = pd.DataFrame({"sym": ["A"] * n + ["B"] * n, "date": list(pd.bdate_range("2020-01-01", periods=n)) * 2,
                       "ret": [0.002] * n + [0.0] * n, "farRet": [0.001] * n + [0.0] * n,
                       "hp26": 0.5, "uni": [True] * n + [False] * n})
    bm = basis_momentum(px)
    assert bm.iloc[n - 1] == pytest.approx(np.log1p(0.002) - np.log1p(0.001)) and bm.iloc[:250].isna().all()
    assert bm.iloc[n:].isna().all() and hedging_pressure(px).iloc[n:].isna().all()
