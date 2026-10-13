# CL1: what the futures curve says. Pre-registration, written before any curve study was run

Protocol tag `cl1`, written 2026-10-12.

**What has been checked so far:**
- which markets and inventory series exist;
- their start dates;
- how often the China warehouse receipts are zero.

**What has not been computed yet:** no slope, correlation or volatility.

Studies 1 and 5 are descriptive. Studies 2, 3 and 4 each have a bar written here. A miss is reported as a miss, and nothing below gets revised after the fact. If a definition turns out to be unworkable, it is fixed in a dated amendment before results are read, the same way C1's amendment A1 was.

## Common definitions

**Curve slope.** The roll yield `ry` from the continuous-contract build (`contangolab/markets/china.py`). It is the same number the carry signal uses:
- **Formula:** `ry = ln(F_held / F_next) × 12 / months between expiries`.
  - `F_held` is the contract held that day (yesterday's highest-open-interest contract).
  - `F_next` is the nearest later expiry with open interest > 0.
- **Sign:** positive means backwardation, negative means contango.
- **Prices:** China uses daily closes. The US uses CME final settlements.
- **Monthly value:** the mean over the month's days on which the market is in the liquid universe.

**Liquid universe.** The rule from the trading research: at least 60 days of history, and 60-day traded notional not in the bottom 20% that day.

**Samples:**
- **China:** 71 markets on SHFE, DCE, CZCE, INE and GFEX. The data starts 2010 for SHFE and 2019 for DCE.
- **US:** 24 CME Group markets from 2015-11, when Databento's open-interest data begins.
- **Both:** the sample ends 2026-10-09.

## Study 1: Curve Atlas (descriptive)

**What is reported:**
- the share of liquid days in backwardation, per market and per sector, China vs US;
- a heatmap of monthly slope by market and month;
- backwardation share by year.

**Written expectations (not bars):**
- Gold and silver are in contango on more than 90% of days in both countries, because their curves only reflect the cost of financing.
- Grains and oilseeds are mostly in contango.
- Energy switches regime:
  - contango in 2015–16 and 2020;
  - backwardation in 2021–22.
- China's ferrous markets (iron ore, coking coal, coke, rebar, hot-rolled coil) are backwardated more often than not.

## Study 2: Twin curves (descriptive, plus test H2)

**Pairs (China ↔ US):**

| Commodity | China | US |
|---|---|---|
| Copper | SHFE CU | COMEX HG |
| Gold | SHFE AU | COMEX GC |
| Silver | SHFE AG | COMEX SI |
| Crude oil | INE SC | NYMEX CL |
| Soybeans | DCE B (imported No.2 soybean) | CBOT ZS |
| Soymeal | DCE M | CBOT ZM |
| Soyoil | DCE Y | CBOT ZL |
| Corn | DCE C | CBOT ZC |
| Hogs | DCE LH | CME HE |

- **Excluded:**
  - GFEX platinum and palladium: listed 2025-11, less than a year of data.
  - CZCE wheat: illiquid.
- **Which months count:** both markets must be in the liquid universe on at least half the month's days.
- **What is reported per pair:**
  - the correlation of monthly slopes;
  - the share of months with the same sign;
  - each side's mean slope.

**Written expectations:**
- Correlation is highest for metals.
- It is lower for the soy complex, because China's curve there follows crush margins and state reserve policy.
- It is lowest for hogs, which have no trade link and have their own disease cycle (African swine fever).

**H2 (cost of carry, gold).** Gold is never scarce and is cheap to store, so its contango should be roughly the local short-term interest rate.
- **Interest rates:**
  - US: 3-month Treasury bill (FRED TB3MS, monthly).
  - China: 3-month interbank rate (FRED/OECD IR3TIB01CNM156N, monthly).
- **Bar:**
  - (a) In each country, a regression of monthly −slope on the local rate has a coefficient between 0.5 and 1.5 and R² ≥ 0.5.
  - (b) The gap in contango between the two countries tracks the gap in interest rates with correlation ≥ 0.7. Contango gap: (−slope_US) − (−slope_CN). Rate gap: US rate − China rate.
- **Silver** gets the same statistics reported, but no bar.

## Study 3: Inventories → curve (the theory of storage; test H3)

**Inventory series (weekly):**
- **China:** exchange warehouse receipts for SHFE, INE and CZCE (`receipts.csv`, from 2015-10). The data has no DCE receipts.
  - A market qualifies if:
    - receipts are zero in at most 25% of weeks;
    - it has at least 156 weeks with both receipts and slope;
    - it is in the liquid universe in at least half of them.
  - Caveat: receipts count only exchange-registered stock, not all inventory.
- **US:** EIA weekly stocks, each paired with one market:

  | EIA series | Paired with |
  |---|---|
  | Cushing, Oklahoma crude stocks (W_EPC0_SAX_YCUOK_MBBL) | CL |
  | Distillate stocks (WDISTUS1) | HO |
  | Gasoline stocks (WGTSTUS1) | RB |
  | Lower-48 working gas in storage (NW2_EPG0_SWO_R48_BCF) | NG |

**Measures, per market, at each inventory date d:**
- The slope is the `ry` of the last trading day on or before d.
- **Seasonal norm** (applied to log(1 + inventory) and to slope separately): the mean over the same ISO week ±2 in all **earlier** years. At least 3 earlier years are required.
- `inv_dev` = log(1 + inventory) minus its seasonal norm. It is then z-scored within the market.
- `slope_dev` = slope minus its seasonal norm.

**Per-market regression:**
- `slope_dev = a + b × inv_dev_z`, with Newey–West standard errors (12 lags).
- The theory predicts b < 0: above-normal stocks mean deeper contango.

**H3 bar:**
- **China:** b < 0 in at least 2/3 of qualifying markets, and a one-sided binomial sign test p < 0.01.
- **US:** b < 0 with Newey–West t ≤ −2 for at least 3 of the 4 pairs.

**Reported, not part of the bar:** the shape of the "storage curve". Pool China's markets, bin `inv_dev_z` into deciles, and report the mean `slope_dev` per decile. The theory predicts a steep curve when stocks are low and a flat one when stocks are high, because contango is capped by the full cost of carry.

## Study 4: Curve and volatility (test H4)

**Measures, per market, daily:**
- Slope tercile within the market, over its liquid days in the full sample.
- Forward volatility: the annualized standard deviation of the held contract's returns over the next 21 trading days.

**H4:** forward volatility in the most-backwardated tercile is higher than in the middle tercile. Low inventories leave no buffer against shocks.
- **Bar, for each country separately:** true in at least 2/3 of liquid markets, and a one-sided sign test p < 0.01.
- **Reported, not part of the bar:** contango tercile vs middle tercile. Kogan, Livdan & Yaron (2009) find a V shape.

## Study 5: Case studies (descriptive)

Full curves (every listed contract with open interest > 0) for:
- CL and INE SC on 2020-01-02, 2020-04-17, 2020-04-20, 2020-04-21 and 2020-07-01 (super-contango, and May WTI at −$37.63);
- CL on 2022-03-08 (backwardation after the invasion of Ukraine).

## Not done

- Nothing here is a trading signal. These studies explain curve shape; whether the curve pays was already tested (carry: China Sharpe 0.30, q 0.28; US −0.11).
- Nothing here is added to `results/trials.jsonl`, which holds return-signal trials. The verdicts go in `docs/protocols/cl1-results.md` and the vault.
