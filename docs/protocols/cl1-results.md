# CL1 results

Rules: [`cl1-curve-studies.md`](cl1-curve-studies.md), committed before any study ran. Run on 2026-10-12. Regenerate with `python -m contangolab study all`.

**Summary:**
- Two of the three pre-registered tests passed in both countries: the theory of storage, and backwardation as a warning of volatility.
- The third, gold's contango as an interest rate, failed: the US side passed and the China side did not.
- Three of the written expectations missed. All misses are reported below as misses.

## A data fix found while running (before any test result was read)

The April 2020 case study showed May WTI (CLK0) at −$37.63 on **17 April**. The true settlement that day was 18.27; −37.63 belongs to 20 April.

- **Cause:** at 21:43 UTC on 20 April, CME sent a settlement record flagged final but labeled with the previous trade date (17 April), carrying the 20 April price. The loader kept the last final record per date, so this one overwrote the real 18.27.
- **Scope:** in all of 2.03M contract-days, 3 have such a record (CLK0 2020-04-17, ZLV1 2021-04-19, ZLU5 2025-03-31).
- **Fix:** `markets/cme.py` ignores a settlement that arrives after a newer trade date's settlement for the same contract was published. A regression test covers it (`tests/test_cme.py`).
- **Effect on C1:** none. All 8 US trading results were rerun, and Sharpe ratios and t-stats are unchanged. Monthly returns differ by under 1e-16, because none of the three contracts was held or used as the next contract on those days.

## Study 1: Curve Atlas (descriptive)

Charts: `results/curve_atlas.png`, `results/backwardation_share.png`. Tables: `results/curve_atlas*.csv`.

Each market's share of liquid days in backwardation, averaged within the sector (markets weighted equally). Both countries use the same window, 2016-02 → 2026-10:

| Sector | China | US |
|---|---|---|
| Energy | 70% (6 markets) | 58% (5) |
| Ferrous | 74% (8) | — |
| Base metals | 42% (11) | 11% (1) |
| Precious metals | 4% (2) | 1% (3) |
| Chemicals | 54% (15) | — |
| Grains & oilseeds | 54% (10) | 20% (6) |
| Softs | 34% (6) | — |
| Livestock | 55% (2) | 38% (3) |

**Expectations, as written:**

| Expectation | Outcome |
|---|---|
| Gold and silver in contango on more than 90% of days in both countries | **Met.** US: 99.7% (gold) and 99.8% (silver). China, same window: 97.9% and 93.5%. Over China's full 2010–26 sample, gold is lower at 89%, because 2010–15 had backwardated spells. |
| Grains mostly in contango | **Met in the US** (20% backwardation). **Missed in China** (54%). |
| Energy switches regime | **Met.** Backwardation share by year, US / China: 2016 10% / 30%; 2020 22% / 26%; 2021 75% / 54%; 2022 86% / 82%. |
| China ferrous mostly backwardated | **Met:** 74%. Iron ore is 99%. |

**Most notable finding:** China's curves spend far more time in backwardation than the same sectors in the US. Base metals: 42% vs 11%. Grains: 54% vs 20%.

**Caveat:** China slopes use closing prices, which may be stale for the less-traded next contract. US slopes use synchronous settlements.

## Study 2: Twin curves

Charts: `results/twin_curves.png`, `results/gold_carry.png`. Table: `results/twin_curves.csv`.

| Commodity | Months | Correlation | Same shape | China in backwardation | US in backwardation |
|---|---|---|---|---|---|
| Copper | 128 | +0.10 | 63% | 42% | 10% |
| Gold | 128 | −0.37 | 99% | 1% | 0% |
| Silver | 128 | −0.41 | 95% | 5% | 0% |
| Crude oil | 100 | +0.53 | 74% | 57% | 79% |
| Soybeans | 39 | −0.30 | 56% | 31% | 18% |
| Soymeal | 90 | +0.23 | 58% | 71% | 47% |
| Soyoil | 91 | +0.44 | 62% | 73% | 47% |
| Corn | 90 | +0.20 | 71% | 32% | 21% |
| Hogs | 64 | +0.23 | 59% | 38% | 28% |

**Expectations missed.** I expected metals to correlate most and hogs least.
- Metals correlate *least*: gold and silver are negative.
- Crude correlates most.
- Hogs are mid-pack.

**Why gold and silver are negative.** Gold's curves almost always share a shape (contango, 99% of months), but their *steepness* is set by local interest rates. From 2022, US rates rose from near zero to about 5% while China's drifted down from about 3% to 1.5%, so the two contangos moved in opposite directions.

**H2, gold cost of carry: FAIL.**

| | Coefficient | R² | Within bar? |
|---|---|---|---|
| US (127 months) | 0.91 | 0.91 | yes |
| China (196 months, 2010-04 → 2026-07) | 0.12 | 0.01 | no |
| US–China contango gap vs rate gap | correlation 0.86 | | yes |

The bar needs every condition, so H2 fails on China.

**Post hoc, not part of the verdict.** On China's months from 2016-03 on, the coefficient is 0.86 and R² 0.28. The fit fails mostly in 2010–15, when Shanghai gold's contango swung far more than the interbank rate.

**Silver (no bar):**
- US: coefficient 0.83, R² 0.87.
- China: coefficient 0.81, R² 0.09.
- Gap correlation: 0.80.

## Study 3: Inventories → curve, the theory of storage

Chart: `results/storage_curve.png`. Tables: `results/storage_regressions.csv`, `results/storage_china_markets.csv`.

**H3: PASS in both countries.**

**China:**
- 27 of the 50 markets with warehouse receipts qualified.
- b < 0 in 25 of 27 (93%); one-sided sign test p = 2.8×10⁻⁶.
- Exceptions: pulp (+0.011) and rapeseed meal (+0.0001).
- Strongest effects (change in annualized slope per 1 standard deviation more stock): soda ash −22 points, nickel −8.9, crude oil −6.2, tin −5.7, rebar −5.0.

**US:** b < 0 with t ≤ −2 in 4 of 4. CL clears with almost no margin.

| Market | b | t (Newey–West) | R² |
|---|---|---|---|
| Natural gas | −0.333 | −2.94 | 0.25 |
| Heating oil | −0.132 | −6.09 | 0.29 |
| Gasoline | −0.116 | −4.42 | 0.19 |
| WTI (Cushing stocks) | −0.091 | −2.02 | 0.10 |

**The storage curve** (deciles of inventory vs mean slope, measured against seasonal norms):

| | Scarcest stocks | Fullest stocks |
|---|---|---|
| China | +7.1% | −2.9% |
| US | +42% | −18% |

- The China curve falls steeply through the low-inventory deciles, then flattens across the top three deciles (−3.9%, −3.7%, −2.9%). That is the shape the theory predicts: contango is capped by the full cost of carry, backwardation is not.
- The US curve falls throughout. With only 4 markets, its shape is less informative.

**Caveat:** China's receipts count only exchange-registered stock, not all inventory.

## Study 4: Curve and volatility

Chart: `results/curve_volatility.png`. Table: `results/curve_volatility.csv`.

**H4: PASS in both countries.** Next-month volatility in the most-backwardated third of days was higher than in the middle third:

| | Markets where true | Sign test p | Median effect |
|---|---|---|---|
| China | 53 of 63 (84%) | 1.7×10⁻⁸ | +16% |
| US | 17 of 18 (94%) | 7.2×10⁻⁵ | +20% |

**Contango vs middle (no bar):**
- China: the median is −1%, and only 31 of 63 markets are higher. No V shape.
- US: the median is +9%, and 10 of 18 are higher, led by energy and precious metals (WTI +51%, platinum +40%, silver +38%). A partial V, as in Kogan, Livdan & Yaron (2009).

## Study 5: Case studies (descriptive)

Chart: `results/april_2020.png`.

**WTI (CME settlements):**
- May contract: 18.27 on 17 April, −37.63 on 20 April, 10.01 on 21 April.
- The contract a year out stayed at $34–35 through 17–20 April.

**Shanghai INE crude (closing prices, yuan per barrel):**
- The front contract was 246.1 on 17 April, 236.6 on 20 April and 224.0 on 21 April.
- Ordinary contango, never negative, with the 12th contract about 56–58 yuan above the front.

**8 March 2022:** WTI April 2022 was at 123.70, $43 above March 2024.

## Audit (2026-10-14): does backwardation depend on which contract the slope uses?

The Study 1 caveat above, tested. The atlas slope compares the held contract with the nearest later
contract that has any open interest. In Chinese markets that contract is often thin: its median open
interest is 5% of the dominant contract's for iron ore and 0.1–0.6% for coke, coking coal, rebar and
hot-rolled coil. So the slope was rebuilt from every listed contract in three more ways (dominant vs
nearest later, vs most-held later, vs nearest later with ≥10% of the dominant OI), plus the share of
downward steps along the whole liquid curve. Module: `contangolab/studies/robustness.py`
(`python -m contangolab study robustness`), table in `results/slope_robustness.csv`.

Share of liquid days in backwardation:

| Market | Window | Atlas | Next | Most-held | ≥10% OI | Whole curve, steps down |
|---|---|---|---|---|---|---|
| Iron ore (DCE) | 2019-04 – 2026-10 | 98.6% | 98.7% | 99.7% | 98.9% | 97.6% |
| Coke (DCE) | 2019-04 – 2026-10 | 79.8% | 79.8% | 85.8% | 80.2% | 69.1% |
| Coking coal (DCE) | 2019-09 – 2026-10 | 58.0% | 57.8% | 64.2% | 55.3% | 53.1% |
| Rebar (SHFE) | 2010-04 – 2026-10 | 64.7% | 64.6% | 51.3% | 61.2% | 57.5% |
| Hot-rolled coil (SHFE) | 2014-06 – 2026-10 | 63.9% | 63.8% | 74.3% | 76.8% | 70.6% |
| WTI crude (CME) | 2016-02 – 2026-10 | 62.8% | 59.7% | 64.5% | 59.7% | 70.3% |
| Natural gas (CME) | 2016-02 – 2026-10 | 27.2% | 27.1% | 29.3% | 27.1% | 36.9% |
| Copper (CME) | 2016-02 – 2026-10 | 10.5% | 10.6% | 10.9% | 10.8% | 13.4% |

**Findings:**
- **Iron ore's 99% is robust.** Every contract choice gives 98–100%, and 97.6% of steps along the
  liquid curve slope down. The median slope is +15% a year. It is a property of the curve, not of a
  thin contract. The window is 2019–26, because the Dalian data here starts in January 2019 (iron ore
  listed in 2013), so the figure says nothing about 2013–18.
- **Rebar and hot-rolled coil are sensitive.** Rebar ranges from 51% to 65% and HRC from 64% to 77%
  depending on the contract compared. Their next contract holds about 0.1–0.2% of the dominant OI,
  so single-market rebar and HRC shares should be quoted as a range. All five ferrous markets
  tested stay above 50% under every measure, so "mostly backwardated" holds; the 74% sector
  average (nine markets) was not recomputed.
- **US markets are stable** (within about 5 points), as expected with synchronous settlements and
  liquid next contracts. WTI is the exception on the whole-curve measure (70% vs 63%), because its
  back months are backwardated more often than the front spread.
