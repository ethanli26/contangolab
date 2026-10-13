# CL2: trading the curve–inventory mismatch. Pre-registration, written before any run

Protocol tag `cl2-v1`, written 2026-10-12.

**What has been run so far:** only CL1, which measured *how* inventories and curve slopes move together. Nothing in this document has been run. In particular, no spread return has been computed.

## Question

The theory of storage ties the slope of the curve to inventories (CL1 H3 passed in both countries). When the curve is tighter or looser than inventories justify, does it correct? If so, can a calendar spread capture that, net of costs?

## Markets

The same markets as CL1 study 3:
- **China:** 27 markets with warehouse receipts (listed in `results/storage_china_markets.csv`, `qualifies = True`).
- **US:** 4 markets, each with an EIA series:

  | Market | Inventory series |
  |---|---|
  | CL | Cushing crude stocks |
  | HO | Distillate stocks |
  | RB | Gasoline stocks |
  | NG | Lower-48 working gas |

## The trade: a calendar spread

**Position.** Long the held (front) contract and short the next expiry, equal notional. Its daily return is `ret − farRet`:
- `ret` is the held contract's own return.
- `farRet` is the return of the next expiry chosen the day before.
- Both legs come from the continuous build, so a roll never creates a jump.

**What it bets on.** The spread earns money when the curve moves toward backwardation, and loses when it moves toward contango.

## Signals, on a weekly grid (the last trading day of each week, t)

### When an inventory number can be used

| Market | Usable from |
|---|---|
| US | the first grid date at least **7 days** after its as-of Friday. EIA publishes the following Wednesday (petroleum) or Thursday (gas); 7 days also covers holiday delays. |
| China | the first grid date **after** its as-of date. Receipts are published that evening. |

### Measures (per market)

- `slope(t)` is the curve slope `ry` at t.
- **Seasonal norms.** Each of `slope` and `log(1 + inventory)` gets its own norm: the mean over the same ISO week ±2 in **earlier years only**, needing at least 3 such years. This is the same norm as CL1.
- `slope_dev(t)` = slope minus its norm.
- `inv_dev(t)` = the latest usable inventory, minus its norm.

### The fair slope and the mismatch

- **Fair slope.** At each t, fit by OLS `slope_dev = a + b × inv_dev` on that market's grid weeks **strictly before t**. At least 104 such weeks are required.
- **Mismatch:** `m(t) = slope_dev(t) − (a + b × inv_dev(t))`.

### Signals

| Signal | Rule | What the position bets |
|---|---|---|
| **MISMATCH** (primary) | position sign = −sign(m(t)) | If the curve is more backwardated than inventories justify, short the front spread, betting it relaxes. If more contango than justified, go long. |
| **REVERT** (control, no inventory) | position sign = −sign(slope_dev(t)) | Same idea from the curve alone: an unusually steep curve returns to its seasonal norm. |

## Book

One book per country per signal, rebalanced at every grid date t:
- **Sizing:** each liquid market with a signal is weighted `sign / σ`, scaled so total gross = 1. `σ` is the standard deviation of that market's daily spread return over the previous 60 trading days.
- **Holding:** from the close of t to the close of the next grid date. The weights stay fixed during the week.
- **Costs:** 3 bps per side **per leg** (2 legs), charged on the weight change at each rebalance. This is the same per-side cost as every other test in this repo.
- **Monthly returns:** daily returns are compounded into calendar months for the statistics.

## Trials and bar

The signals are logged to `results/trials.jsonl` with `family` and `protocol = cl2-v1`.

| Trial | Family | Market |
|---|---|---|
| `cl2_mismatch` | `cl2_curve_mismatch` | `cn_futures` and `us_futures` |
| `cl2_revert` | `cl2_control` | `cn_futures` and `us_futures` |

**Survivor.** The same rule as every other trial:
- BH q ≤ 0.10, computed over the latest trial of every signal in the whole log;
- positive net return in both halves.

**Inventory adds value** (reported per country, not part of the survivor rule): MISMATCH net Sharpe > REVERT net Sharpe in **both** halves.

## Expectations, written now

**China:**
- Weak but possibly positive. Retail-heavy markets with spreads that move a lot can be slow to correct.
- MISMATCH should beat REVERT only if receipts carry information the curve hasn't priced.

**US:**
- Only 4 markets: low power. A pass would need a Sharpe of about 0.65 or more over roughly 8 years.
- Expected to fail. US energy spreads are heavily traded by exactly the desks that watch EIA stocks.

**Most likely overall outcome:** neither survives the correction.

## Not done

- No tuning. The signal is the sign only, and the holding period is fixed at one week.
- No other horizons, thresholds or market subsets will be tried after seeing results unless each one is logged as a new trial.
