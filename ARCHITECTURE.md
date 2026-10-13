# How ContangoLab works

Data moves in one direction through four layers. Each layer only imports the layers before it.

```
markets/  ->  signals/  ->  engine/  ->  studies/
raw exchange    trading     shared tests,     one module per question,
records ->      signals     trial log,        writes results/
curves                      portfolios
```

## markets/: from exchange records to curves

**The problem.** A futures contract expires, so "the price of copper over ten years" is a chain
of contracts. `china.build_continuous` builds that chain with rules fixed before any test:

- **Which contract is followed.** The **main** contract is the one with the most open interest.
  The main never rolls back to an earlier expiry.
- **Which contract is held.** The contract held on day *t* is the main chosen at *t − 1*. A
  day's return is that contract's own close-to-close return, so it never jumps across a roll.
- **Back-adjustment.** Prices are adjusted backwards so that the series' returns equal the held
  contract's returns.
- **Slope.** The curve slope (`ry`) is the annualized log price ratio of the held contract to
  the next listed expiry with open interest.
  - Positive means backwardation.
  - The carry signal trades this same number, so the curve studies and the trading results
    describe one quantity.
- **Universe.** A market is liquid on day *t* if it has at least 60 days of history and its
  60-day traded notional is not in the bottom 20% that day.

`cme.py` turns Databento's CME records into the same input, so the US runs through the
identical code.

**CME data-quality decisions,** each covered by a test in `tests/test_cme.py`:
- **Close price.** The close is CME's **final settlement**. Before 2016–17 the feed did not flag
  settlements as final, so there the close is the last settlement published for that date.
- **Stale records.** A settlement that arrives after a newer trade date's settlement for the
  same contract is ignored. CME occasionally re-sends the next session's price under the
  previous date, which once put −$37.63 on the wrong day.
- **Open interest timing.** Open interest is used one day late, because CME publishes it the
  next morning.

**Other files in markets/:**
- `cftc.py` reads the Commitments of Traders reports. Each report is usable one week after its
  as-of date, and reports caught in a government shutdown wait until the CFTC caught up.
- `curves.py` exposes every listed contract per day, the slope panels, the sector map and the
  nine China–US twin pairs.
  - Tickers collide across the two countries: RB is rebar in Shanghai and gasoline in New York.
  - Anything that mixes them labels each market by country.
- `inventory.py` loads China's warehouse receipts, EIA weekly stocks and FRED interest rates.

## signals/: what is traded

`library.py` holds each signal as a function of the continuous panel:
- trend (TS1, TS12, Donchian, moving-average crossover);
- rank momentum;
- carry;
- basis-momentum;
- hedging pressure.

`registry.py` records, for each signal, *before* it is tested:
- its source;
- its sign (never flipped after seeing results);
- its family (the unit of multiple-testing correction);
- its publication year (for the before/after-publication split).

`tests/test_signals.py` deletes all data after a cutoff date and checks that no signal value
before the cutoff changes.

## engine/: one test for everything

- **`dataset.py`:** one loader per market. Each returns aligned date × market panels plus a
  liquidity mask.
- **`evaluate.py`:** the single monthly test.
  - Cross-sectional signals: long the top quintile, short the bottom quintile.
  - Time-series signals: each market long or short on its own, sized by inverse volatility.
  - Net of costs; reports first-half and second-half results.
- **`trials.py`:** an append-only log (`results/trials.jsonl`) of every evaluation ever run.
  It applies Benjamini–Hochberg correction across the whole log, so a lucky result is judged
  against everything else that was tried.
- **`combine.py`:** walk-forward signal combination (equal weight, IC-weighted, ridge, XGBoost).
  A month's score only uses months before it.
- **`portfolio.py`:** portfolio construction and a simulator with drifting holdings and
  turnover-based costs.
- **`design.py`:** the portfolio design-space search, judged by a selector that only sees the
  past.
- **`overfitting.py`:** the Deflated Sharpe Ratio, the Probability of Backtest Overfitting and
  purged cross-validation splits.

## studies/: one module per question

Each study has a `main()` that prints its tables and writes its charts to `results/`. Run one
with `python -m contangolab study <name>`, or all of them with `study all`.

| Study | Question | Pre-registered test |
|---|---|---|
| `atlas` | How often is each market in contango or backwardation? | descriptive |
| `twins` | Do the same commodity's curves agree across the two countries? Is gold's contango the interest rate? | H2 |
| `storage` | Do inventories set the curve (theory of storage)? | H3 |
| `volatility` | Does backwardation warn of volatility? | H4 |
| `cases` | What did whole curves look like in April 2020 and March 2022? | descriptive |
| `mismatch` | Does a curve that disagrees with inventories correct, and can a calendar spread earn it? Run with `--log` to append its trials. | CL2 |
| `cross_market` | Do the same trading signals work in both countries over the same years? | from the trial log |
| `reconcile_dolphindb` | Does the China pipeline match the DolphinDB build it was ported from? | exact match |

`style.py` fixes the chart colors:
- China is always blue and the US always orange;
- curve shape uses a diverging scale from contango (blue) through gray to backwardation (red).

## How a new question gets added

1. **Write the rules first.** Write the hypothesis, the measures and the pass bar in
   `docs/protocols/`, and commit that before running anything.
2. **Build it.** Add the data in `markets/` and the study in `studies/`, with tests for every
   new statistic.
3. **Report everything.** Run it and record the verdict and every missed expectation in the
   results document.
4. **Log trading tests.** If the study trades, log each evaluation as a trial so the
   multiple-testing correction counts it.
