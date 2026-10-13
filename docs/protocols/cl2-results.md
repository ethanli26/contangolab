# CL2 results: trading the curve–inventory mismatch

Rules: [`cl2-mismatch.md`](cl2-mismatch.md), committed before the first run (69c5e56). Run on 2026-10-12. The four trials are logged in `results/trials.jsonl` (protocol `cl2-v1`; the log now holds 222 trials).

Chart: `results/curve_mismatch.png`. Table: `results/curve_mismatch.csv`.

**Verdict: FAIL in both countries.** Neither book survives the multiple-testing correction, and the inventory signal does not beat the no-inventory control in both halves.

| Book | Months | Net Sharpe | t | Halves | q (all 222 trials) |
|---|---|---|---|---|---|
| China MISMATCH (27 markets) | 82 (2020-01 →) | −0.43 | −1.13 | −0.05 / −0.98 | 0.89 |
| China REVERT, control | 166 (2013-01 →) | −0.14 | −0.52 | +0.24 / −0.68 | 0.74 |
| China REVERT, MISMATCH's months | 82 | −0.68 | −1.77 | −0.71 / −0.68 | — |
| US MISMATCH (CL, HO, RB, NG) | 73 (2020-10 →) | −0.28 | −0.69 | −1.44 / +0.76 | 0.79 |
| US REVERT, control | 106 (2018-01 →) | −0.40 | −1.18 | −0.29 / −0.56 | 0.90 |
| US REVERT, MISMATCH's months | 73 | −0.81 | −2.01 | −0.99 / −0.61 | — |

**Inventory adds value?** No, by the pre-registered test (better than the control in both halves):
- China: better in the first half (−0.05 vs −0.71), worse in the second (−0.98 vs −0.68).
- US: worse in the first half (−1.44 vs −0.99), better in the second (+0.76 vs −0.61).

Over the whole period the inventory version lost less than the control in both countries (−0.43 vs −0.68; −0.28 vs −0.81). That is a hint, not a result.

## Checks run before logging (none depends on the result)

- **Sign of the spread return.** On days when neither leg rolls, the daily spread return (`ret − farRet`) correlates **0.90** (China) and **0.99** (US) with the change in the curve slope. So a long spread does gain when the curve moves toward backwardation, as specified.
- **Look-ahead.** Replacing all slope data after a cutoff with noise leaves every signal before the cutoff unchanged.

## What it means

At a one-week horizon, a curve that is out of line with its seasonal norm or with inventories tends to **stay out of line, or move further**, rather than snap back. Two pieces of evidence:
- Both "bet on correction" books lose money.
- The pure curve-reversion control loses the most.

This fits the basis-momentum evidence (Boons & Prado 2019): spreads trend.

The trade is also tiny in money terms. A spread book with gross exposure 1 had annualized returns of a few tenths of a percent either way. That's because the front two contracts move almost together. A real desk would size spreads with leverage.

## Not done

- **The opposite bet** (mismatches persist, so trade with them) is **not** claimed. Flipping a sign after seeing the result would be exactly the search this protocol exists to prevent. If tested, it gets its own protocol and trial entries, and the four CL2 trials stay in the count.
- **Horizons and thresholds.** No other horizon, threshold or subset was run.
