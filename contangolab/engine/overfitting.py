"""How likely is a backtest result to be luck? Purged cross-validation splits, the Deflated Sharpe
Ratio and the Probability of Backtest Overfitting.

  * ``cpcv_splits``: Combinatorial Purged Cross-Validation (Lopez de Prado, "Advances in Financial
    Machine Learning", ch. 12). The timeline is cut into N groups and every choice of k of them is a
    test set, purged and embargoed at each boundary so a forward-return label never straddles a
    train/test split.
  * ``deflated_sharpe_ratio``: Bailey & Lopez de Prado (2014). Shrinks an observed Sharpe for how
    many configurations were searched (and how spread out their Sharpes were) and for non-normal
    returns. Reports the probability that the true Sharpe beats the best you would expect by luck.
  * ``probability_of_backtest_overfitting``: Bailey, Borwein, Lopez de Prado & Zhu (2014), CSCV.
    Across many train/test splits, how often does the in-sample best configuration land below the
    median out of sample? Near 50% means picking "the best" was picking noise.

``contangolab.engine.design`` runs both over its 32-configuration grid.
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

EULER_GAMMA = 0.5772156649015329  # the Euler-Mascheroni constant, used by the expected-max-SR formula


# --- pure combinatorics: group partitioning, splits, purge/embargo ---------------------
# No market data touches this section; it is unit-tested directly (tests/test_overfitting.py).

def make_groups(n_dates: int, n_groups: int) -> list[np.ndarray]:
    """Positions ``0..n_dates-1`` split into ``n_groups`` contiguous, near-equal blocks."""
    if n_groups < 2 or n_groups > n_dates:
        raise ValueError(f"n_groups must be in [2, {n_dates}], got {n_groups}")
    return np.array_split(np.arange(n_dates), n_groups)


@dataclass(frozen=True)
class CPCVSplit:
    """One combinatorial train/test partition."""

    test_group_ids: tuple[int, ...]
    train_positions: np.ndarray
    test_blocks: list[np.ndarray]   # one contiguous position-array per test group


def cpcv_splits(n_dates: int, n_groups: int, n_test_groups: int,
                embargo_periods: int = 1) -> list[CPCVSplit]:
    """Every C(n_groups, n_test_groups) way of choosing test groups, purged and embargoed.

    PURGE: the one position immediately before a test group's start is dropped from
    train — its forward-return LABEL would span from before the test group into it.
    EMBARGO: ``embargo_periods`` positions immediately after a test group's end are also
    dropped from train — serial correlation could otherwise leak test-period information
    back into a training block that starts right after it.
    """
    groups = make_groups(n_dates, n_groups)
    splits = []
    for test_ids in itertools.combinations(range(n_groups), n_test_groups):
        exclude: set[int] = set()
        test_blocks = []
        for g in test_ids:
            block = groups[g]
            test_blocks.append(block)
            exclude.update(block.tolist())
            start, end = int(block[0]), int(block[-1])
            if start - 1 >= 0:
                exclude.add(start - 1)                        # purge
            for e in range(1, embargo_periods + 1):
                if end + e < n_dates:
                    exclude.add(end + e)                       # embargo
        train_positions = np.array([i for i in range(n_dates) if i not in exclude])
        splits.append(CPCVSplit(test_group_ids=test_ids, train_positions=train_positions,
                                test_blocks=test_blocks))
    return splits


# --- Deflated Sharpe Ratio ---------------------------------------------------------------

def annualized_sharpe(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Plain annualized Sharpe (no risk-free) of a periodic return series — a DSR trial input."""
    r = returns.dropna()
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods_per_year))


def deflated_sharpe_ratio(observed_returns: pd.Series, trial_sharpes_annualized: list,
                          periods_per_year: int = 12) -> dict:
    """Bailey & Lopez de Prado (2014). See the module docstring for what this corrects for.

    ``observed_returns``: the strategy's own (monthly) net-return series to judge — e.g.
    the full-history or OOS series, NOT one CPCV path.
    ``trial_sharpes_annualized``: the Sharpes of the "attempts" this result should be
    judged against — the configurations that were actually searched before this one was
    picked (``contangolab.engine.design`` passes every configuration of the grid). Passing
    CPCV path Sharpes instead would treat resampling noise of ONE configuration as a search
    over many.
    """
    returns = observed_returns.dropna()
    n = len(returns)
    if n < 3:
        raise ValueError("need at least 3 return observations to estimate skew/kurtosis")
    mean, std = returns.mean(), returns.std(ddof=1)
    if std == 0:
        raise ValueError("zero-variance return series — Sharpe is undefined")
    sr_hat = mean / std                                    # per-period (monthly) Sharpe
    g3 = float(skew(returns))
    g4 = float(kurtosis(returns, fisher=False))            # NON-excess kurtosis; normal = 3

    trials = np.asarray([s for s in trial_sharpes_annualized
                        if s is not None and np.isfinite(s)]) / math.sqrt(periods_per_year)
    n_trials = len(trials)
    sr0 = 0.0
    if n_trials >= 2 and trials.var(ddof=1) > 0:
        # Expected maximum Sharpe ratio achievable by chance across n_trials attempts
        # whose Sharpes have variance Var[SR] (Bailey & Lopez de Prado, eq. 7-8).
        z1 = norm.ppf(1.0 - 1.0 / n_trials)
        z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        sr0 = math.sqrt(trials.var(ddof=1)) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)

    denom = math.sqrt(max(1e-12, 1 - g3 * sr_hat + (g4 - 1) / 4.0 * sr_hat ** 2))
    z = (sr_hat - sr0) * math.sqrt(n - 1) / denom
    dsr = float(norm.cdf(z))
    return {
        "sharpe_annualized": float(sr_hat * math.sqrt(periods_per_year)),
        "sharpe_monthly": float(sr_hat),
        "n_obs": n,
        "n_trials": n_trials,
        "sr0_annualized": float(sr0 * math.sqrt(periods_per_year)),
        "skew": g3,
        "kurtosis": g4,
        "z": float(z),
        "dsr": dsr,
    }


# --- Probability of Backtest Overfitting (CSCV) -----------------------------------------

def probability_of_backtest_overfitting(variant_returns: dict, n_groups: int = 8) -> dict:
    """Bailey, Borwein, Lopez de Prado & Zhu (2014) CSCV procedure.

    ``variant_returns``: ``{variant_name: monthly net-return series}`` for >= 2 candidate
    configurations sharing a common date range (the design grid). Splits history
    into ``n_groups`` blocks; for every way of using exactly half as train and the
    complementary half as test (C(n_groups, n_groups//2) combinations), finds the
    IN-SAMPLE-best variant and checks its OUT-OF-SAMPLE rank among all variants. PBO is
    the fraction of splits where that in-sample winner finished at or below the OOS
    median — i.e. where picking "the best" would have been picking noise.
    """
    if len(variant_returns) < 2:
        raise ValueError("PBO needs at least two candidate variants to rank against each other")
    common_index = None
    for series in variant_returns.values():
        common_index = series.index if common_index is None else common_index.intersection(series.index)
    matrix = pd.DataFrame({name: s.reindex(common_index) for name, s in variant_returns.items()}).dropna()
    names = list(matrix.columns)
    n_variants = len(names)
    n_dates = len(matrix)
    if n_dates < n_groups * 2:
        raise ValueError(f"only {n_dates} common dates for {n_groups} groups — too few to split")

    groups = make_groups(n_dates, n_groups)
    half = n_groups // 2
    logits = []
    for train_ids in itertools.combinations(range(n_groups), half):
        test_ids = [g for g in range(n_groups) if g not in train_ids]
        train_pos = np.concatenate([groups[g] for g in train_ids])
        test_pos = np.concatenate([groups[g] for g in test_ids])
        train_block, test_block = matrix.iloc[train_pos], matrix.iloc[test_pos]

        train_sharpe = train_block.mean() / train_block.std(ddof=1)
        test_sharpe = test_block.mean() / test_block.std(ddof=1)
        best = train_sharpe.idxmax()

        # Relative rank of the in-sample winner's OOS Sharpe, omega_c = rank / (N+1)
        # (Bailey et al. 2014, eq. 6) -- NOT rank/N: dividing by N+1 is what centers the
        # median rank exactly on 0.5 (logit 0), so pure noise averages to PBO ~= 50%.
        oos_rank = test_sharpe.rank(method="average")[best]   # 1..n_variants
        rank = oos_rank / (n_variants + 1)
        logits.append(math.log(rank / (1 - rank)))

    logits_arr = np.asarray(logits)
    return {
        "n_splits": len(logits_arr),
        "n_variants": n_variants,
        "variant_names": names,
        "logits": logits_arr.tolist(),
        "pbo": float((logits_arr <= 0).mean()),
    }
