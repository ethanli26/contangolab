"""Allocation across books and the vol dial (BLUEPRINT 11, P5, protocol ``b11-p5-v1``).

Pre-registered in the vault before any run:

  * Books are monthly excess-return series (here: the US long-short selector after carrying costs,
    and the China futures default). Months are aligned by calendar month.
  * Allocation each month uses only months strictly before it (>= ``MIN_JOINT`` months of joint
    history):
        risk_parity    weights proportional to 1 / trailing vol (the simple default)
        equal_capital  50/50, reference only
        max_sharpe     w ~ inverse(shrunk covariance) x trailing mean, clipped at 0; all <= 0 -> risk parity
  * Vol dial: the combination is scaled to a target vol using the ex-ante vol of this month's
    weights on the trailing ``VOL_WINDOW_M`` months of book returns, capped at ``LEVER_CAP``.
    Chosen target = the highest of ``TARGETS`` whose 95th-percentile 10-year max drawdown (block
    bootstrap of the levered OOS months) is at most ``DD_LIMIT``.

Book rebalancing between the two books is monthly and its cost is not modeled (small next to the
books' own trading, which is).
"""

import math

import numpy as np
import pandas as pd

from core.design import halves, stats

PROTOCOL = "b11-p5-v1"
METHODS = ("risk_parity", "equal_capital", "max_sharpe")
BORROW_BPS = 50.0
BORROW_STRESS_BPS = 150.0
FINANCING_BPS = 50.0
MIN_JOINT = 24
SHRINK = 0.5
VOL_WINDOW_M = 36
LEVER_CAP = 4.0
TARGETS = (0.10, 0.15, 0.20, 0.25)
DD_LIMIT = 0.35
BOOT_YEARS, BOOT_BLOCK, BOOT_PATHS, BOOT_Q, BOOT_SEED = 10, 12, 5000, 0.95, 0


def carry_costs(borrow_bps: float = BORROW_BPS, financing_bps: float = FINANCING_BPS):
    """Daily adjustment for an equity book: borrow on short exposure plus a spread on long exposure."""
    def adjust(daily: pd.DataFrame) -> pd.DataFrame:
        d = daily.copy()
        d["carry"] = (d["short"] * borrow_bps + d["long"] * financing_bps) / 1e4 / 252
        d["net"] = d["net"] - d["carry"]
        return d
    return adjust


def by_month(r: pd.Series) -> pd.Series:
    """Re-index a monthly series by calendar month-end, so books on different exchange calendars line up."""
    out = r.copy()
    out.index = out.index.to_period("M").to_timestamp("M")
    return out


def weights(hist: pd.DataFrame, method: str) -> np.ndarray:
    k = hist.shape[1]
    vol = hist.std(ddof=1).to_numpy()
    rp = (1.0 / vol) / (1.0 / vol).sum()
    if method == "equal_capital":
        return np.full(k, 1.0 / k)
    if method == "risk_parity":
        return rp
    if method == "max_sharpe":
        cov = hist.cov().to_numpy()
        shrunk = (1 - SHRINK) * cov + SHRINK * np.diag(np.diag(cov))
        w = np.clip(np.linalg.solve(shrunk, hist.mean().to_numpy()), 0.0, None)
        return w / w.sum() if w.sum() > 0 else rp
    raise ValueError(f"unknown allocation method {method!r}")


def allocate(books: pd.DataFrame, method: str, target: float | None = None,
             min_hist: int = MIN_JOINT) -> pd.DataFrame:
    """Monthly combined return (and weights, leverage) on the joint months after ``min_hist``."""
    joint = books.dropna()
    rows = []
    for k in range(min_hist, len(joint)):
        hist = joint.iloc[:k]
        w = weights(hist, method)
        lev = 1.0
        if target is not None:
            recent = hist.iloc[-VOL_WINDOW_M:]
            ex_ante = math.sqrt(12 * float(w @ recent.cov().to_numpy() @ w))
            lev = min(LEVER_CAP, target / ex_ante) if ex_ante > 0 else 1.0
        rows.append({"date": joint.index[k], "ret": lev * float(joint.iloc[k].to_numpy() @ w), "lev": lev,
                     **{f"w_{c}": wi for c, wi in zip(joint.columns, w)}})
    return pd.DataFrame(rows).set_index("date")


def bootstrap_max_dd(r: pd.Series, years: int = BOOT_YEARS, block: int = BOOT_BLOCK, paths: int = BOOT_PATHS,
                     q: float = BOOT_Q, seed: int = BOOT_SEED) -> float:
    """q-quantile of the max drawdown magnitude over ``years``, from a moving-block bootstrap of r."""
    arr = r.dropna().to_numpy()
    n = years * 12
    nb = math.ceil(n / block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(arr) - block + 1, size=(paths, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(paths, -1)[:, :n]
    eq = np.cumprod(1.0 + arr[idx], axis=1)
    eq = np.concatenate([np.ones((paths, 1)), eq], axis=1)
    dd = 1.0 - (eq / np.maximum.accumulate(eq, axis=1)).min(axis=1)
    return float(np.quantile(dd, q))


def _beats(a: tuple[dict, dict], b: tuple[dict, dict]) -> bool:
    return all((x["sharpe"] or -9) > (y["sharpe"] or -9) for x, y in zip(a, b))


def run(books: dict[str, pd.Series]) -> dict:
    """Apply the P5 protocol to {name: monthly excess-return series}."""
    books = pd.concat({name: by_month(r.dropna()) for name, r in books.items()}, axis=1)
    combos = {m: allocate(books, m) for m in METHODS}
    window = combos["risk_parity"].index
    h = {m: halves(c["ret"], window) for m, c in combos.items()}
    book_h = {b: halves(books[b], window) for b in books.columns}
    bar1 = all(_beats(h["risk_parity"], bh) for bh in book_h.values())
    bar2 = _beats(h["max_sharpe"], h["risk_parity"])
    adopted = "max_sharpe" if bar2 else "risk_parity"

    dial = {}
    for target in TARGETS:
        lev = allocate(books, adopted, target=target)
        r = lev["ret"]
        yearly = (1 + r).groupby(r.index.year).prod() - 1
        dial[target] = {**stats(r), "dd95_10y": bootstrap_max_dd(r), "avg_lev": float(lev["lev"].mean()),
                        "max_lev": float(lev["lev"].max()), "worst_year": float(yearly.min()),
                        "worst_year_at": int(yearly.idxmin()), "series": r}
    ok = [t for t in TARGETS if dial[t]["dd95_10y"] <= DD_LIMIT]
    chosen = max(ok) if ok else None
    return {"books": books, "window": window, "corr": float(books.loc[window].corr().iloc[0, 1]),
            "book_stats": {b: stats(books[b].reindex(window)) for b in books.columns}, "book_halves": book_h,
            "combos": combos, "combo_stats": {m: stats(c["ret"]) for m, c in combos.items()}, "combo_halves": h,
            "bar1": bar1, "bar2": bar2, "adopted": adopted, "dial": dial, "chosen_target": chosen}
