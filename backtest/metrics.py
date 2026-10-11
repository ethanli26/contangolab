"""Performance metrics from a trade log and an equity curve.

Two families of metrics:

  * Trade-based (win rate, payoff ratio, expectancy, profit factor, # trades) come
    from the closed-trade P&Ls.
  * Equity-curve based (total return, CAGR, max drawdown) come from the daily
    equity series.

Every metric is reported overall and broken down by regime. Trade stats are grouped
by the regime recorded at each trade's entry. Equity-curve stats per regime are
computed from the daily equity returns on days carrying that regime label (so the
pieces describe behavior *while* in each regime; because returns compound, the
per-regime totals do not sum to the overall total).
"""

import logging

import pandas as pd

log = logging.getLogger(__name__)

TRADING_DAYS = 252


def _safe_div(numerator: float, denominator: float) -> float | None:
    """Divide, returning ``None`` when the denominator is zero."""
    return numerator / denominator if denominator else None


def trade_metrics(pnls: pd.Series) -> dict:
    """Trade-based metrics from a series of per-trade net P&Ls."""
    pnls = pd.Series(pnls, dtype=float)
    n = int(len(pnls))
    if n == 0:
        return {"num_trades": 0, "win_rate": None, "payoff_ratio": None,
                "expectancy": None, "profit_factor": None, "total_pnl": 0.0}

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0  # negative or 0

    return {
        "num_trades": n,
        "win_rate": len(wins) / n,
        "payoff_ratio": _safe_div(avg_win, abs(avg_loss)),
        "expectancy": float(pnls.mean()),  # average $ P&L per trade
        "profit_factor": _safe_div(wins.sum(), abs(losses.sum())),
        "total_pnl": float(pnls.sum()),
    }


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of an equity curve, as a positive fraction."""
    equity = pd.Series(equity, dtype=float)
    if equity.empty:
        return 0.0
    underwater = equity / equity.cummax() - 1.0
    return float(-underwater.min())


def equity_metrics(equity: pd.Series) -> dict:
    """Total return, CAGR, and max drawdown from an equity curve."""
    equity = pd.Series(equity, dtype=float).dropna()
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return {"total_return": None, "cagr": None, "max_drawdown": None}

    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else None

    return {"total_return": float(total_return), "cagr": cagr, "max_drawdown": max_drawdown(equity)}


def _returns_to_equity(daily_returns: pd.Series) -> pd.Series:
    """Compound a series of daily returns into a normalized equity curve."""
    return (1.0 + daily_returns).cumprod()


def regime_equity_metrics(equity: pd.Series, regime: pd.Series, label: str) -> dict:
    """Equity-curve metrics restricted to days carrying ``label``.

    Daily equity returns on matching days are compounded into a sub-curve; CAGR is
    annualized by the number of days actually spent in the regime.
    """
    daily_returns = pd.Series(equity, dtype=float).pct_change()
    mask = regime.reindex(daily_returns.index) == label
    selected = daily_returns[mask].dropna()
    if selected.empty:
        return {"total_return": None, "cagr": None, "max_drawdown": None, "days": 0}

    sub_equity = _returns_to_equity(selected)
    total_return = float(sub_equity.iloc[-1] - 1.0)
    days = int(len(selected))
    cagr = (1.0 + total_return) ** (TRADING_DAYS / days) - 1.0 if days > 0 else None
    return {"total_return": total_return, "cagr": cagr,
            "max_drawdown": max_drawdown(sub_equity), "days": days}


def compute_overall(trades: pd.DataFrame, equity: pd.Series) -> dict:
    """Merge trade-based and equity-curve metrics for the whole backtest."""
    pnls = trades["pnl"] if "pnl" in trades else pd.Series(dtype=float)
    avg_bars_held = (
        float(trades["bars_held"].mean()) if "bars_held" in trades and not trades.empty else None
    )
    return {**equity_metrics(equity), **trade_metrics(pnls), "avg_bars_held": avg_bars_held}


def compute_by_regime(trades: pd.DataFrame, equity: pd.Series, regime: pd.Series) -> dict[str, dict]:
    """Compute the full metric set for each regime label."""
    result: dict[str, dict] = {}
    for label in ("bull", "bear", "crash"):
        if "regime_at_entry" in trades and not trades.empty:
            pnls = trades.loc[trades["regime_at_entry"] == label, "pnl"]
        else:
            pnls = pd.Series(dtype=float)
        eq = regime_equity_metrics(equity, regime, label)
        result[label] = {**eq, **trade_metrics(pnls)}
    return result


def compute_by_strategy(trades: pd.DataFrame) -> dict[str, dict]:
    """Trade-based metrics grouped by the triggering strategy.

    Returns ``{strategy_name: trade_metrics}``; empty if no strategy column/trades.
    """
    if trades.empty or "strategy" not in trades:
        return {}
    return {name: trade_metrics(group["pnl"]) for name, group in trades.groupby("strategy")}


def worst_drawdowns(equity: pd.Series, n: int = 5) -> pd.DataFrame:
    """Return the ``n`` deepest peak-to-trough drawdown episodes.

    An episode runs from a new equity peak, through the underwater stretch, to the
    point where equity reclaims that peak (or the end of the series if it never
    does). Each row reports peak/trough/recovery dates, depth, and length in days.
    """
    equity = pd.Series(equity, dtype=float).dropna()
    if equity.empty:
        return pd.DataFrame(columns=["peak_date", "trough_date", "recovery_date", "drawdown", "length_days"])

    running_peak = equity.cummax()
    underwater = equity < running_peak

    episodes = []
    start = None
    for i in range(len(equity)):
        if underwater.iloc[i] and start is None:
            start = i - 1 if i > 0 else i  # episode begins at the prior peak
        is_last = i == len(equity) - 1
        if start is not None and (not underwater.iloc[i] or is_last):
            end = i  # equity reclaimed the peak here (or the series ended)
            segment = equity.iloc[start:end + 1]
            trough_idx = segment.idxmin()
            peak_value = equity.iloc[start]
            depth = 1.0 - segment.min() / peak_value
            recovered = (not underwater.iloc[i]) and not is_last
            episodes.append({
                "peak_date": equity.index[start],
                "trough_date": trough_idx,
                "recovery_date": equity.index[i] if recovered else None,
                "drawdown": float(depth),
                "length_days": int((equity.index[i] - equity.index[start]).days),
            })
            start = None

    drawdowns = pd.DataFrame(episodes)
    if drawdowns.empty:
        return drawdowns
    return drawdowns.sort_values("drawdown", ascending=False).head(n).reset_index(drop=True)
