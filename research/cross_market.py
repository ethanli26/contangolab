"""China vs US commodity futures: the same six signals, the same code, the same years.

Descriptive only (the verdicts come from the logged P1b and C1 trials). China's full sample starts
in 2012 and the US one in 2016 (CME open interest begins 2015-11), so China is cut to each US
signal's own months before comparing: the gap shown is between markets, not between periods.

    python -m research.cross_market      # prints the table, writes reports/cross_market.png
"""

import math
from pathlib import Path

import pandas as pd

from core.dataset import load_dataset
from core.evaluate import evaluate
from core.signals import get_signal
from research.report import BLUE, GRAY, INK, INK_2, ORANGE, SURFACE, _style

SHARED = {"ts1": "1-month trend", "ts12": "12-month trend", "rank63": "63-day rank\nmomentum",
          "donch20": "Donchian\nbreakout", "ma20_60": "20/60 moving\naverage", "carry": "Carry"}
OUT = Path(__file__).resolve().parent.parent / "reports" / "cross_market.png"


def _sharpe(net: pd.Series) -> float:
    return float(net.mean() / net.std(ddof=1) * math.sqrt(12))


def table() -> pd.DataFrame:
    cn, us = load_dataset("cn_futures"), load_dataset("us_futures")
    rows = []
    for key, label in SHARED.items():
        _, c = evaluate(get_signal(f"fut_{key}"), cn)
        _, u = evaluate(get_signal(f"usf_{key}"), us)
        c, u = c["net"].astype(float), u["net"].astype(float)
        same = c[(c.index >= u.index.min()) & (c.index <= u.index.max())]
        rows.append({"signal": label, "china_full": _sharpe(c), "china": _sharpe(same), "us": _sharpe(u),
                     "corr": same.corr(u.reindex(same.index)), "months": len(same),
                     "start": u.index.min().date()})
    return pd.DataFrame(rows)


def chart(t: pd.DataFrame, path: Path = OUT) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.4), facecolor=SURFACE)
    _style(ax, "Same signals, same code, same years: China vs US commodity futures",
           "Net Sharpe ratio, monthly rebalance")
    x, w = range(len(t)), 0.36
    for off, col, label, color in [(-w / 2, "china", "China commodity futures", BLUE),
                                   (w / 2, "us", "US commodity futures (CME Group)", ORANGE)]:
        bars = ax.bar([i + off for i in x], t[col], width=w - 0.02, color=color, label=label)
        for b, v in zip(bars, t[col]):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3 if v >= 0 else -10),
                        textcoords="offset points", ha="center", fontsize=7.5, color=INK)
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.set_xticks(list(x), t["signal"], fontsize=8, color=INK_2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    fig.text(0.01, 0.01, "Both markets over the US sample (2016-2026), after costs. Trend returns in the two "
             "markets correlate only 0.3-0.4.", fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    t = table()
    print(t.round(2).to_string(index=False))
    chart(t)
    print(f"wrote {OUT}")
