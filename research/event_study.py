"""Event-study chart for E1: average cumulative abnormal return after each kind of event.

Descriptive only (the verdicts come from the calendar-time portfolios in core/events.py): the
classic picture of how much, and how quickly, prices keep moving after news becomes public.
"""

from pathlib import Path

import numpy as np

from research.report import BLUE, GRAY, INK, INK_2, ORANGE, SURFACE, _style


def car_chart(path: Path, panels: list[tuple[str, list[tuple[str, np.ndarray, str]]]], note: str) -> None:
    """``panels``: [(title, [(label, paths array events x days, color), ...]), ...]; plots the mean path in %."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2), sharey=True, facecolor=SURFACE)
    for ax, (title, lines) in zip(np.atleast_1d(axes), panels):
        ax.axhline(0, color=GRAY, linewidth=0.8)
        for label, paths, color in lines:
            mean = 100 * np.nanmean(paths, axis=0)
            days = np.arange(1, len(mean) + 1)
            ax.plot(days, mean, color=color, linewidth=2)
            ax.annotate(f"{label}\n{mean[-1]:+.1f}%  (n={len(paths):,})", (days[-1], mean[-1]), xytext=(-4, 0),
                        textcoords="offset points", ha="right", va="bottom" if mean[-1] >= 0 else "top",
                        fontsize=7.5, color=INK)
        _style(ax, title, "Avg. return vs universe, %" if ax is np.atleast_1d(axes)[0] else "")
        ax.set_xlabel("Trading days after the news is public", color=INK_2, fontsize=8)
    fig.text(0.01, 0.01, note, fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


__all__ = ["car_chart", "BLUE", "ORANGE", "GRAY"]
