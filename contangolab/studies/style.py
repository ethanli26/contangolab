"""One chart style for every figure in ``results/``: a recessive grid, thin marks, text in ink colors.

Colors have fixed jobs:
  * country: China = BLUE, US = ORANGE (categorical, never swapped)
  * curve shape: a diverging scale, backwardation red (scarce, "tight") through a gray midpoint to
    contango blue (surplus, "storage full")
"""

from pathlib import Path

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8d8b85"
RED = "#e34948"
COUNTRY_COLOR = {"cn": BLUE, "us": ORANGE}
# blue arm (contango) .. gray .. red arm (backwardation), equal steps per arm
DIVERGING = ["#104281", "#256abf", "#6da7ec", "#b7d3f6", "#f0efec", "#f6c3bd", "#ee8d86", "#e34948", "#a8211f"]


def mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                                                                          "DejaVu Sans"],
                         "axes.titlesize": 12, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    return plt


def curve_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("curve", DIVERGING)


def style(ax, title: str, ylabel: str = "", grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="semibold", pad=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.tick_params(which="both", colors=INK_2, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def headline(fig, title: str, subtitle: str = "") -> None:
    """Figure-level title block, left-aligned above the panels."""
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.012, 0.985 - 0.032 * 13 / max(fig.get_figheight(), 4), subtitle, ha="left", va="top",
                 fontsize=9.5, color=INK_2)


def footnote(fig, text: str) -> None:
    fig.text(0.012, 0.008, text, ha="left", va="bottom", fontsize=7.5, color=MUTED)


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=160)
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"wrote {path}")
