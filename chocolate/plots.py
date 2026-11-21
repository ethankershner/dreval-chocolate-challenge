"""Figures used by the notebook that are too long to keep inline."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE, GREY, GREEN, RED = "#3b6ea8", "#d9822b", "#8a8f98", "#4c9a6a", "#b8433a"
COLORS = {
    "PerDrugRidge": GREEN,
    "MultiTaskMLP": ORANGE,
    "AverageEnsemble": RED,
    "EnsembleMF": BLUE,
}
OURS = ["PerDrugRidge", "MultiTaskMLP", "AverageEnsemble"]


def style() -> None:
    """
    Apply one plain matplotlib style to every figure.

    Importing DrEval's report code switches matplotlib to the non-interactive "agg" backend and to the
    Helvetica Neue font, which is often not installed; both are undone here so figures show in the notebook.
    """
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None:
            shell.run_line_magic("matplotlib", "inline")
    except ImportError:
        pass
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "font.size": 10,
        }
    )


def response_matrix(table: pd.DataFrame, ax: plt.Axes) -> None:
    """
    Draw the cell lines x drugs matrix of ln(IC50), cell lines grouped by tissue, drugs ordered by mean response.

    :param table: long table from ``load_response_table``
    :param ax: axes to draw on
    """
    wide = table.dropna(subset=["response"]).pivot_table(index="cell_line_name", columns="pubchem_id", values="response")
    tissue = table.drop_duplicates("cell_line_name").set_index("cell_line_name").tissue.reindex(wide.index)
    wide = wide.loc[tissue.sort_values(kind="stable").index, wide.mean().sort_values().index]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e6e6e6")
    im = ax.imshow(np.ma.masked_invalid(wide.to_numpy()), aspect="auto", cmap=cmap, interpolation="nearest", vmin=-6, vmax=6)
    plt.colorbar(im, ax=ax, label="ln IC50 (µM)", fraction=0.03, pad=0.01)
    bounds = np.flatnonzero(tissue.loc[wide.index].to_numpy()[1:] != tissue.loc[wide.index].to_numpy()[:-1])
    for b in bounds:
        ax.axhline(b + 0.5, color="white", lw=0.4)
    ax.set_xlabel(f"{wide.shape[1]} drugs, ordered from most to least potent on average")
    ax.set_ylabel(f"{wide.shape[0]} cell lines, grouped by tissue")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("Response matrix (grey = no IC50 for this pair)")


def _box(ax, x, y, w, h, text, color, fontsize=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=color, ec="none", alpha=0.92))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white", fontsize=fontsize)


def _arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=12, color="#444444", lw=1.2))


def architecture(n_genes: int, n_drugs: int, hidden: int = 512, dropout: float = 0.2, n_ensemble: int = 10):
    """
    Draw the network: one shared trunk, one output unit per drug.

    :param n_genes: number of input genes
    :param n_drugs: number of output drugs
    :param hidden: hidden width
    :param dropout: hidden dropout
    :param n_ensemble: number of networks averaged
    :returns: the figure
    """
    fig, ax = plt.subplots(figsize=(11.6, 3.4))
    ax.set_xlim(-0.4, 11)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    y, h = 1.0, 1.6
    _box(ax, -0.3, y, 2.3, h, f"Expression\n{n_genes} landmark\ngenes, arcsinh,\nz-scored", GREY)
    _box(ax, 2.5, y, 2.3, h, f"Linear {n_genes}→{hidden}\nReLU\nDropout {dropout}", BLUE)
    _box(ax, 5.3, y, 2.3, h, f"Linear {hidden}→{hidden}\nReLU\nDropout {dropout}", BLUE)
    _box(ax, 8.1, y, 2.7, h, f"Linear {hidden}→{n_drugs}\none output per drug\n(standardized ln IC50)", ORANGE)
    for x0, x1 in [(2.0, 2.5), (4.8, 5.3), (7.6, 8.1)]:
        _arrow(ax, x0, x1, y + h / 2)
    ax.text(5.05, y + h + 0.35, "shared trunk: learned from every drug's measurements", ha="center", color=BLUE, fontsize=10)
    ax.plot([2.5, 7.6], [y + h + 0.18, y + h + 0.18], color=BLUE, lw=1.5)
    ax.text(9.45, y + h + 0.35, "drug-specific", ha="center", color=ORANGE, fontsize=10)
    ax.plot([8.1, 10.8], [y + h + 0.18, y + h + 0.18], color=ORANGE, lw=1.5)
    ax.text(5.3, 0.35, f"Loss: squared error over measured pairs only.  Early stopping on held-out cell lines.  "
            f"Prediction: average of {n_ensemble} networks.", ha="center", fontsize=9.5, color="#333333")
    return fig


def _arrow_xy(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color="#444444", lw=1.2))


def overview(n_genes: int, n_drugs: int):
    """
    Draw the two models and their average.

    :param n_genes: number of input genes
    :param n_drugs: number of drugs
    :returns: the figure
    """
    fig, ax = plt.subplots(figsize=(11.6, 3.4))
    ax.set_xlim(-0.4, 11.2)
    ax.set_ylim(0.2, 3.5)
    ax.axis("off")
    mid = 1.85
    _box(ax, -0.3, mid - 0.6, 2.0, 1.2, f"Expression\n{n_genes} landmark genes", GREY)
    rows = [
        (2.45, f"PerDrugRidge\n{n_drugs} linear models,\npenalty chosen per drug", GREEN),
        (0.3, f"MultiTaskMLP\none network,\n{n_drugs} outputs", ORANGE),
    ]
    for y, text, color in rows:
        _box(ax, 2.6, y, 3.0, 0.95, text, color, fontsize=8.8)
        _arrow_xy(ax, (1.7, mid), (2.6, y + 0.47))
        _arrow_xy(ax, (5.6, y + 0.47), (6.7, mid))
    _box(ax, 6.7, mid - 0.6, 1.9, 1.2, "AverageEnsemble\nmean of the two", RED, fontsize=9)
    _arrow_xy(ax, (8.6, mid), (9.1, mid))
    _box(ax, 9.1, mid - 0.6, 2.0, 1.2, f"ln IC50 for\n{n_drugs} drugs", GREY)
    ax.text(4.1, 3.45, "each trained on its own, on the same data", ha="center", fontsize=9.5, color="#333333")
    return fig


def leaderboard(means: pd.Series, colors: dict[str, str], ax: plt.Axes, top: int = 15) -> None:
    """
    Horizontal bar chart of the mean score of the best models.

    :param means: model -> mean score
    :param colors: model -> colour for highlighted models; the rest are grey
    :param ax: axes to draw on
    :param top: number of models to show
    """
    m = means.sort_values(ascending=False).head(top)[::-1]
    ax.barh(m.index, m.to_numpy(), color=[colors.get(a, GREY) for a in m.index])
    for i, v in enumerate(m.to_numpy()):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8.5)
    ax.set_xlabel("Pearson: normalized (mean of 10 splits)")
    ax.set_xlim(0, m.max() * 1.15)
    ax.grid(axis="y", visible=False)
