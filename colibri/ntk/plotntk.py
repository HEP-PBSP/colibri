"""
colibri.ntk.plotntk.py

Plotting utilities for NTK eigenvalues and eigenvectors.

Design:
- Single `ntk_plot_provider` function handles all cases
- Draw styles: "band" (uncertainty bands) or "replicas" (individual lines)
- Iteration modes: "by_rank" or "by_fit"
"""

import warnings
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Iterator, List, Optional, Tuple

import matplotlib.patches as mpatches
import numpy as np
from matplotlib import rc

from validphys import plotutils

from colibri.constants import FLAVOURS_ID_MAPPINGS
from colibri.ntk.ntkutils import NTKGrid, NTKStats

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath,amssymb}")


HandlerSpec = namedtuple("HandlerSpec", ["color", "alpha"])


@dataclass
class PlotResult:
    """Result from plotting a single figure."""

    fig: Any
    ax: Any
    name: str
    title: str
    handles: list = field(default_factory=list)
    labels: list = field(default_factory=list)


class ComposedHandler:
    """Legend handler for plots with uncertainty bands."""

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height

        patch = mpatches.Rectangle(
            [x0, y0],
            width,
            height,
            facecolor=orig_handle.color,
            alpha=orig_handle.alpha,
            edgecolor="none",
            transform=handlebox.get_transform(),
        )
        line = mpatches.Rectangle(
            [x0, y0 + height / 2 - height * 0.05],
            width,
            height * 0.1,
            facecolor=orig_handle.color,
            alpha=1,
            edgecolor="none",
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(patch)
        handlebox.add_artist(line)
        return [patch, line]


# =============================================================================
# Drawing functions
# =============================================================================


def draw_band(
    ax,
    xgrid: np.ndarray,
    stats: NTKStats,
    label: str,
    error_type: str = "mean",
    handles=None,
    labels=None,
):
    """
    Draw data with uncertainty band.

    Returns array of plotted values for axis scaling.
    """
    color = ax._get_lines.get_next_color()
    alpha = 0.3

    if error_type == "median":
        ax.plot(xgrid, stats.median(), color=color, linewidth=1.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            err68down, err68up = stats.errorbar68()
        lower, upper = err68up, err68down
    elif error_type == "mean":
        ax.plot(xgrid, stats.central_value(), color=color, linewidth=1.5)
        lower, upper = stats.errorbarstd()
    else:
        raise ValueError(f"Unknown error_type '{error_type}'")

    ax.fill_between(xgrid, lower, upper, color=color, alpha=alpha, zorder=1)

    if handles is not None and labels is not None:
        handles.append(HandlerSpec(color=color, alpha=alpha))
        labels.append(label)

    return np.array([lower, upper])


def draw_replicas(ax, xgrid, stats, label, **kwargs):
    """
    Draw individual replica lines with mean overlay.

    Returns array of plotted values for axis scaling.
    """
    color = ax._get_lines.get_next_color()
    data = stats.data
    ax.plot(xgrid, data.T, alpha=0.2, linewidth=0.5, color=color, zorder=1)
    ax.plot(xgrid, stats.central_value(), color=color, linewidth=2, label=label)
    return data


# =============================================================================
# Iteration utilities
# =============================================================================


def iter_by_rank(
    grids: List[NTKGrid], rank_indices: List[int], extra_kwargs: Optional[dict] = None
):
    """
    Yield (rank_index, items) where items is list of (stats, label, xgrid) per grid.
    """
    extra_kwargs = extra_kwargs or {}
    for rank_index in rank_indices:
        items = []
        for grid in grids:
            stats = grid.get_plotting_data(rank_index, **extra_kwargs)
            items.append((stats, grid.label, grid.xgrid))
        yield rank_index, items


def iter_by_fit(
    grids: List[NTKGrid], rank_indices: List[int], extra_kwargs: Optional[dict] = None
):
    """
    Yield (grid, items) where items is list of (stats, label, xgrid) per rank.
    """
    extra_kwargs = extra_kwargs or {}
    for grid in grids:
        items = []
        for rank_index in rank_indices:
            stats = grid.get_plotting_data(rank_index, **extra_kwargs)
            label = grid.get_plotting_label(rank_index, **extra_kwargs)
            items.append((stats, label, grid.xgrid))
        yield grid, items


# =============================================================================
# Main plotting function
# =============================================================================


def ntk_plot_provider(
    grids: List[NTKGrid],
    rank_indices: Optional[List[int]] = None,
    iterator_fn=iter_by_rank,
    draw_fn=draw_band,
    custom_handler=ComposedHandler,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    title_fn=None,
    name_fn=None,
    ylabel_fn=None,
) -> Iterator[Tuple[Any, PlotResult]]:
    """
    Unified NTK plotting function for eigenvalues and eigenvectors.

    Parameters
    ----------
    grids : list of NTKGrid
        Data containers (EigenvalueGrid or EigenvectorGrid)
    rank_indices : list of int, optional
        Which ranks to plot. Default: first 5.
    iterator_fn : callable
        Function to iterate over grids and ranks (e.g., iter_by_rank or
        iter_by_fit)
    draw_fn : callable
        Function to draw data (e.g., draw_band or draw_replicas)
    custom_handler : callable, optional
        Custom legend handler class. If None, uses default legend.
    xscale, yscale : str
        Axis scales ("linear" or "log")
    ymin, ymax : float, optional
        Y-axis limits
    title_fn : callable, optional
        Custom title function: (rank_index, grid) -> str
    name_fn : callable, optional
        Custom name function: (rank_index, grid) -> str
    ylabel_fn : callable, optional
        Custom ylabel function: (rank_index, grid) -> str

    Yields
    ------
    tuple
        (fig, PlotResult) pairs
    """
    if not grids:
        return

    # Determine rank indices
    if rank_indices is None:
        max_ranks = min(grid.n_ranks for grid in grids)
        rank_indices = list(range(min(5, max_ranks)))

    # Get common xgrid
    common_xgrid = grids[0].xgrid
    xlabel = grids[0].xlabel

    iterator = iterator_fn(grids, rank_indices)
    for grid, items in iterator:
        fig, ax = plotutils.subplots(figsize=(8, 6))
        handles, labels_list = [], []
        all_vals = []

        title = title_fn(grid)
        name = name_fn(grid)
        ylabel = ylabel_fn(grid)
        ax.set_title(title)

        # Draw each item
        for stats, label, xgrid in items:
            vals = draw_fn(
                ax,
                xgrid,
                stats,
                label,
                handles=handles,
                labels=labels_list,
            )
            if vals is not None:
                all_vals.append(np.atleast_2d(vals))

        # Configure axes
        if xscale and xscale != "linear":
            ax.set_xscale(xscale)
        if yscale and yscale != "linear":
            ax.set_yscale(yscale)

        if all_vals:
            plotutils.frame_center(ax, common_xgrid, np.concatenate(all_vals))
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
        if ymax is not None:
            ax.set_ylim(top=ymax)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(common_xgrid[0], common_xgrid[-1])
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)

        # Legend
        if custom_handler is None:
            ax.legend()
        else:
            ax.legend(handles, labels_list, handler_map={HandlerSpec: custom_handler()})

        result = PlotResult(
            fig=fig, ax=ax, name=name, title=title, handles=handles, labels=labels_list
        )
        yield fig, result


# =============================================================================
# Convenience functions
# =============================================================================


def plot_eigvals_by_rank(
    eigval_grids_by_fit,
    rank_indices: Optional[list] = None,
    error_type: str = "mean",
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
):
    """Plot eigenvalues, one figure per rank showing all fits."""
    yield from ntk_plot_provider(
        eigval_grids_by_fit,
        rank_indices,
        draw_fn=partial(draw_band, error_type=error_type),
        iterator_fn=iter_by_rank,
        title_fn=lambda rank_index: rf"$\lambda^{{({rank_index + 1})}}$",
        name_fn=lambda rank_index: f"lambda_{rank_index + 1}",
        ylabel_fn=lambda rank_index: rf"$\lambda^{{({rank_index + 1})}}$",
        xscale=xscale,
        yscale=yscale,
        ymin=ymin,
        ymax=ymax,
    )


def plot_eigvals_by_fit(
    eigval_grids_by_fit,
    rank_indices: Optional[list] = None,
    error_type: str = "mean",
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
):
    """Plot eigenvalues, one figure per fit showing multiple ranks."""
    yield from ntk_plot_provider(
        eigval_grids_by_fit,
        rank_indices,
        draw_fn=partial(draw_band, error_type=error_type),
        iterator_fn=iter_by_fit,
        title_fn=lambda grid: grid.label,
        name_fn=lambda grid: f"eigvals_{grid.label}",
        ylabel_fn=lambda _: r"$\textrm{NTK eigenvalues}$",
        xscale=xscale,
        yscale=yscale,
        ymin=ymin,
        ymax=ymax,
    )


def plot_eigvals_replicas_by_rank(
    eigval_grids_by_fit,
    rank_indices: Optional[list] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
):
    """Plot eigenvalue replicas, one figure per rank."""
    yield from ntk_plot_provider(
        eigval_grids_by_fit,
        rank_indices,
        draw_fn=draw_replicas,
        iterator_fn=iter_by_rank,
        title_fn=lambda rank_index: rf"$\lambda^{{({rank_index + 1})}}$",
        name_fn=lambda rank_index: f"lambda_replicas_{rank_index + 1}",
        ylabel_fn=lambda rank_index: rf"$\lambda^{{({rank_index + 1})}}$",
        xscale=xscale,
        yscale=yscale,
        ymin=ymin,
        ymax=ymax,
        custom_handler=None,
    )


def plot_eigenvectors_by_rank_and_flavour(
    eigvecs_grids_by_fit,
    flavour_indices: list,
    error_type: str = "mean",
    rank_indices: Optional[list] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
):
    """
    Plot eigenvector components, one figure per (rank, flavour).
    """
    if rank_indices is None:
        max_ranks = min(grid.n_ranks for grid in eigvecs_grids_by_fit)
        rank_indices = list(range(min(5, max_ranks)))

    for flavour_index in flavour_indices:
        flavour_name = FLAVOURS_ID_MAPPINGS[flavour_index]
        yield from ntk_plot_provider(
            eigvecs_grids_by_fit,
            rank_indices=rank_indices,
            iterator_fn=partial(
                iter_by_rank, extra_kwargs={"flavour_index": flavour_index}
            ),
            draw_fn=partial(draw_band, error_type=error_type),
            title_fn=lambda _: f"{flavour_name}",
            name_fn=lambda rank_index: f"eigvec_{rank_index + 1}_{flavour_name}",
            ylabel_fn=lambda _: f"{flavour_name}",
            xscale=xscale,
            yscale=yscale,
            ymin=ymin,
            ymax=ymax,
        )


def plot_eigenvectors_by_fit_and_flavour(
    eigvecs_grids_by_fit,
    flavour_indices: list,
    error_type: str = "mean",
    rank_indices: Optional[list] = None,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
):
    """
    Plot eigenvector components, one figure per (fit, flavour).
    """
    if rank_indices is None:
        max_ranks = min(grid.n_ranks for grid in eigvecs_grids_by_fit)
        rank_indices = list(range(min(5, max_ranks)))

    for flavour_index in flavour_indices:
        flavour_name = FLAVOURS_ID_MAPPINGS[flavour_index]
        yield from ntk_plot_provider(
            eigvecs_grids_by_fit,
            rank_indices=rank_indices,
            iterator_fn=partial(
                iter_by_fit, extra_kwargs={"flavour_index": flavour_index}
            ),
            draw_fn=partial(draw_band, error_type=error_type),
            title_fn=lambda grid: f"{grid.label} - {flavour_name}",
            name_fn=lambda _: f"eigvecs_{flavour_name}",
            ylabel_fn=lambda _: f"{flavour_name}",
            xscale=xscale,
            yscale=yscale,
            ymin=ymin,
            ymax=ymax,
        )
