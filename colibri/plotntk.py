"""
colibri.plotntk.py

Module providing plotting utilities for NTK eigenvalues.

The plotting framework follows the validphys pdfplots.py design pattern, supporting:
1. Plotting eigenvalues of different ranks for the same fit
2. Plotting eigenvalues of different fits with the same rank
"""

import abc
import warnings
from collections import namedtuple
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import matplotlib.patches as mpatches
from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath,amssymb}")

from colibri.ntk import EigenvalueGrid
from validphys import plotutils


HandlerSpec = namedtuple("HandlerSpec", ["color", "alpha", "hatch", "outer"])


class ComposedHandler:
    """
    Legend handler for eigenvalue plots with uncertainty bands.
    """

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height

        patches = []

        if orig_handle.outer:
            wpad = width * 0.1
            hpad = height * 0.1
            outer = mpatches.Rectangle(
                [x0, y0],
                width,
                height,
                facecolor="none",
                linestyle="dashed",
                edgecolor=orig_handle.color,
                transform=handlebox.get_transform(),
            )
            handlebox.add_artist(outer)
            patches.append(outer)
        else:
            wpad = hpad = 0

        patch = mpatches.Rectangle(
            [x0 + wpad, y0 + hpad],
            width - 2 * wpad,
            height - 2 * hpad,
            facecolor=orig_handle.color,
            alpha=orig_handle.alpha,
            hatch=orig_handle.hatch,
            edgecolor="none",
            transform=handlebox.get_transform(),
        )

        handlebox.add_artist(patch)
        patches.append(patch)
        return patches


class PlotState(SimpleNamespace):
    """
    State object for eigenvalue plotting.

    This class encapsulates the iteration strategy, allowing the same plotter
    to handle both "by-rank" (one figure per rank, multiple fits) and "by-fit"
    (one figure per fit, multiple ranks) modes.

    The state holds:
    - fig, ax: matplotlib figure and axes
    - name: identifier for the figure (used in filenames)
    - title: plot title
    - ylabel: y-axis label
    - handles, labels: accumulated legend entries
    - hatchit: iterator for hatch patterns

    Use the class methods `by_rank()` and `by_fit()` to create appropriately
    configured states.
    """

    @classmethod
    def by_rank(
        cls,
        rank_index: int,
        eigval_grids: List[EigenvalueGrid],
        epochs: np.ndarray,
    ):
        """
        Create state for plotting multiple fits at a fixed rank.

        Parameters
        ----------
        rank_index : int
            The eigenvalue rank to plot
        eigval_grids : list of EigenvalueGrid
            Grids to draw (one per fit)
        epochs : ndarray
            Common epochs for x-axis

        Returns
        -------
        PlotState
            Configured for by-rank iteration
        """
        fig, ax = plotutils.subplots(figsize=(8, 6))
        return cls(
            fig=fig,
            ax=ax,
            mode="by_rank",
            rank_index=rank_index,
            _grids=eigval_grids,
            _epochs=epochs,
            name=f"lambda_{rank_index + 1}",
            title=rf"Eigenvalue $\lambda^{{({rank_index + 1})}}$",
            ylabel=rf"$\lambda^{{({rank_index + 1})}}$",
            handles=[],
            labels=[],
            hatchit=plotutils.hatch_iter(),
        )

    @classmethod
    def by_fit(
        cls,
        grid: EigenvalueGrid,
        rank_indices: List[int],
    ):
        """
        Create state for plotting multiple ranks for a fixed fit.

        Parameters
        ----------
        grid : EigenvalueGrid
            The fit to plot
        rank_indices : list of int
            Ranks to draw

        Returns
        -------
        PlotState
            Configured for by-fit iteration
        """
        fig, ax = plotutils.subplots(figsize=(8, 6))
        return cls(
            fig=fig,
            ax=ax,
            mode="by_fit",
            grid=grid,
            _rank_indices=rank_indices,
            _epochs=np.array(grid.epochs),
            name=f"fit_{grid.label}",
            title=rf"Eigenvalues - {grid.label}",
            ylabel=r"Eigenvalue",
            handles=[],
            labels=[],
            hatchit=plotutils.hatch_iter(),
        )

    def iter_items(self):
        """
        Yield (trajectory, label, epochs) tuples for each item to draw.

        In by-rank mode: iterates over fits, yields trajectory at fixed rank
        In by-fit mode: iterates over ranks, yields trajectory at fixed grid

        Yields
        ------
        tuple
            (MCStats trajectory, str label, ndarray epochs)
        """
        if self.mode == "by_rank":
            for grid in self._grids:
                traj = grid.get_eigenvalue_trajectory(self.rank_index)
                yield traj, grid.label, np.array(grid.epochs)
        else:  # by_fit
            for rank_index in self._rank_indices:
                traj = self.grid.get_eigenvalue_trajectory(rank_index)
                label = rf"$\lambda^{{({rank_index + 1})}}$"
                yield traj, label, self._epochs


class EigvalPlotter(abc.ABC):
    """
    Abstract base class for eigenvalue plotting.

    This follows the validphys PDFPlotter pattern. Subclasses implement the
    `draw()` method to render eigenvalues with different styles (lines, bands, etc.).

    The plotter supports two iteration modes via PlotState:
    - by_rank: One figure per rank, multiple fits per figure (default)
    - by_fit: One figure per fit, multiple ranks per figure

    Parameters
    ----------
    eigval_grids : list of EigenvalueGrid
        List of eigenvalue data containers, one per fit to compare
    rank_indices : list of int, optional
        Which eigenvalue ranks to plot. If None, plots first 5.
    xscale : str, optional
        X-axis scale ("linear", "log"). Default is "linear".
    yscale : str, optional
        Y-axis scale ("linear", "log"). Default is "log".
    ymin : float, optional
        Minimum y-axis value
    ymax : float, optional
        Maximum y-axis value

    Examples
    --------
    >>> grids = [eigval_grid_fit1, eigval_grid_fit2]
    >>> plotter = BandEigvalPlotter(grids, rank_indices=[0, 1, 2])
    >>> for fig, state in plotter:
    ...     fig.savefig(f"eigenvalue_{state.name}.pdf")
    """

    def __init__(
        self,
        eigval_grids: List[EigenvalueGrid],
        rank_indices: Optional[List[int]] = None,
        xscale: Optional[str] = None,
        yscale: Optional[str] = "log",
        ymin: Optional[float] = None,
        ymax: Optional[float] = None,
    ):
        self.eigval_grids = eigval_grids
        self._xscale = xscale or "linear"
        self._yscale = yscale or "log"
        self.ymin = ymin
        self.ymax = ymax

        # Determine rank indices to plot
        if rank_indices is None:
            max_ranks = min(grid.n_eigenvalues for grid in eigval_grids)
            self._rank_indices = list(range(min(5, max_ranks)))
        else:
            self._rank_indices = rank_indices

        # Get common epochs across all grids
        self._epochs = self._get_common_epochs()

    def _get_common_epochs(self) -> List[int]:
        """Get epochs that are common across all grids."""
        if not self.eigval_grids:
            return []
        common = set(self.eigval_grids[0].epochs)
        for grid in self.eigval_grids[1:]:
            common &= set(grid.epochs)
        return sorted(common)

    @property
    def epochs(self) -> np.ndarray:
        """Epochs array for plotting."""
        return np.array(self._epochs)

    @property
    def xscale(self) -> str:
        return self._xscale

    @property
    def yscale(self) -> str:
        return self._yscale

    @property
    def firstgrid(self) -> EigenvalueGrid:
        """Reference grid for determining dimensions."""
        if self.eigval_grids:
            return self.eigval_grids[0]
        raise AttributeError("Need at least one EigenvalueGrid")

    def iter_states(self):
        """
        Yield PlotState objects for each figure.

        Override this method to change iteration strategy.

        Yields
        ------
        PlotState
            Configured state for each figure
        """
        pass

    @abc.abstractmethod
    def draw(self, trajectory, label, epochs, state: PlotState) -> Optional[np.ndarray]:
        """
        Draw one eigenvalue trajectory on the current axes.

        Parameters
        ----------
        trajectory : MCStats
            Eigenvalue trajectory data
        label : str
            Legend label for this trajectory
        epochs : ndarray
            Epochs for x-axis
        state : PlotState
            Current plotting state

        Returns
        -------
        np.ndarray or None
            Array of values used for autoscaling y-limits, or None
        """
        pass

    def legend(self, state: PlotState):
        """
        Add legend to the axis.

        Parameters
        ----------
        state : PlotState
            State object containing handles and labels
        """
        return state.ax.legend()

    def __iter__(self):
        yield from self()

    def __call__(self):
        """
        Iterate over figures, yielding (fig, state) pairs.

        Yields
        ------
        tuple
            (matplotlib.figure.Figure, PlotState) pairs
        """
        if not self.eigval_grids:
            return

        for state in self.iter_states():
            ax = state.ax
            ax.set_title(state.title)

            all_vals = []
            for trajectory, label, epochs in state.iter_items():
                limits = self.draw(trajectory, label, epochs, state)
                if limits is not None:
                    all_vals.append(np.atleast_2d(limits))

            # Set axis properties
            if self._xscale and self._xscale != "linear":
                ax.set_xscale(self._xscale)
            if self._yscale and self._yscale != "linear":
                ax.set_yscale(self._yscale)

            if self.ymin is not None:
                ax.set_ylim(bottom=self.ymin)
            if self.ymax is not None:
                ax.set_ylim(top=self.ymax)

            ax.set_xlabel(r"$\rm Epoch$")
            ax.set_ylabel(state.ylabel)
            ax.set_xlim(self.epochs[0], self.epochs[-1])

            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.3)

            self.legend(state)
            yield state.fig, state


class BandEigvalPlotter(EigvalPlotter):
    """
    Plot eigenvalues with uncertainty bands.

    Shows central value as a line with 68% confidence interval as a shaded band.
    Different items are distinguished by color and hatch patterns.

    Parameters
    ----------
    eigval_grids : list of EigenvalueGrid
        List of eigenvalue data containers
    show_mc_errors : bool, optional
        If True, show additional 1-sigma dashed lines for MC stats. Default True.
    legend_stat_labels : bool, optional
        If True, include statistical info in legend labels. Default True.
    **kwargs
        Additional arguments passed to EigvalPlotter

    Examples
    --------
    >>> plotter = BandEigvalPlotter(
    ...     [grid_L0, grid_L1, grid_L2],
    ...     rank_indices=[0, 1, 2, 3, 4],
    ...     show_mc_errors=True,
    ... )
    >>> for fig, state in plotter:
    ...     fig.savefig(f"ntk_eigvals_{state.name}.pdf")
    """

    def __init__(
        self,
        eigval_grids: List[EigenvalueGrid],
        show_mc_errors: bool = True,
        legend_stat_labels: bool = True,
        **kwargs,
    ):
        self.show_mc_errors = show_mc_errors
        self.legend_stat_labels = legend_stat_labels
        super().__init__(eigval_grids, **kwargs)

    def legend(self, state: PlotState):
        """Add legend with custom handlers for band visualization."""
        return state.ax.legend(
            state.handles, state.labels, handler_map={HandlerSpec: ComposedHandler()}
        )

    def draw(self, trajectory, label, epochs, state: PlotState) -> Optional[np.ndarray]:
        """
        Draw eigenvalue trajectory with uncertainty band.

        Parameters
        ----------
        trajectory : MCStats
            Eigenvalue trajectory data
        label : str
            Legend label
        epochs : ndarray
            Epochs for x-axis
        state : PlotState
            Current plotting state

        Returns
        -------
        np.ndarray
            Array of [lower_bound, upper_bound] for autoscaling
        """
        ax = state.ax
        hatchit = state.hatchit
        handles = state.handles
        labels = state.labels

        # Compute statistics
        cv = trajectory.central_value()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            err68down, err68up = trajectory.errorbar68()

        # Plot styling
        hatch = next(hatchit)
        color = ax._get_lines.get_next_color()

        # Plot central value line
        ax.plot(epochs, cv, color=color, linewidth=1.5)

        # Plot uncertainty band
        alpha = 0.5
        ax.fill_between(epochs, err68up, err68down, color=color, alpha=alpha, zorder=1)

        outer = False
        if self.show_mc_errors:
            errorstdup, errorstddown = trajectory.errorbarstd()
            ax.plot(epochs, errorstdup, linestyle="--", color=color, alpha=0.7)
            ax.plot(epochs, errorstddown, linestyle="--", color=color, alpha=0.7)
            outer = True

        # Create legend entry
        handle = HandlerSpec(color=color, alpha=alpha, hatch=hatch, outer=outer)
        handles.append(handle)
        labels.append(label)

        return np.array([err68down, err68up])


class ByRankBandPlotter(BandEigvalPlotter):
    """
    Plot eigenvalues with bands, one figure per rank showing multiple fits.

    This is the default view:
    - BandEigvalPlotter: one figure per rank, multiple fits
    - ByFitBandPlotter: one figure per fit, multiple ranks

    Parameters
    ----------
    eigval_grids : list of EigenvalueGrid
        List of eigenvalue data containers
    rank_indices : list of int, optional
        Which ranks to plot on each figure
    **kwargs
        Additional arguments passed to BandEigvalPlotter

    Examples
    --------
    >>> plotter = ByRankBandPlotter(
    ...     [grid_L0, grid_L1, grid_L2],
    ...     rank_indices=[0, 1, 2, 3, 4],
    ... )
    >>> for fig, state in plotter:
    ...     fig.savefig(f"ntk_eigvals_{state.name}.pdf")
    """

    def iter_states(self):
        """Yield PlotState for each rank."""
        for rank_index in self._rank_indices:
            yield PlotState.by_rank(rank_index, self.eigval_grids, self.epochs)


class ByFitBandPlotter(BandEigvalPlotter):
    """
    Plot eigenvalues with bands, one figure per fit showing multiple ranks.

    This is the "transposed" view compared to BandEigvalPlotter:
    - BandEigvalPlotter: one figure per rank, multiple fits
    - ByFitBandPlotter: one figure per fit, multiple ranks

    Parameters
    ----------
    eigval_grids : list of EigenvalueGrid
        List of eigenvalue data containers
    rank_indices : list of int, optional
        Which ranks to plot on each figure
    **kwargs
        Additional arguments passed to BandEigvalPlotter

    Examples
    --------
    >>> plotter = ByFitBandPlotter(
    ...     [grid_L0, grid_L1],
    ...     rank_indices=[0, 1, 2, 3, 4],
    ... )
    >>> for fig, state in plotter:
    ...     fig.savefig(f"ntk_eigvals_{state.name}.pdf")
    """

    def iter_states(self):
        """Yield PlotState for each fit."""
        for grid in self.eigval_grids:
            yield PlotState.by_fit(grid, self._rank_indices)


# =============================================================================
# Provider Functions for Reportengine Integration
# =============================================================================


def plot_eigvals_by_rank(
    eigval_grids_by_fit,
    rank_indices=None,
    xscale=None,
    yscale="log",
    ymin=None,
    ymax=None,
    show_mc_errors: bool = True,
    legend_stat_labels: bool = True,
):
    """
    Plot eigenvalue evolution across epochs, one figure per rank.

    Each figure shows how a specific eigenvalue (e.g., λ¹, λ², etc.) evolves
    across epochs for all provided fits.

    Use case: Comparing the same eigenvalue rank across different fits
    (e.g., L0, L1, L2 architectures).

    Parameters
    ----------
    eigval_grids_by_fit : list of EigenvalueGrid
        List of eigenvalue data containers, one per fit
    rank_indices : list of int, optional
        Which eigenvalue ranks to plot. Default: first 5
    xscale : str, optional
        X-axis scale ("linear", "log"). Default: "linear"
    yscale : str, optional
        Y-axis scale ("linear", "log"). Default: "log"
    ymin : float, optional
        Minimum y-axis value
    ymax : float, optional
        Maximum y-axis value
    show_mc_errors : bool, optional
        Show 1-sigma error bounds. Default: True
    legend_stat_labels : bool, optional
        Include statistical info in legend. Default: True

    Yields
    ------
    tuple
        (matplotlib.figure.Figure, PlotState) pairs for each rank
    """
    yield from ByRankBandPlotter(
        eigval_grids_by_fit,
        rank_indices=rank_indices,
        xscale=xscale,
        yscale=yscale,
        ymin=ymin,
        ymax=ymax,
        show_mc_errors=show_mc_errors,
        legend_stat_labels=legend_stat_labels,
    )


def plot_eigvals_by_fit(
    eigval_grids_by_fit,
    rank_indices=None,
    xscale=None,
    yscale="log",
    ymin=None,
    ymax=None,
    show_mc_errors: bool = True,
    legend_stat_labels: bool = True,
):
    """
    Plot eigenvalue evolution across epochs, one figure per fit.

    Each figure shows how multiple eigenvalues evolve across epochs for a
    single fit.

    Use case: Comparing eigenvalue spectrum evolution within a single fit.

    Parameters
    ----------
    eigval_grids_by_fit : list of EigenvalueGrid
        List of eigenvalue data containers, one per fit
    rank_indices : list of int, optional
        Which eigenvalue ranks to plot on each figure. Default: first 5
    xscale : str, optional
        X-axis scale ("linear", "log"). Default: "linear"
    yscale : str, optional
        Y-axis scale ("linear", "log"). Default: "log"
    ymin : float, optional
        Minimum y-axis value
    ymax : float, optional
        Maximum y-axis value
    show_mc_errors : bool, optional
        Show 1-sigma error bounds. Default: True
    legend_stat_labels : bool, optional
        Include statistical info in legend. Default: True

    Yields
    ------
    tuple
        (matplotlib.figure.Figure, PlotState) pairs for each fit
    """
    yield from ByFitBandPlotter(
        eigval_grids_by_fit,
        rank_indices=rank_indices,
        xscale=xscale,
        yscale=yscale,
        ymin=ymin,
        ymax=ymax,
        show_mc_errors=show_mc_errors,
        legend_stat_labels=legend_stat_labels,
    )
