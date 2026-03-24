"""
colibri.ntk.ntk.py

This module contains the routine that computes the Neural Tangent Kernel (NTK)
for a given PDF model and provides statistical analysis tools for NTK ensembles.

"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from reportengine import collect
from validphys.core import FitSpec

from colibri.ntk.ntkutils import (
    NTKGrid,
    NTKStats,
    compute_eigenvalues_for_replica,
    get_completed_replicas,
    get_replica_idx_list,
    load_eigenvalues_ensemble,
)

log = logging.getLogger(__name__)


class EigenvalueGrid(NTKGrid):
    """
    Container for eigenvalue data from a single fit.

    This class holds eigenvalue statistics across epochs and provides methods
    to extract eigenvalue trajectories for plotting. Implements the NTKGrid
    interface for use with generic plotting utilities.

    Parameters
    ----------
    label : str
        Human-readable label for the fit (e.g., "L0", "L1", "NNPDF4.0")
    epochs : list of int
        List of epoch numbers
    eigenvalues_stats : dict
        Dictionary mapping epoch -> NTKStats object containing eigenvalues.
        Each NTKStats has shape (nreplicas, n_eigenvalues).

    Attributes
    ----------
    label : str
        Label for this fit
    epochs : list of int
        Available epochs
    n_eigenvalues : int
        Number of eigenvalues per replica
    nreplicas : int
        Number of replicas in the ensemble
    """

    def __init__(
        self,
        label: str,
        epochs: List[int],
        eigenvalues_stats: Dict[int, NTKStats],
    ):
        if not epochs:
            raise ValueError("epochs cannot be empty")
        if not eigenvalues_stats:
            raise ValueError("eigenvalues_stats cannot be empty")

        self._label = label
        self.epochs = sorted(epochs)
        self._eigenvalues_stats = eigenvalues_stats

        first_epoch = self.epochs[0]
        first_stats = self._eigenvalues_stats[first_epoch]
        self.nreplicas = first_stats.data.shape[0]
        self.n_eigenvalues = first_stats.data.shape[1]

    @property
    def label(self) -> str:
        """Human-readable label for this grid (e.g., fit name)."""
        return self._label

    @property
    def n_ranks(self) -> int:
        """Number of eigenvalue ranks available."""
        return self.n_eigenvalues

    @property
    def xgrid(self) -> np.ndarray:
        """X-axis grid for plotting (epochs)."""
        return np.array(self.epochs)

    @property
    def xlabel(self) -> str:
        """Label for x-axis."""
        return r"$\rm Epochs$"

    def get_plotting_data(self, rank_index: int, **kwargs) -> NTKStats:
        """
        Get plotting data for a specific eigenvalue rank.

        Parameters
        ----------
        rank_index : int
            Index of the eigenvalue (0 = largest eigenvalue)
        **kwargs
            Ignored for eigenvalues (no additional selection needed)

        Returns
        -------
        NTKStats
            Statistics for the eigenvalue trajectory, shape (nreplicas, n_epochs)
        """
        if rank_index < 0 or rank_index >= self.n_eigenvalues:
            raise ValueError(
                f"rank_index {rank_index} out of range [0, {self.n_eigenvalues})"
            )
        data_by_epoch = [
            self._eigenvalues_stats[epoch].data[:, rank_index] for epoch in self.epochs
        ]

        # Stack into (nreplicas, n_epochs) array
        combined_data = np.stack(data_by_epoch, axis=1)
        return NTKStats(combined_data)

    def get_plotting_label(self, rank_index: int, **kwargs) -> str:
        """
        Get legend label for a specific eigenvalue rank.

        Parameters
        ----------
        rank_index : int
            Index of the eigenvalue (0 = largest eigenvalue)
        **kwargs
            Ignored for eigenvalues

        Returns
        -------
        str
            LaTeX-formatted label (e.g., r"$\\lambda^{(1)}$")
        """
        return rf"$\lambda^{{({rank_index + 1})}}$"


def eigenvalues_ensemble(
    fit: FitSpec,
    replicas_path: Path,
    replica_index_list: Optional[tuple] = None,
    max_epoch: Optional[int] = None,
    force_recompute: bool = False,
    max_workers: Optional[int] = None,
    name: Optional[str] = None,
    kwargs: frozenset = frozenset({}),
):
    """
    Compute NTK eigenvalues for all replicas across all specified epochs.

    This function computes eigenvalues immediately after each NTK and discards
    the NTK matrix. Each replica is saved to disk as soon as it completes.

    Parameters
    ----------
    fit : FitSpec
        The fit object containing fit information
    replicas_path : Path
        Path to the replicas directory
    replica_index_list : tuple, optional
        Specific replica indices to compute. If None, computes all. Mainly for
        testing.
    max_epoch : int, optional
        Maximum number of epochs to consider per replica. If None, uses all
        available epochs. Mainly for testing.
    force_recompute : bool, optional
        If True, recompute all replicas even if cached. Default is False.
    max_workers : int, optional
        Maximum number of parallel workers. If None, defaults to min(10,
        n_replicas).
    name: str, optional
        Optional name to include in the filename for clarity when saving results.
    kwargs : dict, optional
        Additional kwargs to pass to compute_eigenvalues_at_epoch_for_replica

    Returns
    -------
    dict
        Dictionary with keys:
        - `eigenvalues_by_epoch`: dict mapping epoch -> ndarray (n_replicas, n_eigenvalues)
        - `epochs`: list of epochs - 'ntk_shape': shape of NTK matrix
        - `ntk_shape`: shape of the NTK matrix before flattening
        - `replica_indices`: list of replica indices included
    """

    # Determine which replicas to compute
    if replica_index_list is None:
        replica_index_list = get_replica_idx_list(replicas_path)

    # Check for already completed replicas
    if force_recompute:
        completed = []
        log.info("Force recompute enabled: ignoring cached replicas")
    else:
        completed = get_completed_replicas(replicas_path, name)

    pending = sorted([r for r in replica_index_list if r not in completed])

    if not pending:
        log.info(f"All {len(completed)} replicas already computed. Loading from cache.")
        return load_eigenvalues_ensemble(replicas_path, max_epoch, name)

    log.info(
        f"Computing eigenvalues: {len(pending)} pending, "
        f"{len(completed)} already done"
    )

    n_pending = len(pending)
    if max_workers is None:
        max_workers = min(10, n_pending)
    log.info(f"Using max_workers={max_workers} for parallel computation")

    # Compute pending replicas in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_replica = {
            executor.submit(
                compute_eigenvalues_for_replica,
                fit.name,
                replicas_path,
                replica_idx,
                max_epoch,
                name,
                **(kwargs),
            ): replica_idx
            for replica_idx in pending
        }

        # TODO: Do we want to keep that status bar with tqdm?
        # Track progress
        futures_iter = as_completed(future_to_replica)
        for future in tqdm(futures_iter, total=n_pending, desc="Computing eigenvalues"):
            replica_idx = future_to_replica[future]
            try:
                result = future.result()
                if result is None:
                    log.warning(f"Replica {replica_idx} failed")
            except Exception as e:
                log.warning(f"Error computing replica {replica_idx}: {e}")

    # Load all results (completed + newly computed)
    return load_eigenvalues_ensemble(replicas_path, max_epoch, name)


def eigenvalue_grid(fit: FitSpec, eigenvalues_ensemble) -> EigenvalueGrid:
    """
    Create an EigenvalueGrid from NTK eigenvalues ensemble data.

    Parameters
    ----------
    fit : FitSpec
        The fit object containing fit information
    eigenvalues_ensemble : dict
        Output from eigenvalues_ensemble function

    Returns
    -------
    EigenvalueGrid
        The constructed EigenvalueGrid object
    """
    epochs = eigenvalues_ensemble["epochs"]
    eigvals_by_epoch = eigenvalues_ensemble["eigenvalues_by_epoch"]
    label = fit.label

    # Wrap numpy arrays in NTKStats objects
    eigenvalues_stats = {
        epoch: NTKStats(data) for epoch, data in eigvals_by_epoch.items()
    }

    return EigenvalueGrid(
        label=label,
        epochs=epochs,
        eigenvalues_stats=eigenvalues_stats,
    )


# Collect eigenvalue grids across fits
eigval_grids_by_fit = collect("eigenvalue_grid", ("fits",))
