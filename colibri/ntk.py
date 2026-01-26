"""
colibri.ntk.py

This module contains the routine that computes the Neural Tangent Kernel (NTK)
for a given PDF model and provides statistical analysis tools for NTK ensembles.

"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from typing import List, Dict, Union
from reportengine import collect

from colibri.ntkutils import (
    compute_eigenvalues_for_replica,
    get_replica_idx_list,
    load_eigenvalues_ensemble,
    get_completed_replicas,
)
from validphys.core import MCStats

log = logging.getLogger(__name__)

class NTKStats(MCStats):
    """
    Container for NTK statistics across replicas at a single epoch.
    """
    def central_value(self):
        return self.data.mean(axis=0)

    def error_members(self):
        return self.data[0:]
    
    def median(self):
        return np.median(self.data, axis=0)

class EigenvalueGrid:
    """
    Container for eigenvalue data from a single fit.

    This class holds eigenvalue statistics across epochs and provides methods
    to extract eigenvalue trajectories for plotting.

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

        self.label = label
        self.epochs = sorted(epochs)
        self._eigenvalues_stats = eigenvalues_stats

        # Validate and extract dimensions
        first_epoch = self.epochs[0]
        first_stats = self._eigenvalues_stats[first_epoch]
        self.nreplicas = first_stats.data.shape[0]
        self.n_eigenvalues = first_stats.data.shape[1]

    def get_eigenvalue_at_epoch(self, epoch: int) -> NTKStats:
        """
        Get NTKStats for all eigenvalues at a specific epoch.

        Parameters
        ----------
        epoch : int
            Epoch number

        Returns
        -------
        NTKStats
            Statistics for all eigenvalues at this epoch, shape (nreplicas, n_eigenvalues)
        """
        if epoch not in self._eigenvalues_stats:
            raise ValueError(f"Epoch {epoch} not found. Available: {self.epochs}")
        return self._eigenvalues_stats[epoch]

    def get_eigenvalue_trajectory(self, rank_index: int) -> NTKStats:
        """
        Get NTKStats for a single eigenvalue across all epochs.

        Parameters
        ----------
        rank_index : int
            Index of the eigenvalue (0 = largest eigenvalue)

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
            self._eigenvalues_stats[epoch].data[:, rank_index]
            for epoch in self.epochs
        ]

        # Stack into (nreplicas, n_epochs) array
        combined_data = np.stack(data_by_epoch, axis=1)
        return NTKStats(combined_data)

    def slice_eigenvalues_at_epoch(
        self, epoch: int, indices: Union[int, list, slice]
    ) -> NTKStats:
        """
        Select specific eigenvalues at an epoch.

        Parameters
        ----------
        epoch : int
            Epoch number
        indices : int, list, or slice
            Eigenvalue indices to select

        Returns
        -------
        NTKStats
            Statistics for selected eigenvalues
        """
        stats = self.get_eigenvalue_at_epoch(epoch)
        data = stats.data

        if isinstance(indices, int):
            sliced_data = data[:, indices : indices + 1]
        elif isinstance(indices, (list, tuple, slice)):
            sliced_data = data[:, indices]
        else:
            raise TypeError(f"Unsupported index type: {type(indices)}")

        return NTKStats(sliced_data)

def ntk_eigenvalues_ensemble(
    fit,
    replicas_path,
    max_workers=None,
    force_recompute: bool = False,
    replica_index_list=None,
    common_epochs_rule: str = "longest",
    max_epoch: int = None,
):
    """
    Compute NTK eigenvalues for all replicas using streaming approach.

    This function computes eigenvalues immediately after each NTK and discards
    the NTK matrix, minimizing memory usage. Each replica is saved to disk
    as soon as it completes, allowing resumption if computation is interrupted.

    Parameters
    ----------
    fit : Fit
        The fit object containing model information
    replicas_path : Path
        Path to the replicas directory
    max_workers : int, optional
        Maximum number of parallel workers. If None, defaults to min(32, n_replicas).
    force_recompute : bool, optional
        If True, recompute all replicas even if cached. Default is False.
    replica_index_list : list, optional
        Specific replica indices to compute. If None, computes all.
    common_epochs_rule : str, optional
        Rule for selecting common epochs across replicas.
    max_epoch : int, optional
        Maximum number of epochs to consider per replica.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'eigenvalues_by_epoch': dict mapping epoch -> ndarray (n_replicas, n_eigenvalues)
        - 'epochs': list of epochs
        - 'ntk_shape': shape of NTK matrix
        - 'replica_indices': list of replica indices included
    """

    # Determine which replicas to compute
    if replica_index_list is None:
        replica_index_list = get_replica_idx_list(replicas_path)

    # Check for already completed replicas
    if force_recompute:
        completed = []
        log.info("Force recompute enabled: ignoring cached replicas")
    else:
        completed = get_completed_replicas(replicas_path)

    pending = sorted([r for r in replica_index_list if r not in completed])

    if not pending:
        log.info(f"All {len(completed)} replicas already computed. Loading from cache.")
        return load_eigenvalues_ensemble(replicas_path, common_epochs_rule, max_epoch)

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
                max_epoch
            ): replica_idx
            for replica_idx in pending
        }

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
    return load_eigenvalues_ensemble(replicas_path, common_epochs_rule, max_epoch)

def eigenvalue_grid(fit, ntk_eigenvalues_ensemble) -> EigenvalueGrid:
    """
    Create an EigenvalueGrid from NTK eigenvalues ensemble data.

    Parameters
    ----------
    fit : Fit
        The fit object containing model information
    ntk_eigenvalues_ensemble : dict
        Output from ntk_eigenvalues_ensemble function
    
    Returns
    -------
    EigenvalueGrid
        The constructed EigenvalueGrid object
    """
    epochs = ntk_eigenvalues_ensemble['epochs']
    eigvals_by_epoch = ntk_eigenvalues_ensemble['eigenvalues_by_epoch']
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
