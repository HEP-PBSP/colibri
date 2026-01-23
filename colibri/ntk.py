"""
colibri.ntk.py

This module contains the routine that computes the Neural Tangent Kernel (NTK)
for a given PDF model and provides statistical analysis tools for NTK ensembles.

"""

import logging

import numpy as np

from typing import List, Dict, Union
from reportengine import collect

from colibri.utils import get_pdf_model
from colibri.ntkutils import compute_ntk, compute_eigendecomposition, get_replica_idx_list
from validphys.core import MCStats

log = logging.getLogger(__name__)


def ntk_ensemble(fit, replicas_path):
    """
    Compute NTK for all replicas.

    This is a provider function that computes the NTK ensemble across all replicas.
    It returns a dictionary mapping epochs to lists of NTK matrices.

    Parameters
    ----------
    loaded_model : PDFModel
        The PDF model instance
    replicas_path : Path
        Path to the replicas directory

    Returns
    -------
    dict
        Dictionary with keys:
        - 'ntk_by_epoch': dict mapping epoch -> list of NTK matrices (one per replica)
        - 'epochs': list of common epochs across all replicas
        - `ntk_shape`: shape of the NTK matrix before flattening
    """
    # Storage for NTKs organized by epoch
    ntk_by_epoch = {}
    common_epochs = None
    pdf_model = get_pdf_model(fit.name)
    ntk_shape = None

    replica_index_list = get_replica_idx_list(replicas_path)
    for replica_idx in replica_index_list:
        try: 
            ntk_list, epochs, ntk_shape = compute_ntk(pdf_model, replicas_path, replica_idx)

            if ntk_shape is None:
                ntk_shape = ntk_shape

            # First replica: initialize the structure
            if common_epochs is None:
                common_epochs = epochs
                for epoch in epochs:
                    ntk_by_epoch[epoch] = []

            # Check that all replicas have the same epochs
            if epochs != common_epochs:
                log.warning(
                    f"Replica {replica_idx} has different epochs {epochs} "
                    f"vs common {common_epochs}. Using intersection."
                )
                common_epochs = sorted(set(epochs) & set(common_epochs))

            # Add NTKs to the appropriate epoch
            for epoch, ntk in zip(epochs, ntk_list):
                if epoch in common_epochs:
                    ntk_by_epoch[epoch].append(ntk)

        except FileNotFoundError as e:
            log.warning(f"Skipping replica {replica_idx}: {e}")
            continue

    # Clean up ntk_by_epoch to only include common epochs
    ntk_by_epoch = {epoch: ntk_by_epoch[epoch] for epoch in common_epochs}

    log.info(f"NTK ensemble computed for epochs: {common_epochs}")

    return {
        "ntk_by_epoch": ntk_by_epoch,
        "epochs": common_epochs,
        "ntk_shape": ntk_shape,
    }


def ntk_eigendecomposition_ensemble(ntk_ensemble):
    """
    Compute eigendecomposition for the entire NTK ensemble.

    This is a provider function that takes the output of `ntk_ensemble`
    and computes eigenvalues and eigenvectors for each NTK matrix.

    Parameters
    ----------
    ntk_ensemble : dict
        Dictionary from ntk_ensemble() containing NTK matrices

    Returns
    -------
    dict
        Dictionary with keys:
        - 'eigenvalues_by_epoch': dict mapping epoch -> ndarray of shape (nreplicas, n_eigenvalues)
        - 'eigenvectors_by_epoch': dict mapping epoch -> list of ndarray (one per replica)
        - 'epochs': list of epochs
        - `ntk_shape`: shape of the NTK matrix before flattening
    """
    ntk_by_epoch = ntk_ensemble["ntk_by_epoch"]
    epochs = ntk_ensemble["epochs"]

    log.info(f"Computing eigendecomposition for {len(epochs)} epochs")

    eigenvalues_by_epoch = {}
    eigenvectors_by_epoch = {}

    for epoch in epochs:
        ntk_list = ntk_by_epoch[epoch]
        n_replicas_at_epoch = len(ntk_list)

        eigenvalues_list = []
        eigenvectors_list = []

        for replica_idx, ntk in enumerate(ntk_list):
            log.debug(f"Eigendecomposition for epoch {epoch}, replica {replica_idx}")
            eigvals, eigvecs = compute_eigendecomposition(ntk, hermitian=True)
            eigenvalues_list.append(eigvals)
            eigenvectors_list.append(eigvecs)

        # Stack eigenvalues into (nreplicas, n_eigenvalues) array
        eigenvalues_by_epoch[epoch] = np.stack(eigenvalues_list, axis=0)
        eigenvectors_by_epoch[epoch] = eigenvectors_list

    log.info("Eigendecomposition complete")

    return {
        "eigenvalues_by_epoch": eigenvalues_by_epoch,
        "eigenvectors_by_epoch": eigenvectors_by_epoch,
        "epochs": epochs,
        "ntk_shape": ntk_ensemble["ntk_shape"],
    }


def ntk_eigenvalues(ntk_eigendecomposition_ensemble, epoch=None):
    """
    Extract eigenvalues for a specific epoch or all epochs.

    Parameters
    ----------
    ntk_eigendecomposition_ensemble : dict
        Output from ntk_eigendecomposition_ensemble()
    epoch : int, optional
        Specific epoch to extract. If None, returns dict of all epochs.

    Returns
    -------
    ndarray or dict
        If epoch is specified: array of shape (nreplicas, n_eigenvalues)
        If epoch is None: dict mapping epoch -> array
    """
    eigenvalues_by_epoch = ntk_eigendecomposition_ensemble["eigenvalues_by_epoch"]

    if epoch is not None:
        if epoch not in eigenvalues_by_epoch:
            raise ValueError(
                f"Epoch {epoch} not found. Available: {list(eigenvalues_by_epoch.keys())}"
            )
        return eigenvalues_by_epoch[epoch]

    return eigenvalues_by_epoch


def ntk_eigenvectors(ntk_eigendecomposition_ensemble, epoch=None):
    """
    Extract eigenvectors for a specific epoch or all epochs.

    Parameters
    ----------
    ntk_eigendecomposition_ensemble : dict
        Output from ntk_eigendecomposition_ensemble()
    epoch : int, optional
        Specific epoch to extract. If None, returns dict of all epochs.

    Returns
    -------
    list or dict
        If epoch is specified: list of eigenvector arrays (one per replica)
        If epoch is None: dict mapping epoch -> list of arrays
    """
    eigenvectors_by_epoch = ntk_eigendecomposition_ensemble["eigenvectors_by_epoch"]

    if epoch is not None:
        if epoch not in eigenvectors_by_epoch:
            raise ValueError(
                f"Epoch {epoch} not found. Available: {list(eigenvectors_by_epoch.keys())}"
            )
        return eigenvectors_by_epoch[epoch]

    return eigenvectors_by_epoch


def ntk_eigenvalues_stats_all_epochs(ntk_eigenvalues):
    """
    Create MCStats objects for eigenvalues at all epochs.

    Parameters
    ----------
    ntk_eigenvalues : dict
        Dictionary mapping epoch -> eigenvalues array

    Returns
    -------
    dict
        Dictionary mapping epoch -> MCStats object
    """
    return {epoch: MCStats(eigvals) for epoch, eigvals in ntk_eigenvalues.items()}


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
        Dictionary mapping epoch -> MCStats object containing eigenvalues.
        Each MCStats has shape (nreplicas, n_eigenvalues).

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

    Examples
    --------
    >>> eigval_grid = EigenvalueGrid(
    ...     label="My Fit",
    ...     epochs=[0, 100, 200],
    ...     eigenvalues_stats=eigvals_stats_dict
    ... )
    >>> traj = eigval_grid.get_eigenvalue_trajectory(rank_index=0)
    """

    def __init__(
        self,
        label: str,
        epochs: List[int],
        eigenvalues_stats: Dict[int, MCStats],
    ):
        self.label = label
        self.epochs = sorted(epochs)
        self._eigenvalues_stats = eigenvalues_stats

        # Validate and extract dimensions
        first_epoch = self.epochs[0]
        first_stats = self._eigenvalues_stats[first_epoch]
        self.nreplicas = first_stats.data.shape[0]
        self.n_eigenvalues = first_stats.data.shape[1]

    def get_eigenvalue_at_epoch(self, epoch: int) -> MCStats:
        """
        Get MCStats for all eigenvalues at a specific epoch.

        Parameters
        ----------
        epoch : int
            Epoch number

        Returns
        -------
        MCStats
            Statistics for all eigenvalues at this epoch, shape (nreplicas, n_eigenvalues)
        """
        if epoch not in self._eigenvalues_stats:
            raise ValueError(f"Epoch {epoch} not found. Available: {self.epochs}")
        return self._eigenvalues_stats[epoch]

    def get_eigenvalue_trajectory(self, rank_index: int) -> MCStats:
        """
        Get MCStats for a single eigenvalue across all epochs.

        This is similar to `combine_distributions(ctx.eigvals_time).slice((slice(None), idx))`
        from yadlt.

        Parameters
        ----------
        rank_index : int
            Index of the eigenvalue (0 = largest eigenvalue)

        Returns
        -------
        MCStats
            Statistics for the eigenvalue trajectory, shape (nreplicas, n_epochs)
        """
        if rank_index < 0 or rank_index >= self.n_eigenvalues:
            raise ValueError(
                f"rank_index {rank_index} out of range [0, {self.n_eigenvalues})"
            )

        data_by_epoch = []
        for epoch in self.epochs:
            stats = self._eigenvalues_stats[epoch]
            eigenval_at_epoch = stats.data[:, rank_index]
            data_by_epoch.append(eigenval_at_epoch)

        # Stack into (nreplicas, n_epochs) array
        combined_data = np.stack(data_by_epoch, axis=1)
        return MCStats(combined_data)

    def slice_eigenvalues_at_epoch(
        self, epoch: int, indices: Union[int, list, slice]
    ) -> MCStats:
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
        MCStats
            Statistics for selected eigenvalues
        """
        stats = self.get_eigenvalue_at_epoch(epoch)
        data = stats.data

        if isinstance(indices, int):
            sliced_data = data[:, indices : indices + 1]
        elif isinstance(indices, (list, tuple)):
            sliced_data = data[:, indices]
        elif isinstance(indices, slice):
            sliced_data = data[:, indices]
        else:
            raise TypeError(f"Unsupported index type: {type(indices)}")

        return MCStats(sliced_data)


def eigenvalue_grid_from_stats(
    label: str,
    ntk_eigenvalues_stats_all_epochs: Dict[int, MCStats],
) -> EigenvalueGrid:
    """
    Create an EigenvalueGrid from an ntk_eigenvalues_stats_all_epochs dictionary.

    This is a convenience function to create an EigenvalueGrid from the output
    of the `ntk_eigenvalues_stats_all_epochs` provider.

    Parameters
    ----------
    label : str
        Label for this fit
    ntk_eigenvalues_stats_all_epochs : dict
        Dictionary mapping epoch -> MCStats

    Returns
    -------
    EigenvalueGrid
    """
    epochs = sorted(ntk_eigenvalues_stats_all_epochs.keys())
    return EigenvalueGrid(
        label=label,
        epochs=epochs,
        eigenvalues_stats=ntk_eigenvalues_stats_all_epochs,
    )


def eigenvalue_grid(
    fit,
    ntk_eigenvalues_stats_all_epochs,
):
    """
    Create an EigenvalueGrid from NTK eigenvalue statistics.

    This is a provider function for reportengine integration.

    Parameters
    ----------
    fit_label : str
        Label for this fit
    ntk_eigenvalues_stats_all_epochs : dict
        Dictionary mapping epoch -> MCStats (from ntk_eigenvalues_stats_all_epochs)

    Returns
    -------
    EigenvalueGrid
    """
    label = fit.label
    return eigenvalue_grid_from_stats(label, ntk_eigenvalues_stats_all_epochs)

# Collect eigenvalue grids across fits
eigval_grids_by_fit = collect("eigenvalue_grid", ("fits",))
