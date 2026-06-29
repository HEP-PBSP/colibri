"""
colibri.ntk.eigenvector.py
"""

import functools
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

from reportengine import collect
from validphys.core import FitSpec

from colibri.constants import XGRID, FLAVOUR_TO_ID_MAPPING
from colibri.ntk.ntkutils import (
    NTKGrid,
    NTKStats,
    NTK_ORDERING,
    compute_eigenvectors_at_epoch_for_replica,
    get_replica_idx_list,
)

log = logging.getLogger(__name__)


class EigenvectorGrid(NTKGrid):
    """
    Container for eigenvector data from a single fit at a specific epoch.

    This class holds eigenvector statistics and provides methods to extract
    eigenvector components for plotting. Implements the NTKGrid interface
    for use with generic plotting utilities.

    The eigenvector data has shape (nreplicas, n_eigenvectors, nflavors * n_xgrid).

    Parameters
    ----------
    label : str
        Human-readable label for the fit (e.g., "L0", "L1")
    epoch : int
        The epoch at which eigenvectors were computed
    shape : tuple[int]
        The shape of the NTK matrix before flattening (nflavors, n_xgrid, nflavors, n_xgrid)
    eigenvectors_stat : NTKStats
        Statistics object with data of shape (nreplicas, n_eigenvectors, nflavors * n_xgrid)

    Attributes
    ----------
    label : str
        Label for this fit
    epoch : int
        Epoch number
    nflavors : int
        Number of flavours
    n_eigenvectors : int
        Number of eigenvectors
    n_xgrid : int
        Number of x-grid points
    nreplicas : int
        Number of replicas
    """

    def __init__(
        self,
        label: str,
        epoch: int,
        shape: tuple[int],
        eigenvectors_stat: NTKStats,
    ):
        # Check input dimension
        if len(eigenvectors_stat.data.shape) != 3:
            raise ValueError(
                f"eigenvectors_stat data must be 3D (nreplicas, n_eigenvalues, n_parameters), "
                f"got shape {eigenvectors_stat.data.shape}"
            )

        self._label = label
        self.epoch = epoch
        self.nflavors = shape[0]
        self._eigenvectors_stat = eigenvectors_stat

        self.nreplicas = len(eigenvectors_stat.error_members())
        self.n_eigenvectors = shape[0] * shape[1]
        self.n_xgrid = shape[1]

    # NTKGrid interface implementation
    @property
    def label(self) -> str:
        """Human-readable label for this grid (e.g., fit name)."""
        return self._label

    @property
    def n_ranks(self) -> int:
        """Number of eigenvector ranks available."""
        return self.n_eigenvectors

    @property
    def xgrid(self) -> np.ndarray:
        """X-axis grid for plotting (x values)."""
        return np.array(XGRID)

    @property
    def xlabel(self) -> str:
        """Label for x-axis."""
        return r"$x$"

    def get_stat(self) -> NTKStats:
        """Get the full eigenvector statistics object."""
        return self._eigenvectors_stat

    def get_plotting_data(
        self, rank_index: int, flavour_index: int = 0, **kwargs
    ) -> NTKStats:
        """
        Get eigenvector component for plotting.

        This method reshapes (nreplicas, nflavors*n_xgrid) -> (nreplicas,
        nflavors, n_xgrid) and selects the specified flavour.

        Parameters
        ----------
        rank_index : int
            Index of the eigenvector (1 = largest eigenvalue's eigenvector)
        flavour_index : int, optional
            The ID that represents a specific flavour (see
            `FLAVOUR_TO_ID_MAPPING` in `colibri.constants`)
        **kwargs
            Additional kwargs (ignored)

        Returns
        -------
        NTKStats
            Statistics for the eigenvector component, shape (nreplicas, n_xgrid)
        """
        if rank_index <= 0 or rank_index > self.n_eigenvectors:
            raise ValueError(
                f"rank_index {rank_index} out of range [0, {self.n_eigenvectors}]"
            )
        if flavour_index < 0 or flavour_index >= self.nflavors:
            raise ValueError(
                f"flavour_index {flavour_index} out of range [0, {self.nflavors})"
            )

        # Get eigenvector data for the specified rank: (nreplicas, n_flaovors * n_xgrid)
        eigvec_data = self._eigenvectors_stat.data[:, :, rank_index - 1]

        # Reshape to (nreplicas, nflavors, n_xgrid)
        reshaped = eigvec_data.reshape(
            self.nreplicas, self.nflavors, self.n_xgrid, order=NTK_ORDERING
        )

        # Select the specified flavour: (nreplicas, n_xgrid)
        flavour_data = reshaped[:, flavour_index, :]

        return NTKStats(flavour_data)

    def get_plotting_label(
        self, rank_index: int, flavour_index: int = 0, **kwargs
    ) -> str:
        """
        Get legend label for a specific eigenvector component.

        Parameters
        ----------
        rank_index : int
            Index of the eigenvector (1 = largest eigenvalue's eigenvector)
        flavour_index : int, optional
            Index of the flavour (see `FLAVOUR_TO_ID_MAPPING` in `colibri.constants`)
        **kwargs
            Additional kwargs (ignored)

        Returns
        -------
        str
            LaTeX-formatted label (e.g., r"$v^{(1)}_{\rm GLUON}$")
        """
        # TODO: choose if to include flavour in label
        return rf"$z^{{({rank_index})}}$"


@functools.cache
def eigenvectors_ensemble_at_epoch(
    fit: FitSpec,
    replicas_path: Path,
    epoch: int,
    replica_index_list: Optional[tuple] = None,
    max_workers: Optional[int] = None,
    kwargs: frozenset = frozenset({}),
):
    """
    Compute NTK eigenvalues for all replicas for a specified epoch.

    Parameters
    ----------
    fit : FitSpec
        The fit object containing fit information
    replicas_path : Path
        Path to the replicas directory
    epoch : int
        The epoch at which to compute eigenvectors
    replica_index_list : tuple, optional
        Specific replica indices to compute. If None, computes all. Mainly for
        testing.
    max_workers : int, optional
        Maximum number of parallel workers. If None, defaults to min(10,
        n_replicas).
    kwargs : dict, optional
        Additional kwargs to pass to compute_eigenvectors_at_epoch_for_replica

    Returns
    -------
    dict
        Dictionary with keys:
        - `eigenvectors_data`: ndarray (n_replicas, n_flav * n_xgrid, n_eigenvectors) (flavor-major)
        - `epoch`: the epoch number
        - `ntk_shape`: shape of the NTK matrix before flattening
        - `replica_indices`: list of replica indices included
    """

    # Determine which replicas to compute
    if replica_index_list is None:
        replica_index_list = get_replica_idx_list(replicas_path)
    n_replicas = len(replica_index_list)

    if max_workers is None:
        max_workers = min(10, n_replicas)
    log.info(f"Using max_workers={max_workers} for parallel computation")

    eigenvectors_data = {}
    shape = None

    # Compute pending replicas in parallel
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_replica = {
            executor.submit(
                compute_eigenvectors_at_epoch_for_replica,
                fit.name,
                replicas_path,
                replica_idx,
                epoch,
                **dict(kwargs),
            ): replica_idx
            for replica_idx in replica_index_list
        }

        futures_iter = as_completed(future_to_replica)
        for future in tqdm(
            futures_iter, total=n_replicas, desc="Computing eigenvectors"
        ):
            replica_idx = future_to_replica[future]
            try:
                result = future.result()
                if result is None:
                    log.warning(f"Replica {replica_idx} failed")
                eigenvectors_data[replica_idx] = result[2]
                with lock:
                    if shape is None:
                        shape = result[3]
            except Exception as e:
                log.warning(f"Error computing replica {replica_idx}: {e}")

        # Check we have at least one replicas
        if not eigenvectors_data:
            raise RuntimeError("No replicas were successfully computed.")

        # Convert the final dict into a ndarray
        eigenvectors_data = np.array(
            [
                eigenvectors_data[replica_idx]
                for replica_idx in sorted(eigenvectors_data.keys())
            ]
        )  # Shape: (nreplicas, n_eigenvectors, n_eigenvectors)

        return {
            "eigenvectors_data": eigenvectors_data,
            "epoch": epoch,
            "shape": shape,
            "replica_indices": replica_index_list,
        }


def eigenvector_grid(fit: FitSpec, eigenvectors_ensemble_at_epoch) -> EigenvectorGrid:
    """
    Create an EigenvectorGrid from NTK eigenvectors ensemble data.

    Parameters
    ----------
    fit : FitSpec
        The fit object containing fit information
    eigenvectors_ensemble_at_epoch : dict
        Output from eigenvectors_ensemble_at_epoch function

    Returns
    -------
    EigenvectorGrid
        The constructed EigenvectorGrid object
    """
    eigenvectors_array = eigenvectors_ensemble_at_epoch["eigenvectors_data"]
    eigenvectors_stat = NTKStats(data=eigenvectors_array)
    shape = eigenvectors_ensemble_at_epoch["shape"]

    return EigenvectorGrid(
        label=fit.label,
        epoch=eigenvectors_ensemble_at_epoch["epoch"],
        shape=shape,
        eigenvectors_stat=eigenvectors_stat,
    )


def eigenvectors_at_epoch(
    eigenvector_grid: EigenvectorGrid,
    evol_fl_names: list = list(FLAVOUR_TO_ID_MAPPING.keys()),
) -> NTKStats:
    """Returns DataFrame with eigenvector components for specified flavours."""
    eigvec_data = (
        eigenvector_grid.get_stat().data
    )  # Shape (nreplicas, nflavors * n_xgrid, n_eigenvectors)

    cl_index = pd.Index(range(eigenvector_grid.n_eigenvectors), name="rank")

    # Index follows NTK_ORDERING: for each flavour, all x-points in sequence
    index = pd.MultiIndex.from_tuples(
        [(fl, i + 1) for fl in evol_fl_names for i in range(len(XGRID))],
        names=["flavour", "x"],
    )
    if len(evol_fl_names) > eigenvector_grid.nflavors:
        raise ValueError(
            f"flavour_indices {evol_fl_names} out of range [0, {eigenvector_grid.nflavors})"
        )

    dfs = [
        pd.DataFrame(data=eigvec_data[k], index=index, columns=cl_index)
        for k in range(eigenvector_grid.nreplicas)
    ]

    return NTKStats(dfs)


eigvecs_grids_by_fit = collect("eigenvector_grid", ("fits",))
