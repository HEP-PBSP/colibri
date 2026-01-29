"""
colibri.ntkutils.py

Module containing several utils for the analysis of the NTK.

"""

from __future__ import annotations

import abc
import logging
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from validphys.core import MCStats

from colibri.constants import XGRID
from colibri.utils import get_pdf_model

log = logging.getLogger(__name__)

NTK_EIGVAL_TOKEN = "ntk_eigenvalues"


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


class NTKGrid(abc.ABC):
    """
    Abstract base class for NTK data containers that can be plotted.

    This interface allows plotting utilities to work uniformly with both
    eigenvalue and eigenvector data. Each implementation must provide:
    - A label identifying the data source (e.g., fit name)
    - The x-axis grid for plotting (e.g., epochs or XGRID)
    - Methods to extract plotting data for specific ranks
    """

    @property
    @abc.abstractmethod
    def label(self) -> str:
        """Human-readable label for this grid (e.g., fit name)."""
        pass

    @property
    @abc.abstractmethod
    def n_ranks(self) -> int:
        """Number of eigenvalue/eigenvector ranks available."""
        pass

    @property
    @abc.abstractmethod
    def xgrid(self) -> np.ndarray:
        """X-axis grid for plotting."""
        pass

    @property
    @abc.abstractmethod
    def xlabel(self) -> str:
        """Label for x-axis."""
        pass

    @abc.abstractmethod
    def get_plotting_data(self, rank_index: int, **kwargs) -> NTKStats:
        """
        Get plotting data (y-values) for a specific rank.

        Parameters
        ----------
        rank_index : int
            Index of the eigenvalue/eigenvector rank (0 = largest)
        **kwargs
            Additional selection parameters as needed by different
            implementations (e.g., flavour_index for eigenvectors)

        Returns
        -------
        NTKStats
            Statistics object containing data of shape (nreplicas, n_xgrid)
        """
        pass

    @abc.abstractmethod
    def get_plotting_label(self, rank_index: int, **kwargs) -> str:
        """
        Get legend label for a specific rank.

        Parameters
        ----------
        rank_index : int
            Index of the eigenvalue/eigenvector rank
        **kwargs
            Additional selection parameters

        Returns
        -------
        str
            LaTeX-formatted label for the legend
        """
        pass


@lru_cache
def get_parameters_all_epochs(replicas_path, replica_index):
    """
    Get paths to model parameters files at all epochs for a given replica.

    Parameters
    ----------
    replicas_path : Path
        Path to the replicas directory
    replica_index : int
        Index of the replica to retrieve

    Returns
    -------
    dict
        Dictionary mapping epoch number to parameter file Path
    """
    params_folder = replicas_path / f"replica_{replica_index}/parameters"
    param_files = list(params_folder.glob("*.npz"))
    param_files.sort(key=lambda f: int(f.stem.split("_")[-1]))

    param_epochs_dict = {}

    for param_file in param_files:
        epoch = int(param_file.stem.split("_")[-1])
        param_epochs_dict[epoch] = param_file

    return param_epochs_dict


def get_replica_idx_list(replicas_path):
    """
    Determine the available replica indices by counting
    the replica directories.

    Parameters
    ----------
    replicas_path : Path
        Path to the replicas directory

    Returns
    -------
    list
        List of replica indices found
    """
    replicas_path = Path(replicas_path)
    if not replicas_path.exists():
        raise FileNotFoundError(f"Replicas path does not exist: {replicas_path}")

    # Count directories named "replica_*"
    replica_dirs = sorted(replicas_path.glob("replica_*"))
    rep_list = [int(d.name.split("_")[1]) for d in replica_dirs]
    return rep_list


def compute_ntk(pdf_model, params):
    """
    Compute the NTK matrix given model parameters.

    Parameters
    ----------
    pdf_model : PDFModel
        The PDF model instance
    params : dict
        Model parameters

    Returns
    -------
    ntk : jnp.ndarray
        The NTK matrix
    ntk_shape : tuple
        Shape of the NTK matrix
    """
    pdf_func = pdf_model.grid_values_func(XGRID)
    jacobian_func = jax.jacfwd(pdf_func)
    jacobian = jacobian_func(params)

    # Compute NTK (14,50,14,50) -> assumes shape from jacobian
    ntk = jnp.einsum("ijk,lmk->ijlm", jacobian, jacobian)

    # Flatten to (N_grid*N_flavors)×(N_grid*N_flavors)
    d1, d2, d3, d4 = ntk.shape
    ntk = ntk.reshape(d1 * d2, d3 * d4)

    return ntk, (d1, d2, d3, d4)


def compute_eigendecomposition(ntk_matrix, hermitian=True):
    """
    Compute eigendecomposition of an NTK matrix.

    Parameters
    ----------
    ntk_matrix : ndarray
        The NTK matrix to decompose
    hermitian : bool, optional
        Whether to use hermitian eigendecomposition (default: True)

    Returns
    -------
    eigenvalues : ndarray
        Eigenvalues in descending order
    eigenvectors : ndarray
        Corresponding eigenvectors (columns)
    """
    if hermitian:
        # For symmetric/hermitian matrices, use eigh for better numerical stability
        eigenvalues, eigenvectors = np.linalg.eigh(ntk_matrix)
        # Sort in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    else:
        # Use SVD for general matrices
        eigenvectors, eigenvalues, _ = np.linalg.svd(ntk_matrix)

    return eigenvalues, eigenvectors


def compute_eigenvalues_for_replica(
    fit_name: str, replicas_path: Path, replica_idx: int, max_epoch=None
):
    """
    Compute the NTK eigenvalues for a given replica across all epochs.

    Parameters
    ----------
    fit_name : str
        Name of the fit (used to load pdf_model)
    replicas_path : Path
        Path to the replicas directory
    replica_idx : int
        Replica index to compute
    max_epoch : int, optional
        Maximum epoch number to consider.

    Returns
    -------
    tuple or None
        (replica_idx, epochs, ntk_shape) on success, None on failure
    """
    try:
        # Create fresh pdf_model to avoid JAX tracer leaks
        pdf_model = get_pdf_model(fit_name)
        param_files = get_parameters_all_epochs(replicas_path, replica_idx)

        eigenvalues_list = []
        epochs = []
        ntk_shape = None

        for epoch, param_file in param_files.items():
            if max_epoch is not None and epoch > max_epoch:
                continue
            params = jnp.load(param_file)["params"]

            ntk, shape = compute_ntk(pdf_model, params)
            if ntk_shape is None:
                ntk_shape = shape

            eigvals, _ = compute_eigendecomposition(ntk, hermitian=True)
            eigenvalues_list.append(eigvals)
            epochs.append(epoch)

        # Stack eigenvalues: (n_epochs, n_eigenvalues)
        eigenvalues = np.stack(eigenvalues_list, axis=0)

        # Save immediately to disk
        save_replica_eigenvalues(
            eigenvalues=eigenvalues,
            epochs=epochs,
            replica_idx=replica_idx,
            replicas_path=replicas_path,
            ntk_shape=ntk_shape,
        )

        return (replica_idx, epochs, ntk_shape)

    except FileNotFoundError as e:
        log.warning(f"Skipping replica {replica_idx}: {e}")
        return None


def get_completed_replicas(replicas_path: Path, token: str = NTK_EIGVAL_TOKEN) -> list:
    """
    Utility function to get list of replica indices for which
    the NTK eigenvalues have already been computed.

    Parameters
    ----------
    replicas_path : Path
        Directory containing replica folders.

    Returns
    -------
    list
        List of completed replica indices
    """
    replicas_path = Path(replicas_path)
    completed = []

    for replica_folder in replicas_path.glob("replica_*"):
        try:
            idx = int(replica_folder.stem.split("_")[1])
            replica_file = replica_folder / f"{token}_{idx}.npz"
            if replica_file.exists():
                completed.append(idx)
        except (ValueError, IndexError) as e:
            log.debug(f"Skipping folder {replica_folder.name}: {e}")
            continue

    return sorted(completed)


def save_replica_eigenvalues(
    eigenvalues: np.ndarray,
    epochs: list,
    replica_idx: int,
    replicas_path: Path,
    ntk_shape: tuple = None,
) -> None:
    """
    Save eigenvalues for a single replica to disk.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues array of shape (n_epochs, n_eigenvalues)
    epochs : list
        List of epoch numbers
    replica_idx : int
        Replica index
    replicas_path : Path
        Directory to save results
    ntk_shape : tuple, optional
        Shape of the NTK matrix (saved in metadata)
    """
    replica_file = (
        replicas_path / f"replica_{replica_idx}/{NTK_EIGVAL_TOKEN}_{replica_idx}.npz"
    )
    np.savez_compressed(
        replica_file,
        eigenvalues=eigenvalues,
        epochs=np.array(epochs),
        ntk_shape=ntk_shape,
    )
    log.debug(f"Saved eigenvalues for replica {replica_idx} to {replica_file}")


def load_replica_eigenvalues(replica_idx: int, cache_dir: Path) -> dict:
    """
    Load eigenvalues for a single replica from disk.

    Parameters
    ----------
    replica_idx : int
        Replica index
    cache_dir : Path
        Directory containing saved results

    Returns
    -------
    dict
        Dictionary with 'eigenvalues' (n_epochs, n_eigenvalues) and 'epochs'
    """
    replica_file = (
        cache_dir / f"replica_{replica_idx}/{NTK_EIGVAL_TOKEN}_{replica_idx}.npz"
    )

    if not replica_file.exists():
        raise FileNotFoundError(f"Replica {replica_idx} not found at {replica_file}")

    data = np.load(replica_file)
    return {
        "eigenvalues": data["eigenvalues"],
        "epochs": data["epochs"].tolist(),
        "ntk_shape": data["ntk_shape"],
    }


def load_eigenvalues_ensemble(
    replicas_path: Path, max_epoch=None
) -> dict:
    """
    Load all replica eigenvalues into an ensemble format.

    Parameters
    ----------
    replicas_path : Path
        Path to replica folders.
    max_epoch : int, optional
        Maximum epoch to consider. It filters out replicas that do not have
        data up to this epoch.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'eigenvalues_by_epoch': dict mapping epoch -> ndarray (n_replicas, n_eigenvalues)
        - 'epochs': list of epochs
        - 'ntk_shape': shape of NTK matrix
        - 'replica_indices': list of replica indices included
    """
    completed_replicas = get_completed_replicas(replicas_path)

    if not completed_replicas:
        raise ValueError(f"No completed replicas found in {replicas_path}")

    # Load all replicas
    all_eigenvalues = []
    included_replicas = []
    ntk_shape = None
    for replica_idx in completed_replicas:
        data = load_replica_eigenvalues(replica_idx, replicas_path)
        epochs = np.array(data["epochs"])
        eigenvalues = data["eigenvalues"]  # (n_epochs, n_eigenvalues)

        if ntk_shape is None:
            ntk_shape = data["ntk_shape"]
        
        # If max_epoch is set, filter epochs and eigenvalues
        if max_epoch is not None:
            if max_epoch not in epochs:
                log.warning(
                    f"Replica {replica_idx} does not contain epoch {max_epoch}. "
                    f"Last epoch is {epochs[-1]}. Excluded from ensemble."
                )
                continue
            mask = [e <= max_epoch for e in epochs]
            epochs = epochs[mask]
            eigenvalues = eigenvalues[mask]

        all_eigenvalues.append((replica_idx, epochs, eigenvalues))
        included_replicas.append(replica_idx)

    if not all_eigenvalues:
        raise ValueError("No replicas have epochs up to the specified max_epoch.")
    
    # Determine common epochs across included replicas
    all_epoch_sets = [set(epochs) for _, epochs, _ in all_eigenvalues]
    common_epochs = sorted(set.intersection(*all_epoch_sets))
    if not common_epochs:
        raise ValueError("No common epochs found across replicas.")
    
    eigenvalues_by_epoch = {epoch: [] for epoch in common_epochs}
    for replica_idx, epochs, eigenvalues in all_eigenvalues:
        epoch_to_idx = {e: i for i, e in enumerate(epochs)}
        for epoch in common_epochs:
            idx = epoch_to_idx[epoch]
            eigenvalues_by_epoch[epoch].append(eigenvalues[idx])

    # Stack into arrays
    for epoch in common_epochs:
        eigenvalues_by_epoch[epoch] = np.stack(eigenvalues_by_epoch[epoch], axis=0)

    log.info(
        f"Loaded eigenvalues ensemble: {len(included_replicas)} replicas, "
        f"{len(common_epochs)} epochs"
    )

    return {
        "eigenvalues_by_epoch": eigenvalues_by_epoch,
        "epochs": common_epochs,
        "ntk_shape": ntk_shape,
        "replica_indices": included_replicas,
    }


def compute_eigenvectors_at_apoch_for_replica(
    fit_name: str,
    replicas_path: Path,
    replica_idx: int,
    epoch: int,
):
    """
    Compute the eigenvectors of the NTK at a given epoch for a specific replica.

    Parameters
    ----------
    fit_name : str
        Name of the fit (used to load pdf_model)
    replicas_path : Path
        Path to the replicas directory
    replica_idx : int
        Replica index to compute
    epoch : int
        Epoch number at which to compute eigenvectors
    """
    try:
        pdf_model = get_pdf_model(fit_name)
        param_files = get_parameters_all_epochs(replicas_path, replica_idx)

        params_at_epoch = param_files.get(epoch, None)
        if params_at_epoch is None:
            raise ValueError(
                f"Epoch {epoch} not found for replica {replica_idx} in fit {fit_name}"
            )

        params = jnp.load(params_at_epoch)["params"]
        ntk, shape = compute_ntk(pdf_model, params)
        _, eigvecs = compute_eigendecomposition(ntk, hermitian=True)

        return (replica_idx, epoch, eigvecs, shape)
    except Exception as e:
        log.warning(f"Skipping replica {replica_idx}: {e}")
        return None
