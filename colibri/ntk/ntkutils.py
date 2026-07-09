"""
colibri.ntkutils.py

Module containing several utils for the analysis of the NTK.

"""

from __future__ import annotations

import abc
import functools
import logging
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from validphys.core import MCStats

from colibri.constants import XGRID
from colibri.utils import get_pdf_model

log = logging.getLogger(__name__)

NTK_EIGVAL_TOKEN = "ntk_eigenvalues"
NTK_ORDERING = "C"  # flavour-major C-order: all x-points per flavour, then next flavour


def _check_replicas(self, other):
    if isinstance(other, NTKStats) and self.data.shape[0] != other.data.shape[0]:
        raise ValueError(
            f"NTKStats replica count mismatch: {self.data.shape[0]} vs {other.data.shape[0]}"
        )


def _check_indices(self, other, op_name):
    other_index, other_columns = self._get_index(other)
    if op_name == "__matmul__":
        if (
            self._df_columns is not None
            and other_index is not None
            and not self._df_columns.equals(other_index)
        ):
            raise ValueError(
                f"Index mismatch in matmul: left columns {self._df_columns.tolist()} "
                f"do not match right index {other_index.tolist()}"
            )
    elif op_name == "__rmatmul__":
        if (
            other_columns is not None
            and self._df_index is not None
            and not other_columns.equals(self._df_index)
        ):
            raise ValueError(
                f"Index mismatch in matmul: left columns {other_columns.tolist()} "
                f"do not match right index {self._df_index.tolist()}"
            )
    else:
        if (
            self._df_index is not None
            and other_index is not None
            and not self._df_index.equals(other_index)
        ):
            raise ValueError(
                f"Index mismatch in {op_name}: left index {self._df_index.tolist()} "
                f"does not match right index {other_index.tolist()}"
            )
        if (
            self._df_columns is not None
            and other_columns is not None
            and not self._df_columns.equals(other_columns)
        ):
            raise ValueError(
                f"Column index mismatch in {op_name}: left columns {self._df_columns.tolist()} "
                f"do not match right columns {other_columns.tolist()}"
            )


def _checks_ntkstats_compat(method):
    """Decorator that applies replica and index compatibility checks."""

    @functools.wraps(method)
    def wrapper(self, other):
        _check_replicas(self, other)
        _check_indices(self, other, method.__name__)
        return method(self, other)

    return wrapper


class NTKStats(MCStats):
    """
    Container for NTK statistics across replicas at a single epoch.

    When constructed with a list of DataFrames, the index is preserved and
    accessible via the ``frames`` property, while ``data`` remains a numpy
    array for all statistical operations.
    """

    # Tell numpy's ufunc dispatch (which backs the @ operator since numpy ≥ 1.16)
    # to return NotImplemented rather than coercing this object, so that Python
    # can fall through to NTKStats.__rmatmul__ when numpy arrays appear on the left.
    __array_ufunc__ = None

    def __init__(self, data):
        if isinstance(data, list) and data and isinstance(data[0], pd.DataFrame):
            # Check if columns and index are consistent across all DataFrames
            for df in data:
                if not isinstance(df, pd.DataFrame):
                    raise ValueError(
                        "All items in the data list must be pandas DataFrames."
                    )
            # if not all(df.index.equals(data[0].index) for df in data):
            #     raise ValueError("All DataFrames must have the same index.")
            # if not all(df.columns.equals(data[0].columns) for df in data):
            #     raise ValueError("All DataFrames must have the same columns.")
            self._df_index = data[0].index
            self._df_columns = data[0].columns
            super().__init__(np.stack([df.values for df in data]))
        else:
            self._df_index = None
            self._df_columns = None
            super().__init__(data)

        self.shape = self.data.shape[1:]
        self.ndim = (
            len(self.data.shape) - 1
        )  # Number of dimensions of the observable (e.g., 0 for scalar, 1 for vector, 2 for matrix)
        self.nreplica = self.data.shape[0]  # Number of replicas (first dimension)

    @property
    def frames(self):
        """Return data as a list of DataFrames (preserving the original index), or None."""
        if self._df_index is None:
            return self.data
        return [
            pd.DataFrame(self.data[k], index=self._df_index, columns=self._df_columns)
            for k in range(len(self.data))
        ]

    def _with_index(self, data: np.ndarray) -> NTKStats:
        """Wrap a numpy array in NTKStats, preserving this instance's index metadata."""
        result = NTKStats(data)
        result._df_index = self._df_index
        result._df_columns = self._df_columns
        return result

    def _other_data(self, other):
        return other.data if isinstance(other, NTKStats) else other

    def central_value(self):
        cv = self.data.mean(axis=0)
        if self._df_index is not None and self._df_columns is not None:
            return pd.DataFrame(cv, index=self._df_index, columns=self._df_columns)
        return cv

    def error_members(self):
        return self.data

    def median(self):
        med = np.median(self.data, axis=0)
        if self._df_index is not None and self._df_columns is not None:
            return pd.DataFrame(med, index=self._df_index, columns=self._df_columns)
        return med

    def std_error(self):
        std = np.std(self.data, axis=0)
        if self._df_index is not None and self._df_columns is not None:
            return pd.DataFrame(std, index=self._df_index, columns=self._df_columns)
        return std

    @_checks_ntkstats_compat
    def __add__(self, other):
        return self._with_index(self.data + self._other_data(other))

    @_checks_ntkstats_compat
    def __radd__(self, other):
        return self._with_index(self._other_data(other) + self.data)

    @_checks_ntkstats_compat
    def __sub__(self, other):
        return self._with_index(self.data - self._other_data(other))

    @_checks_ntkstats_compat
    def __rsub__(self, other):
        return self._with_index(self._other_data(other) - self.data)

    @_checks_ntkstats_compat
    def __mul__(self, other):
        return self._with_index(self.data * self._other_data(other))

    @_checks_ntkstats_compat
    def __rmul__(self, other):
        return self._with_index(self._other_data(other) * self.data)

    @_checks_ntkstats_compat
    def __truediv__(self, other):
        return self._with_index(self.data / self._other_data(other))

    @_checks_ntkstats_compat
    def __rtruediv__(self, other):
        return self._with_index(self._other_data(other) / self.data)

    @property
    def T(self) -> NTKStats:
        """Transpose each replica's matrix; requires 3D data (Nrep, d1, d2)."""
        if self.data.ndim != 3:
            raise ValueError(
                f"Transpose requires matrix observables (Nrep, d1, d2), got {self.data.shape}"
            )
        result = NTKStats(self.data.transpose(0, 2, 1))
        result._df_index = self._df_columns
        result._df_columns = self._df_index
        return result

    def _get_index(self, other):
        """Return (row_index, col_index) for other, supporting DataFrame and NTKStats."""
        if isinstance(other, pd.DataFrame):
            return other.index, other.columns
        if isinstance(other, NTKStats):
            return other._df_index, other._df_columns
        return None, None

    def set_index(self, index, columns):
        """Return a new NTKStats with the given index and columns."""
        self._df_index = index
        self._df_columns = columns

    @_checks_ntkstats_compat
    def __matmul__(self, other) -> NTKStats:
        _, other_cols = self._get_index(other)

        other_data = (
            other.values if isinstance(other, pd.DataFrame) else self._other_data(other)
        )

        if isinstance(other, NTKStats) and other_data.ndim == 2:
            # Vector per replica (Nrep, n): treat as (Nrep, n, 1), multiply, then squeeze.
            result_data = (self.data @ other_data[:, :, None]).squeeze(-1)
        else:
            # Plain 2D matrix (no replica dim): add batch dim so numpy broadcasts over replicas.
            if (
                not isinstance(other, NTKStats)
                and isinstance(other_data, np.ndarray)
                and other_data.ndim == 2
            ):
                other_data = other_data[None]
            result_data = self.data @ other_data
        result = NTKStats(result_data)
        result._df_index = self._df_index
        result._df_columns = other_cols
        return result

    @_checks_ntkstats_compat
    def __rmatmul__(self, other) -> NTKStats:
        # treat `other` as the left operand: other @ self
        other_index, _ = self._get_index(other)

        other_data = (
            other.values if isinstance(other, pd.DataFrame) else self._other_data(other)
        )

        if self.data.ndim == 2:
            # Vector per replica (Nrep, n): treat as (Nrep, n, 1), multiply, then squeeze.
            if (
                not isinstance(other, NTKStats)
                and isinstance(other_data, np.ndarray)
                and other_data.ndim == 2
            ):
                other_data = other_data[None]
            result_data = (other_data @ self.data[:, :, None]).squeeze(-1)
        else:
            # Plain 2D matrix (no replica dim): add batch dim so numpy broadcasts over replicas.
            if (
                not isinstance(other, NTKStats)
                and isinstance(other_data, np.ndarray)
                and other_data.ndim == 2
            ):
                other_data = other_data[None]
            result_data = other_data @ self.data

        result = NTKStats(result_data)
        result._df_index = other_index
        result._df_columns = self._df_columns
        return result

    def _assert_eigenvalues(self):
        if self.data.ndim != 2:
            raise ValueError(
                f"Operation requires 1D observables per replica (shape (Nrep, n)), "
                f"got {self.data.shape}"
            )

    def _as_diag_matrices(self, vals: np.ndarray) -> NTKStats:
        """Build (Nrep, n, n) diagonal matrices from (Nrep, n) values."""
        n = vals.shape[1]
        return NTKStats(vals[:, :, None] * np.eye(n))

    def as_diag(self) -> NTKStats:
        """Convert (Nrep, n) eigenvalues to (Nrep, n, n) diagonal matrices."""
        self._assert_eigenvalues()
        return self._as_diag_matrices(self.data)

    def exp_kernel(self, t: float) -> NTKStats:
        """
        Compute ``diag(1 - exp(-t * λ))`` for each replica.

        Parameters
        ----------
        t : float
            Time parameter controlling the decay rate.

        Returns
        -------
        NTKStats
            Shape ``(Nrep, n, n)`` — diagonal matrices per replica.
        """
        self._assert_eigenvalues()
        return self._as_diag_matrices(1.0 - np.exp(-t * self.data))

    def exp_kernel_decay(self, t: float) -> NTKStats:
        """
        Compute ``diag(exp(-t * λ))`` for each replica.

        Parameters
        ----------
        t : float
            Time parameter controlling the decay rate.

        Returns
        -------
        NTKStats
            Shape ``(Nrep, n, n)`` — diagonal matrices per replica.
        """
        self._assert_eigenvalues()
        return self._as_diag_matrices(np.exp(-t * self.data))

    def reshape(self, new_shape) -> NTKStats:
        """Return a new NTKStats with data reshaped to new_shape.

        If the reshape is from data with ndim = 1 to ndim = 2 (vector to
        matrix), the original index (if any) is split into equal parts and
        assigned to rows and columns of the new shape. For other reshapes, the
        original index is discarded and set to None, since it may not be
        meaningful after reshaping.
        """
        if (self.ndim == 2 and self.shape[-1] == 1) and len(new_shape) == 2:
            # Reshaping from vector to matrix: split index if possible
            if self._df_index is not None:
                n_rows, n_cols = new_shape
                if len(self._df_index) != n_rows * n_cols:
                    raise ValueError(
                        f"Cannot reshape with index: original length {len(self._df_index)} "
                        f"does not match new shape {new_shape}"
                    )
                row_index = self._df_index.droplevel(-1)[::n_cols]
                col_index = self._df_index.droplevel(0)[:n_cols]
            else:
                row_index = None
                col_index = None
        else:
            # For other reshapes, discard index since it may not be meaningful
            row_index = None
            col_index = None

        result = NTKStats(self.data.reshape((self.nreplica, *new_shape)))
        result._df_index = row_index
        result._df_columns = col_index
        return result


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


def generate_filename(replica_idx: int, name: str = None) -> str:
    """
    Generate a filename for saving NTK eigenvalues based on replica index and an optional name.

    Parameters
    ----------
    replica_idx : int
        Index of the replica
    name : str, optional
        Optional name to include in the filename for clarity

    Returns
    -------
    str
        Generated filename string
    """
    if name is None:
        return f"{NTK_EIGVAL_TOKEN}_{replica_idx}.npz"
    else:
        return f"{NTK_EIGVAL_TOKEN}_{name}_{replica_idx}.npz"


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


def compute_ntk(pdf_model, params, **kwargs):
    """
    Compute the NTK matrix given model parameters.

    Parameters
    ----------
    pdf_model : PDFModel
        The PDF model instance
    params : dict
        Model parameters
    **kwargs
        Additional arguments for the pdf_model.grid_values_func (e.g., exclude_layers
        for the n3fit model)

    Returns
    -------
    ntk : jnp.ndarray
        The NTK matrix
    ntk_shape : tuple
        Shape of the NTK matrix
    """
    pdf_func = pdf_model.grid_values_func(XGRID, **kwargs)
    jacobian_func = jax.jacfwd(pdf_func)
    jacobian = jacobian_func(params)

    # Compute NTK (nf,ng,nf,ng) -> assumes shape from jacobian
    ntk = jnp.einsum("ijk,lmk->ijlm", jacobian, jacobian)

    # Flatten to (nflavors * n_xgrid) × (nflavors * n_xgrid)
    d1, d2, d3, d4 = ntk.shape  # d1=nf, d2=ng, d3=nf, d4=ng
    ntk = ntk.reshape(d1 * d2, d3 * d4, order=NTK_ORDERING)

    # Materialise the NTK to host (NumPy), then drop JAX's compilation cache. Each call
    # rebuilds the model -> a fresh jacfwd -> a new XLA program with this epoch's weights
    # baked in; without clearing, those compiled programs accumulate (~0.3 GB/call). This
    # matters because the report recomputes the *same* snapshot epochs many times (the
    # eigenvector_grid is resolved per presentation leaf, and the bounded functools cache
    # thrashes when those epochs interleave), so the per-call leak compounds -> OOM.
    ntk = np.asarray(ntk)
    jax.clear_caches()

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
    fit_name: str,
    replicas_path: Path,
    replica_idx: int,
    max_epoch=None,
    name: str = None,
    pending_epochs: list = None,
    **kwargs,
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
    name : str, optional
        Optional name to include in the filename for clarity when saving results.

    Returns
    -------
    tuple or None
        (replica_idx, epochs, ntk_shape) on success, None on failure
    """
    try:
        # Create fresh pdf_model to avoid JAX tracer leaks
        pdf_model = get_pdf_model(fit_name, replica_idx=replica_idx)
        param_files = get_parameters_all_epochs(replicas_path, replica_idx)

        eigenvalues_list = []
        epochs = []
        ntk_shape = None

        if pending_epochs is not None:
            # Filter param_files to only include pending epochs
            log.info(
                f"Replica {replica_idx}: computing eigenvalues for pending epochs {pending_epochs}"
            )
            param_files = {
                epoch: param_file
                for epoch, param_file in param_files.items()
                if epoch in pending_epochs
            }

        for epoch, param_file in param_files.items():
            if max_epoch is not None and epoch > max_epoch:
                continue
            params = jnp.load(param_file)["params"]

            ntk, shape = compute_ntk(pdf_model, params, **kwargs)
            if ntk_shape is None:
                ntk_shape = shape

            eigvals, _ = compute_eigendecomposition(ntk, hermitian=True)
            eigenvalues_list.append(eigvals)
            epochs.append(epoch)

        # pending_epochs means that we are only computing a subset of epochs for a replica
        # that may already have some epochs computed. In that case, we want to append the
        # new eigenvalues to the existing ones and save the combined result.
        if pending_epochs is not None:
            try:
                existing_data = load_replica_eigenvalues(
                    replica_idx, replicas_path, name
                )
                existing_epochs = existing_data["epochs"]
                existing_eigenvalues = existing_data["eigenvalues"]

                # Combine existing and new eigenvalues
                combined_epochs = existing_epochs + epochs
                combined_eigenvalues = np.vstack(
                    [existing_eigenvalues, eigenvalues_list]
                )

                # Sort by epoch
                sorted_indices = np.argsort(combined_epochs)
                epochs = [combined_epochs[i] for i in sorted_indices]
                eigenvalues_list = [combined_eigenvalues[i] for i in sorted_indices]

            except FileNotFoundError:
                log.info(
                    f"No existing data found for replica {replica_idx}, saving new data."
                )

        # Stack eigenvalues: (n_epochs, n_eigenvalues)
        eigenvalues = np.stack(eigenvalues_list, axis=0)

        # Save immediately to disk
        save_replica_eigenvalues(
            eigenvalues=eigenvalues,
            epochs=epochs,
            replica_idx=replica_idx,
            replicas_path=replicas_path,
            ntk_shape=ntk_shape,
            name=name,
        )

        return (replica_idx, epochs, ntk_shape)

    except FileNotFoundError as e:
        log.warning(f"Skipping replica {replica_idx}: {e}")
        return None


def get_completed_replicas(replicas_path: Path, name: str = None) -> list:
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
            filename = generate_filename(idx, name)
            replica_file = replica_folder / f"{filename}"
            if replica_file.exists():
                completed.append(idx)
        except (ValueError, IndexError) as e:
            log.debug(f"Skipping folder {replica_folder.name}: {e}")
            continue

    return sorted(completed)


def get_completed_epochs_for_replica(replicas_path: Path, name: str = None) -> dict:
    """
    Get a dictionary mapping replica indices to lists of completed epochs.

    Parameters
    ----------
    replicas_path : Path
        Directory containing replica folders.
    name : str, optional
        Optional name to include in the filename to specify the set of eigenvalues.

    Returns
    -------
    dict
        Dictionary mapping replica index -> list of completed epochs
    """
    completed_epochs = {}
    for replica_folder in replicas_path.glob("replica_*"):
        try:
            idx = int(replica_folder.stem.split("_")[1])
            filename = generate_filename(idx, name)
            replica_file = replica_folder / f"{filename}"
            if replica_file.exists():
                data = np.load(replica_file)
                completed_epochs[idx] = data["epochs"].tolist()
        except (ValueError, IndexError) as e:
            log.debug(f"Skipping folder {replica_folder.name}: {e}")
            continue

    return completed_epochs


def save_replica_eigenvalues(
    eigenvalues: np.ndarray,
    epochs: list,
    replica_idx: int,
    replicas_path: Path,
    ntk_shape: tuple = None,
    name: str = None,
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
    name: str, optional
        Optional name to include in the filename for clarity
    """
    filename = generate_filename(replica_idx, name)
    replica_file = replicas_path / f"replica_{replica_idx}/{filename}"
    np.savez_compressed(
        replica_file,
        eigenvalues=eigenvalues,
        epochs=np.array(epochs),
        ntk_shape=ntk_shape,
    )
    log.debug(f"Saved eigenvalues for replica {replica_idx} to {replica_file}")


def load_replica_eigenvalues(
    replica_idx: int, cache_dir: Path, name: str = None
) -> dict:
    """
    Load eigenvalues for a single replica from disk.

    Parameters
    ----------
    replica_idx : int
        Replica index
    cache_dir : Path
        Directory containing saved results
    name: str, optional
        Optional name to include in the filename to specify the set of eigenvalues.

    Returns
    -------
    dict
        Dictionary with 'eigenvalues' (n_epochs, n_eigenvalues) and 'epochs'
    """
    filename = generate_filename(replica_idx, name)
    replica_file = cache_dir / f"replica_{replica_idx}/{filename}"

    if not replica_file.exists():
        raise FileNotFoundError(f"Replica {replica_idx} not found at {replica_file}")

    data = np.load(replica_file)
    return {
        "eigenvalues": data["eigenvalues"],
        "epochs": data["epochs"].tolist(),
        "ntk_shape": data["ntk_shape"],
    }


def load_eigenvalues_ensemble(
    replicas_path: Path,
    max_epoch=None,
    name: str = None,
    replica_index_list: list = None,
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
    name: str, optional
        Optional name to include in the filename to specify the set of eigenvalues.
    replica_index_list : list, optional
        Optional list of replica indices to include. If None, includes all completed replicas.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'eigenvalues_by_epoch': dict mapping epoch -> ndarray (n_replicas, n_eigenvalues)
        - 'epochs': list of epochs
        - 'ntk_shape': shape of NTK matrix
        - 'replica_indices': list of replica indices included
    """
    completed_replicas = get_completed_replicas(replicas_path, name)

    if not completed_replicas:
        raise ValueError(f"No completed replicas found in {replicas_path}")

    if replica_index_list is not None:
        completed_replicas = [
            idx for idx in completed_replicas if idx in replica_index_list
        ]
        if not completed_replicas:
            raise ValueError(
                f"No completed replicas found in {replicas_path} matching indices {replica_index_list}"
            )

    # Load all replicas
    all_eigenvalues = []
    included_replicas = []
    ntk_shape = None
    for replica_idx in completed_replicas:
        data = load_replica_eigenvalues(replica_idx, replicas_path, name)
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


def compute_eigenvectors_at_epoch_for_replica(
    fit_name: str, replicas_path: Path, replica_idx: int, epoch: int, **kwargs
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
    **kwargs
        Additional arguments for the pdf_model.grid_values_func (e.g., exclude_layers
        for the n3fit model)
    """
    try:
        pdf_model = get_pdf_model(fit_name, replica_idx=replica_idx)
        param_files = get_parameters_all_epochs(replicas_path, replica_idx)

        params_at_epoch = param_files.get(epoch, None)
        if params_at_epoch is None:
            raise ValueError(
                f"Epoch {epoch} not found for replica {replica_idx} in fit {fit_name}"
            )

        params = jnp.load(params_at_epoch)["params"]
        ntk, shape = compute_ntk(pdf_model, params, **kwargs)
        _, eigvecs = compute_eigendecomposition(ntk, hermitian=True)

        return (replica_idx, epoch, eigvecs, shape)
    except Exception as e:
        log.warning(f"Skipping replica {replica_idx}: {e}")
        return None
