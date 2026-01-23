"""
colibri.ntkutils.py

Module containing several utils for the analysis of the NTK.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

from colibri.constants import XGRID


def compute_ntk(pdf_model, replicas_path, replica_index):
    """
    Compute NTK for a single replica across all stored epochs.

    Parameters
    ----------
    pdf_model : PDFModel
        The PDF model instance
    replicas_path : Path
        Path to the replicas directory
    replica_index : int
        Index of the replica to compute

    Returns
    -------
    ntk_by_epochs : list of jnp.ndarray
        List of NTK matrices, one per epoch
    epochs : list of int
        List of epoch numbers corresponding to the NTKs
    """
    pdf_func = pdf_model.grid_values_func(XGRID)
    jacobian_func = jax.jacfwd(pdf_func)

    params_folder = replicas_path / f"replica_{replica_index}/parameters"
    if not params_folder.exists() or not any(params_folder.glob("*.npz")):
        raise FileNotFoundError(
            f"Parameters folder {params_folder} does not exist or is empty."
        )

    param_files = list(params_folder.glob("*.npz"))
    param_files.sort(key=lambda f: int(f.stem.split("_")[-1]))

    ntk_by_epochs = []
    epochs = []

    for param_file in param_files:
        epoch = int(param_file.stem.split("_")[-1])

        params = jnp.load(param_file)["params"]
        jacobian = jacobian_func(params)

        # Compute NTK (14,50,14,50) -> assumes shape from jacobian
        ntk = jnp.einsum("ijk,lmk->ijlm", jacobian, jacobian)

        # Flatten to (N_grid*N_flavors)×(N_grid*N_flavors)
        d1, d2, d3, d4 = ntk.shape
        ntk = ntk.reshape(d1 * d2, d3 * d4)

        ntk_by_epochs.append(np.array(ntk))
        epochs.append(epoch)

    return ntk_by_epochs, epochs, ntk.shape


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
        eigenvectors, eigenvalues, _ = np.linalg.svd(ntk_matrix, hermitian=hermitian)

    return eigenvalues, eigenvectors


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
    int
        Number of replicas found
    """
    replicas_path = Path(replicas_path)
    if not replicas_path.exists():
        raise FileNotFoundError(f"Replicas path does not exist: {replicas_path}")

    # Count directories named "replica_*"
    replica_dirs = sorted(replicas_path.glob("replica_*"))
    rep_list = [int(d.name.split("_")[1]) for d in replica_dirs]
    return rep_list
