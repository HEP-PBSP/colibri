"""
super_net.covmats.py

Module containing covariance matrices functions.

Author: Mark N. Costantini 
Notes: Several functions are taken from validphys.covmats
Date: 11.11.2023
"""

import jax.numpy as jnp
import jax.scipy.linalg as jla

import numpy as np

from validphys import covmats


def sqrt_covmat_jax(covariance_matrix):
    """
    Same as `validphys.covmats.sqrt_covmat` but
    for jax.numpy arrays

    Parameters
    ----------
    covariance_matrix : jnp.ndarray
        A positive definite covariance matrix, which is N_dat x N_dat (where
        N_dat is the number of data points after cuts) containing uncertainty
        and correlation information.

    Returns
    -------
    sqrt_mat : jnp.ndarray
        The square root of the input covariance matrix, which is N_dat x N_dat
        (where N_dat is the number of data points after cuts), and which is the
        the lower triangular decomposition. The following should be ``True``:
        ``jnp.allclose(sqrt_covmat @ sqrt_covmat.T, covariance_matrix)``.
    """

    dimensions = covariance_matrix.shape

    if covariance_matrix.size == 0:
        raise ValueError("Attempting the decomposition of an empty matrix.")
    elif dimensions[0] != dimensions[1]:
        raise ValueError(
            "The input covariance matrix should be square but "
            f"instead it has dimensions {dimensions[0]} x "
            f"{dimensions[1]}"
        )

    sqrt_diags = jnp.sqrt(jnp.diag(covariance_matrix))
    correlation_matrix = covariance_matrix / sqrt_diags[:, jnp.newaxis] / sqrt_diags
    decomp = jla.cholesky(correlation_matrix)
    sqrt_matrix = (decomp * sqrt_diags).T
    return sqrt_matrix


def dataset_inputs_covmat_from_systematics(
    data,
    experimental_commondata_tuple,
):
    """
    Similar to validphys.covmats.dataset_inputs_covmat_from_systematics
    but jax.numpy array.

    Note: see production rule in `config.py` for commondata_tuple options.
    """

    covmat = jnp.array(
        covmats.dataset_inputs_covmat_from_systematics(
            experimental_commondata_tuple,
            data.dsinputs,
            use_weights_in_covmat=False,
            norm_threshold=None,
            _list_of_central_values=None,
            _only_additive=False,
        )
    )
    return covmat


def super_net_dataset_inputs_t0_predictions(_pred_t0data, t0_pdf_grid):
    """
    Similar to validphys.covmats.dataset_inputs_t0_predictions.

    Parameters
    ----------
    _pred_t0data: jax.jit compiled function
        function taking a pdf grid and returning
        theory prediction for one data group

    t0_pdf_grid: jnp.array

    Returns
    -------
    t0predictions: list
        list of theory predictions for each dataset
    """
    # central PDF member for t0 predictions
    pred = _pred_t0data(t0_pdf_grid[0])
    t0predictions = [np.array(pred[i]) for i in range(len(pred))]

    return t0predictions


def dataset_inputs_t0_covmat_from_systematics(
    data, experimental_commondata_tuple, super_net_dataset_inputs_t0_predictions
):
    """
    Similar as `validphys.covmats.dataset_inputs_t0_covmat_from_systematics`
    but jax.numpy array.

    Note: see production rule in `config.py` for commondata_tuple options.
    """

    covmat = jnp.array(
        covmats.dataset_inputs_t0_covmat_from_systematics(
            experimental_commondata_tuple,
            data_input=data.dsinputs,
            use_weights_in_covmat=False,
            norm_threshold=None,
            dataset_inputs_t0_predictions=super_net_dataset_inputs_t0_predictions,
        )
    )
    return covmat
