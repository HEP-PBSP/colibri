"""
super_net.covmats.py

Module containing covariance matrices functions.

Author: Mark N. Costantini 
Notes: Several functions are taken from validphys.covmats
Date: 11.11.2023
"""

import jax.numpy as jnp
import jax.scipy.linalg as jla

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
    commondata_tuple,
):
    """ 
    Similar to validphys.covmats.dataset_inputs_covmat_from_systematics
    but jax.numpy array.

    Note: see production rule in `config.py` for commondata_tuple options.
    """

    covmat = jnp.array(
        covmats.dataset_inputs_covmat_from_systematics(
            commondata_tuple,
            data.dsinputs,
            use_weights_in_covmat=False,
            norm_threshold=None,
            _list_of_central_values=None,
            _only_additive=False,
        )
    )
    return covmat


def dataset_inputs_t0_covmat_from_systematics(
    data,
    commondata_tuple,
    make_pred_data,
    t0pdfset
):
    """
    Similar as `validphys.covmats.dataset_inputs_t0_covmat_from_systematics`
    but jax.numpy array.

    Note: see production rule in `config.py` for commondata_tuple options.
    """
    covmat = jnp.array(
        covmats.dataset_inputs_t0_covmat_from_systematics(
            commondata_tuple,
            data_input=data.dsinputs,
            use_weights_in_covmat=False,
            norm_threshold=None,
            dataset_inputs_t0_predictions=make_pred_data(t0pdfset),
        )
    )
    return covmat
