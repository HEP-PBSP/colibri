"""
colibri.loss_functions.py

This module provides the functions necessary for the computation of the chi2.
"""

import jax.numpy as jnp


def chi2(central_values, predictions, inv_covmat):
    """
    Compute the chi2 loss.

    Parameters
    ----------
    central_values: jnp.ndarray
        The central values of the data.

    predictions: jnp.ndarray
        The predictions of the model.

    inv_covmat: jnp.ndarray
        The inverse of the covariance matrix.

    Returns
    -------
    loss: jnp.ndarray
        The chi2 loss.
    """
    diff = predictions - central_values

    # TODO: if inv_covmat is diagonal (a 1-D array) then loss can be computed more efficiently
    if inv_covmat.ndim == 1:
        loss = jnp.sum(diff**2 * inv_covmat)
    else:
        loss = jnp.einsum("i,ij,j", diff, inv_covmat, diff)

    return loss
