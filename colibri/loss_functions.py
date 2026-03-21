"""
colibri.loss_functions.py

This module provides the functions necessary for the computation of the chi2.
"""

import jax.numpy as jnp
import jax.lax.linalg as jlinalg


def chi2(central_values, predictions, sqrt_covmat):
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

    # whiten the diff
    z = jlinalg.triangular_solve(sqrt_covmat, diff, left_side=True, lower=True)

    loss = jnp.dot(z, z)
    return loss
