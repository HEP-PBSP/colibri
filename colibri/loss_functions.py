"""
colibri.loss_functions.py

This module provides the functions necessary for the computation of the chi2.
"""

import jax.numpy as jnp


def chi2(whitened_data, predictions, inv_sqrt_covmat):
    """
    Compute the chi2 loss in the whitened basis.

    The data are assumed to be pre-whitened (``whitened_data = L^{-1} d``).
    The predictions are whitened on-the-fly via ``inv_sqrt_covmat``:

        chi2 = ||whitened_data - inv_sqrt_covmat @ predictions||^2

    Parameters
    ----------
    whitened_data: jnp.ndarray
        Pre-whitened central values: ``L^{-1} d``.

    predictions: jnp.ndarray
        Model predictions in the original (unwhitened) basis.

    inv_sqrt_covmat: jnp.ndarray
        Inverse Cholesky factor ``L^{-1}``, shape ``(n_selected, n_data)``.
        Applies the whitening transform to predictions and selects the
        relevant data subset simultaneously.

    Returns
    -------
    loss: jnp.ndarray
        The chi2 loss (scalar).
    """
    diff = whitened_data - inv_sqrt_covmat @ predictions
    return jnp.dot(diff, diff)
