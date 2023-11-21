"""
wmin.wmin_loss_functions.py

This module provides the functions necessary for the computation of the chi2 in the wmin parameterisation.

Author: L. Mantani
Date: 21.11.2023
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla

from super_net.covmats import sqrt_covmat_jax


def make_chi2_wmin_opt(make_data_values, make_pred_data, weight_minimization_grid):
    """
    Returns a jax.jit compiled function that computes the chi2
    of a pdf grid optimised for weight minimisation.

    Notes:
        - Does not include positivity constraint.
        - This function is designed for Bayesian like PDF fits.
        - Works only for DIS datasets.

    Parameters
    ----------
    make_data_values: training_validation.MakeDataValues
        dataclass containing data for training and validation.

    make_pred_data: theory_predictions.make_pred_data
        super_net provider for (fktable) theory predictions.

    Returns
    -------
    @jax.jit Callable
        function to compute chi2 of a pdf grid.

    """
    training_data = make_data_values.training_data
    central_values = training_data.central_values
    covmat = training_data.covmat
    central_values_idx = training_data.central_values_idx

    # Precompute predictions for the basis of wmin
    predictions = jnp.array(
        [
            make_pred_data(weight_minimization_grid.wmin_INPUT_GRID[i])
            for i in range(len(weight_minimization_grid.wmin_INPUT_GRID))
        ]
    )

    # Invert the covmat
    inv_covmat = jla.inv(covmat)

    @jax.jit
    def chi2(wmin_weights):
        """ """
        theory = jnp.einsum("i,ij", wmin_weights, predictions)

        diff = theory[central_values_idx] - central_values

        loss = jnp.einsum("i,ij,j", diff, inv_covmat, diff)

        return loss

    return chi2
