"""
colibri.loss_functions.py

This module provides the functions necessary for the computation of the chi2.

Author: Mark N. Costantini
Date: 11.11.2023
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla


def make_chi2(_data_values, _pred_data, vectorized=False):
    """
    Returns a jax.jit compiled function that computes the chi2
    of a pdf grid on a dataset.

    Notes:
        - Does not include positivity constraint.
        - This function is designed for Bayesian like PDF fits.
        - allows for vectorized evaluation of the chi2.

    Parameters
    ----------
    _data_values: training_validation.MakeDataValues
        dataclass containing data for training and validation.

    _pred_data: theory_predictions._pred_data
        colibri provider for (fktable) theory predictions.

    vectorized: bool, default is False

    Returns
    -------
    @jax.jit Callable
        function to compute chi2 of a pdf grid.

    """
    training_data = _data_values.training_data
    central_values = training_data.central_values
    covmat = training_data.covmat
    central_values_idx = training_data.central_values_idx

    # Invert the covmat
    # We use this instead of Cholesky decomposition
    # since we do it only once and for all at the beginning
    inv_covmat = jla.inv(covmat)

    @jax.jit
    def chi2(pdf):
        """ """
        diff = _pred_data(pdf)[central_values_idx] - central_values

        loss = jnp.einsum("i,ij,j", diff, inv_covmat, diff)

        return loss

    if vectorized:
        # return jnp.vectorize(chi2, signature="(m,n)->()")
        return jax.vmap(chi2, in_axes=(0,), out_axes=0)
    return chi2


def make_chi2_with_positivity(
    _data_values,
    _pred_data,
    _posdata_split,
    _penalty_posdata,
    alpha=1e-7,
    lambda_positivity=1000,
    vectorized=False,
):
    """
    Returns a jax.jit compiled function that computes the chi2
    of a pdf grid on a dataset.

    Notes:
        - This function is designed for Bayesian like PDF fits.

    Parameters
    ----------
    _data_values: training_validation.MakeDataValues
        dataclass containing data for training and validation.

    _pred_data: theory_predictions._pred_data
        colibri provider for (fktable) theory predictions.

    _posdata_split: training_validation.PosdataTrainValidationSplit
        dataclass inheriting from utils.TrainValidationSplit

    _penalty_posdata: theory_predictions._penalty_posdata
        colibri provider used to compute positivity penalty.

    alpha: float

    lambda_positivity: float

    vectorized: bool, default is False

    Returns
    -------
    @jax.jit Callable
        function to compute chi2 of a pdf grid.

    """
    training_data = _data_values.training_data
    central_values = training_data.central_values
    covmat = training_data.covmat
    central_values_idx = training_data.central_values_idx

    # Invert the covmat
    inv_covmat = jla.inv(covmat)

    posdata_training_idx = _posdata_split.training

    @jax.jit
    def chi2(pdf):
        """ """
        diff = _pred_data(pdf)[central_values_idx] - central_values

        loss = jnp.einsum("i,ij,j", diff, inv_covmat, diff)

        # add penalty term due to positivity
        pos_penalty = _penalty_posdata(pdf, alpha, lambda_positivity)[
            posdata_training_idx
        ]

        loss += jnp.sum(pos_penalty)

        return loss

    if vectorized:
        # jnp.vectorize(chi2, signature="(m,n)->()")
        return jax.vmap(chi2, in_axes=(0,), out_axes=0)

    return chi2
