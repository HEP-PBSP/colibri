"""
colibri.mc_loss_functions.py

This module provides the functions necessary for the computation of the chi2 for a MC fit.


Date: 17.01.2024
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla

from reportengine import collect
from colibri.covmats import sqrt_covmat_jax


def make_chi2_training_data(mc_pseudodata, fit_covariance_matrix, _pred_data):
    """
    Returns a jax.jit compiled function that computes the chi2
    of a pdf grid on a training data batch.

    Notes:
        - Does not include positivity constraint.
        - This function is designed for Monte Carlo like PDF fits.

    Parameters
    ----------
    mc_pseudodata: mc_utils.MCPseudodata
        dataclass containing Monte Carlo pseudodata.

    fit_covariance_matrix: jnp.array
        covariance matrix of the fit (see config.produce_fit_covariance_matrix).

    _pred_data: theory_predictions._pred_data
        colibri provider for (fktable) theory predictions.

    Returns
    -------
    @jax.jit Callable
        function to compute chi2 of a pdf grid on a data batch.

    """
    tr_idx = mc_pseudodata.training_indices
    central_values = mc_pseudodata.pseudodata[tr_idx]
    covmat = fit_covariance_matrix[tr_idx][:, tr_idx]

    @jax.jit
    def chi2(pdf, batch_idx):
        """
        Parameters
        ----------
        pdf: jnp.array
            pdf grid.

        batch_idx: jnp.array
            array of data batch indices.

        Returns
        -------
        float
            loss function value

        """
        diff = _pred_data(pdf)[tr_idx][batch_idx] - central_values[batch_idx]

        # batch covariance matrix before decomposing it
        batched_covmat = covmat[batch_idx][:, batch_idx]
        # decompose covmat after having batched it!
        sqrt_covmat = jnp.array(sqrt_covmat_jax(batched_covmat))

        # solve_triangular: solve the equation a x = b for x, assuming a is a triangular matrix.
        chi2_vec = jla.solve_triangular(sqrt_covmat, diff, lower=True)
        loss = jnp.sum(chi2_vec**2)
        return loss

    return chi2


mc_replicas_make_chi2_training_data = collect(
    "make_chi2_training_data", ("trval_replica_indices",)
)


def make_chi2_training_data_with_positivity(
    mc_pseudodata, mc_posdata_split, fit_covariance_matrix, _pred_data, _penalty_posdata
):
    """
    Returns a jax.jit compiled function that computes the chi2
    of a pdf grid on a training data batch including positivity penalty.

    Notes:
        - This function is designed for Monte Carlo like PDF fits.

    Parameters
    ----------
    mc_pseudodata: mc_utils.MCPseudodata
        dataclass containing Monte Carlo pseudodata.

    fit_covariance_matrix: jnp.array
        covariance matrix of the fit (see config.produce_fit_covariance_matrix).

    _pred_data: theory_predictions._pred_data
        colibri provider for (fktable) theory predictions.

    _posdata_split: training_validation.PosdataTrainValidationSplit
        dataclass inheriting from utils.TrainValidationSplit

    _penalty_posdata: theory_predictions._penalty_posdata
        colibri provider used to compute positivity penalty.

    Returns
    -------
    @jax.jit Callable
        function to compute chi2 of a pdf grid on a data batch.
    """

    tr_idx = mc_pseudodata.training_indices
    central_values = mc_pseudodata.pseudodata[tr_idx]
    covmat = fit_covariance_matrix[tr_idx][:, tr_idx]

    posdata_training_idx = mc_posdata_split.training

    @jax.jit
    def chi2(pdf, batch_idx, alpha, lambda_positivity):
        """
        Parameters
        ----------
        pdf: jnp.array
            pdf grid.

        batch_idx: jnp.array
            array of data batch indices.

        alpha: float

        lambda_positivity: float

        Returns
        -------
        float
            loss function value

        """
        diff = _pred_data(pdf)[tr_idx][batch_idx] - central_values[batch_idx]

        # batch covariance matrix before decomposing it
        batched_covmat = covmat[batch_idx][:, batch_idx]
        # decompose covmat after having batched it!
        sqrt_covmat = jnp.array(sqrt_covmat_jax(batched_covmat))

        # solve_triangular: solve the equation a x = b for x, assuming a is a triangular matrix.
        chi2_vec = jla.solve_triangular(sqrt_covmat, diff, lower=True)
        loss = jnp.sum(chi2_vec**2)

        # add penalty term due to positivity
        pos_penalty = _penalty_posdata(pdf, alpha, lambda_positivity)[
            posdata_training_idx
        ]
        loss += jnp.sum(pos_penalty)

        return loss

    return chi2


mc_replicas_make_chi2_training_data_with_positivity = collect(
    "make_chi2_training_data_with_positivity", ("trval_replica_indices",)
)


def make_chi2_validation_data(mc_pseudodata, fit_covariance_matrix, _pred_data):
    """
    Returns a jax.jit compiled function that computes the chi2
    of a pdf grid on validation data.

    Notes:
        - Does not include positivity constraint.
        - This function is designed for Monte Carlo like PDF fits.

    Parameters
    ----------
    mc_pseudodata: mc_utils.MCPseudodata
        dataclass containing Monte Carlo pseudodata.

    fit_covariance_matrix: jnp.array
        covariance matrix of the fit (see config.produce_fit_covariance_matrix).

    _pred_data: theory_predictions._pred_data
        colibri provider for (fktable) theory predictions.

    Returns
    -------
    @jax.jit Callable
        function to compute chi2 of a pdf grid on validation data.
    """
    val_idx = mc_pseudodata.validation_indices
    central_values = mc_pseudodata.pseudodata[val_idx]
    covmat = fit_covariance_matrix[val_idx][:, val_idx]

    # decompose covmat
    sqrt_covmat = jnp.array(sqrt_covmat_jax(covmat))

    @jax.jit
    def chi2(pdf):
        """ """
        diff = _pred_data(pdf)[val_idx] - central_values

        # solve_triangular: solve the equation a x = b for x, assuming a is a triangular matrix.
        chi2_vec = jla.solve_triangular(sqrt_covmat, diff, lower=True)
        loss = jnp.sum(chi2_vec**2)
        return loss

    return chi2


mc_replicas_make_chi2_validation_data = collect(
    "make_chi2_validation_data", ("trval_replica_indices",)
)


def make_chi2_validation_data_with_positivity(
    mc_pseudodata, mc_posdata_split, fit_covariance_matrix, _pred_data, _penalty_posdata
):
    """
    Returns a jax.jit compiled function that computes the chi2
    of a pdf grid on validation data.

    Notes:
        - This function is designed for Monte Carlo like PDF fits.

    Parameters
    ----------
    mc_pseudodata: mc_utils.MCPseudodata
        dataclass containing Monte Carlo pseudodata.

    fit_covariance_matrix: jnp.array
        covariance matrix of the fit (see config.produce_fit_covariance_matrix).

    _pred_data: theory_predictions._pred_data
        colibri provider for (fktable) theory predictions.

    _posdata_split: training_validation.PosdataTrainValidationSplit
        dataclass inheriting from utils.TrainValidationSplit

    _penalty_posdata: theory_predictions._penalty_posdata
        colibri provider used to compute positivity penalty.

    Returns
    -------
    @jax.jit Callable
        function to compute chi2 of a pdf grid on validation data.
    """
    if not mc_pseudodata.trval_split:
        return lambda pdf, alpha, lambda_positivity: jnp.nan

    val_idx = mc_pseudodata.validation_indices
    central_values = mc_pseudodata.pseudodata[val_idx]
    covmat = fit_covariance_matrix[val_idx][:, val_idx]

    posdata_validation_idx = mc_posdata_split.validation
    # decompose covmat
    sqrt_covmat = jnp.array(sqrt_covmat_jax(covmat))

    @jax.jit
    def chi2(pdf, alpha, lambda_positivity):
        """ """
        diff = _pred_data(pdf)[val_idx] - central_values

        # solve_triangular: solve the equation a x = b for x, assuming a is a triangular matrix.
        chi2_vec = jla.solve_triangular(sqrt_covmat, diff, lower=True)
        loss = jnp.sum(chi2_vec**2)

        # add penalty term due to positivity
        pos_penalty = _penalty_posdata(pdf, alpha, lambda_positivity)[
            posdata_validation_idx
        ]
        loss += jnp.sum(pos_penalty)

        return loss

    return chi2


mc_replicas_make_chi2_validation_data_with_positivity = collect(
    "make_chi2_validation_data_with_positivity", ("trval_replica_indices",)
)
