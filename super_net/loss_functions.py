"""
super_net.loss_functions.py

This module provides the functions necessary for the computation of the chi2.

Author: Mark N. Costantini
Date: 11.11.2023
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla

from super_net.covmats import sqrt_covmat_jax

from reportengine import collect


def make_chi2_training_data(make_data_values, make_pred_data):
    """
    Returns a jax.jit compiled function that computes the chi2
    of a pdf grid on a training data batch.

    Notes:
        - Does not include positivity constraint.
        - This function is designed for Monte Carlo like PDF fits.

    Parameters
    ----------
    make_data_values: training_validation.MakeDataValues
        dataclass containing data for training and validation.

    make_pred_data: theory_predictions.make_pred_data
        super_net provider for (fktable) theory predictions.

    Returns
    -------
    @jax.jit Callable
        function to compute chi2 of a pdf grid on a data batch.

    """
    training_data = make_data_values.training_data
    central_values = training_data.central_values
    covmat = training_data.covmat
    central_values_idx = training_data.central_values_idx

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
        diff = (
            make_pred_data(pdf)[central_values_idx][batch_idx]
            - central_values[batch_idx]
        )

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
    make_data_values, make_pred_data, make_posdata_split, make_penalty_posdata
):
    """
    Returns a jax.jit compiled function that computes the chi2
    of a pdf grid on a training data batch including positivity penalty.

    Notes:
        - This function is designed for Monte Carlo like PDF fits.

    Parameters
    ----------
    make_data_values: training_validation.MakeDataValues
        dataclass containing data for training and validation.

    make_pred_data: theory_predictions.make_pred_data
        super_net provider for (fktable) theory predictions.

    make_posdata_split: training_validation.PosdataTrainValidationSplit
        dataclass inheriting from monte_carlo_utils.TrainValidationSplit

    make_penalty_posdata: theory_predictions.make_penalty_posdata
        super_net provider used to compute positivity penalty.

    Returns
    -------
    @jax.jit Callable
        function to compute chi2 of a pdf grid on a data batch.
    """
    training_data = make_data_values.training_data
    central_values = training_data.central_values
    covmat = training_data.covmat
    central_values_idx = training_data.central_values_idx

    posdata_training_idx = make_posdata_split.training

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
        diff = (
            make_pred_data(pdf)[central_values_idx][batch_idx]
            - central_values[batch_idx]
        )

        # batch covariance matrix before decomposing it
        batched_covmat = covmat[batch_idx][:, batch_idx]
        # decompose covmat after having batched it!
        sqrt_covmat = jnp.array(sqrt_covmat_jax(batched_covmat))

        # solve_triangular: solve the equation a x = b for x, assuming a is a triangular matrix.
        chi2_vec = jla.solve_triangular(sqrt_covmat, diff, lower=True)
        loss = jnp.sum(chi2_vec**2)

        # add penalty term due to positivity
        pos_penalty = make_penalty_posdata(pdf, alpha, lambda_positivity)[
            posdata_training_idx
        ]
        loss += jnp.sum(pos_penalty)

        return loss

    return chi2


mc_replicas_make_chi2_training_data_with_positivity = collect(
    "make_chi2_training_data_with_positivity", ("trval_replica_indices",)
)


def make_chi2_validation_data(make_data_values, make_pred_data):
    """
    Returns a jax.jit compiled function that computes the chi2
    of a pdf grid on validation data.

    Notes:
        - Does not include positivity constraint.
        - This function is designed for Monte Carlo like PDF fits.

    Parameters
    ----------
    make_data_values: training_validation.MakeDataValues
        dataclass containing data for training and validation.

    make_pred_data: theory_predictions.make_pred_data
        super_net provider for (fktable) theory predictions.

    Returns
    -------
    @jax.jit Callable
        function to compute chi2 of a pdf grid on validation data.
    """
    validation_data = make_data_values.validation_data
    central_values = validation_data.central_values
    covmat = validation_data.covmat
    central_values_idx = validation_data.central_values_idx

    @jax.jit
    def chi2(pdf):
        """ """
        diff = make_pred_data(pdf)[central_values_idx] - central_values

        # decompose covmat
        sqrt_covmat = jnp.array(sqrt_covmat_jax(covmat))

        # solve_triangular: solve the equation a x = b for x, assuming a is a triangular matrix.
        chi2_vec = jla.solve_triangular(sqrt_covmat, diff, lower=True)
        loss = jnp.sum(chi2_vec**2)
        return loss

    return chi2


mc_replicas_make_chi2_validation_data = collect(
    "make_chi2_validation_data", ("trval_replica_indices",)
)


def make_chi2_validation_data_with_positivity(
    make_data_values, make_pred_data, make_posdata_split, make_penalty_posdata
):
    """
    Returns a jax.jit compiled function that computes the chi2
    of a pdf grid on validation data.

    Notes:
        - This function is designed for Monte Carlo like PDF fits.

    Parameters
    ----------
    make_data_values: training_validation.MakeDataValues
        dataclass containing data for training and validation.

    make_pred_data: theory_predictions.make_pred_data
        super_net provider for (fktable) theory predictions.

    make_posdata_split: training_validation.PosdataTrainValidationSplit
        dataclass inheriting from monte_carlo_utils.TrainValidationSplit

    make_penalty_posdata: theory_predictions.make_penalty_posdata
        super_net provider used to compute positivity penalty.

    Returns
    -------
    @jax.jit Callable
        function to compute chi2 of a pdf grid on validation data.
    """
    validation_data = make_data_values.validation_data
    central_values = validation_data.central_values
    covmat = validation_data.covmat
    central_values_idx = validation_data.central_values_idx

    posdata_validation_idx = make_posdata_split.validation

    @jax.jit
    def chi2(pdf, alpha, lambda_positivity):
        """ """
        diff = make_pred_data(pdf)[central_values_idx] - central_values

        # decompose covmat
        sqrt_covmat = jnp.array(sqrt_covmat_jax(covmat))

        # solve_triangular: solve the equation a x = b for x, assuming a is a triangular matrix.
        chi2_vec = jla.solve_triangular(sqrt_covmat, diff, lower=True)
        loss = jnp.sum(chi2_vec**2)

        # add penalty term due to positivity
        pos_penalty = make_penalty_posdata(pdf, alpha, lambda_positivity)[
            posdata_validation_idx
        ]
        loss += jnp.sum(pos_penalty)

        return loss

    return chi2


mc_replicas_make_chi2_validation_data_with_positivity = collect(
    "make_chi2_validation_data_with_positivity", ("trval_replica_indices",)
)


def make_chi2(make_data_values, make_pred_data):
    """
    Returns a jax.jit compiled function that computes the chi2
    of a pdf grid on a dataset.

    Notes:
        - Does not include positivity constraint.
        - This function is designed for Bayesian like PDF fits.

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

    # Invert the covmat
    inv_covmat = jla.inv(covmat)

    @jax.jit
    def chi2(pdf):
        """ """
        diff = make_pred_data(pdf)[central_values_idx] - central_values

        loss = jnp.einsum("i,ij,j", diff, inv_covmat, diff)

        return loss

    return chi2


def make_chi2_with_positivity(
    make_data_values,
    make_pred_data,
    make_posdata_split,
    make_penalty_posdata,
    alpha=1e-7,
    lambda_positivity=1000,
):
    """
    Returns a jax.jit compiled function that computes the chi2
    of a pdf grid on a dataset.

    Notes:
        - This function is designed for Bayesian like PDF fits.

    Parameters
    ----------
    make_data_values: training_validation.MakeDataValues
        dataclass containing data for training and validation.

    make_pred_data: theory_predictions.make_pred_data
        super_net provider for (fktable) theory predictions.

    make_posdata_split: training_validation.PosdataTrainValidationSplit
        dataclass inheriting from monte_carlo_utils.TrainValidationSplit

    make_penalty_posdata: theory_predictions.make_penalty_posdata
        super_net provider used to compute positivity penalty.

    alpha: float

    lambda_positivity: float

    Returns
    -------
    @jax.jit Callable
        function to compute chi2 of a pdf grid.

    """
    training_data = make_data_values.training_data
    central_values = training_data.central_values
    covmat = training_data.covmat
    central_values_idx = training_data.central_values_idx

    # Invert the covmat
    inv_covmat = jla.inv(covmat)

    posdata_training_idx = make_posdata_split.training

    @jax.jit
    def chi2(pdf):
        """ """
        diff = make_pred_data(pdf)[central_values_idx] - central_values

        loss = jnp.einsum("i,ij,j", diff, inv_covmat, diff)

        # add penalty term due to positivity
        pos_penalty = make_penalty_posdata(pdf, alpha, lambda_positivity)[
            posdata_training_idx
        ]
        loss += jnp.sum(pos_penalty)

        return loss

    return chi2
