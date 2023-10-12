"""
TODO
"""
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla

from super_net.covmats import sqrt_covmat_jax

from reportengine import collect


def make_chi2_training_data(make_data_values, make_pred_data):
    """
    TODO
    """
    training_data = make_data_values.training_data
    central_values = training_data.central_values
    covmat = training_data.covmat
    central_values_idx = training_data.central_values_idx

    @jax.jit
    def chi2(pdf, batch_idx):
        """
        TODO
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
    TODO
    """
    training_data = make_data_values.training_data
    central_values = training_data.central_values
    covmat = training_data.covmat
    central_values_idx = training_data.central_values_idx

    posdata_training_idx = make_posdata_split.training

    @jax.jit
    def chi2(pdf, batch_idx, alpha, lambda_positivity):
        """
        TODO
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
    TODO
    """
    validation_data = make_data_values.validation_data
    central_values = validation_data.central_values
    covmat = validation_data.covmat
    central_values_idx = validation_data.central_values_idx

    @jax.jit
    def chi2(pdf):
        """
        TODO
        note: no batches training for validation data
        """
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
    TODO
    """
    validation_data = make_data_values.validation_data
    central_values = validation_data.central_values
    covmat = validation_data.covmat
    central_values_idx = validation_data.central_values_idx

    posdata_validation_idx = make_posdata_split.validation

    @jax.jit
    def chi2(pdf, alpha, lambda_positivity):
        """
        TODO
        """
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
    TODO
    N.B. make_chi2 function suited for a bayesian fit
    """
    training_data = make_data_values.training_data
    central_values = training_data.central_values
    covmat = training_data.covmat
    central_values_idx = training_data.central_values_idx

    @jax.jit
    def chi2(pdf):
        """
        TODO
        """
        diff = make_pred_data(pdf)[central_values_idx] - central_values

        # decompose covmat after having batched it!
        sqrt_covmat = jnp.array(sqrt_covmat_jax(covmat))

        # solve_triangular: solve the equation a x = b for x, assuming a is a triangular matrix.
        chi2_vec = jla.solve_triangular(sqrt_covmat, diff, lower=True)
        loss = jnp.sum(chi2_vec**2)
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
    TODO
    """
    training_data = make_data_values.training_data
    central_values = training_data.central_values
    covmat = training_data.covmat
    central_values_idx = training_data.central_values_idx

    posdata_training_idx = make_posdata_split.training

    @jax.jit
    def chi2(pdf):
        """
        TODO
        """
        diff = make_pred_data(pdf)[central_values_idx] - central_values

        sqrt_covmat = jnp.array(sqrt_covmat_jax(covmat))

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
