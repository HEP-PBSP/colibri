import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla

from super_net.theory_predictions import make_pred_data, make_penalty_posdata
from super_net.covmats import sqrt_covmat_jax


def make_chi2_training_data(
    data,
    data_training,
    posdatasets,
    posdata_training_index
):
    """
    Compute the chi2 between experimental (or pseudo data) central values
    and theoretical predictions using the (t0) experimental covariance matrix.
    The chi2 is computed on a data batch.

    Returns
    -------
    @jax.jit CompiledFunction
        Compiled function taking pdf grid, and indexes for
        the data batch (random subset of datapoints of the
        whole dataset) in input and returning the chi2
        between experimental central values and th. predictions
        using the exp covmat.
    """

    central_values_train = data_training["central_values_train"]
    covmat_train = data_training["t0covmat_train"]
    indices_train = data_training["central_values_train_index"]

    pred = make_pred_data(data)

    pos_penalty_func = make_penalty_posdata(posdatasets)

    @jax.jit
    def chi2(pdf, batch_idx, alpha, lambda_positivity):
        """
        Compute batched chi2

        Parameters
        ----------
        pdf :

        batch_idx :

        """
        diff = pred(pdf)[indices_train][batch_idx] - central_values_train[batch_idx]

        # batch covariance matrix before decomposing it
        batched_covmat = covmat_train[batch_idx][:, batch_idx]
        # decompose covmat after having batched it!
        sqrt_covmat = jnp.array(sqrt_covmat_jax(batched_covmat))

        # solve_triangular: solve the equation a x = b for x, assuming a is a triangular matrix.
        chi2_vec = jla.solve_triangular(sqrt_covmat, diff, lower=True)
        loss = jnp.sum(chi2_vec**2)
        
        pos_penalty = pos_penalty_func(pdf, alpha, lambda_positivity)[posdata_training_index]
        loss += jnp.sum(pos_penalty)
        
        return loss

    return chi2


def make_chi2_validation_data(data, data_validation, posdatasets, posdata_validation_index):
    """
    Compute the chi2 on the validation set. The chi2 is
    computed between experimental central values
    and theoretical predictions using the experimental
    covariance matrix.

    Returns
    -------
    @jax.jit CompiledFunction
        Compiled function taking pdf grid in input and returning the chi2
        between experimental central values and th. predictions
        using the exp covmat.
    """

    central_values_val = data_validation["central_values_val"]
    covmat_val = data_validation["t0covmat_val"]
    indices_val = data_validation["central_values_val_index"]

    pred = make_pred_data(data)

    pos_penalty_func = make_penalty_posdata(posdatasets)

    @jax.jit
    def chi2(pdf, alpha, lambda_positivity):
        """

        Parameters
        ----------
        pdf :

        """
        diff = pred(pdf)[indices_val] - central_values_val

        sqrt_covmat = jnp.array(sqrt_covmat_jax(covmat_val))

        # solve_triangular: solve the equation a x = b for x, assuming a is a triangular matrix.
        chi2_vec = jla.solve_triangular(sqrt_covmat, diff, lower=True)
        loss = jnp.sum(chi2_vec**2)

        pos_penalty = pos_penalty_func(pdf, alpha, lambda_positivity)[posdata_validation_index]
        loss += jnp.sum(pos_penalty)
        return loss

    return chi2
