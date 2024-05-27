"""
colibri.loss_functions.py

This module provides the functions necessary for the computation of the chi2.

"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla


def make_chi2(central_covmat_index, vectorized=False):
    """
    Returns a jax.jit compiled function that computes the chi2
    of a pdf grid on a dataset.

    Notes:
        - Does not include positivity constraint.
        - This function is designed for Bayesian like PDF fits.
        - allows for vectorized evaluation of the chi2.

    Parameters
    ----------
    central_covmat_index: commondata_utils.CentralCovmatIndex class
        dataclass containing central values and covmat.

    vectorized: bool, default is False

    Returns
    -------
    @jax.jit Callable
        function to compute chi2 of a pdf grid.

    """
    central_values = central_covmat_index.central_values
    covmat = central_covmat_index.covmat

    # Invert the covmat
    # We use this instead of Cholesky decomposition
    # since we do it only once and for all at the beginning
    inv_covmat = jla.inv(covmat)

    @jax.jit
    def chi2(predictions):
        """ """
        diff = predictions - central_values

        loss = jnp.einsum("i,ij,j", diff, inv_covmat, diff)

        return loss

    if vectorized:
        # return jnp.vectorize(chi2, signature="(m,n)->()")
        return jax.vmap(chi2, in_axes=(0,), out_axes=0)
    return chi2


def make_chi2_with_positivity(
    central_covmat_index,
    _penalty_posdata,
    alpha=1e-7,
    lambda_positivity=1000,
    vectorized=False,
    float_type=None,
):
    """
    Returns a jax.jit compiled function that computes the chi2
    of a pdf grid on a dataset.

    Notes:
        - This function is designed for Bayesian like PDF fits.

    Parameters
    ----------
    central_covmat_index: commondata_utils.CentralCovmatIndex class
        dataclass containing central values and covmat.

    _penalty_posdata: theory_predictions._penalty_posdata
        colibri provider used to compute positivity penalty.

    alpha: float

    lambda_positivity: float

    vectorized: bool, default is False

    float_type: type, default is None

    Returns
    -------
    @jax.jit Callable
        function to compute chi2 of a pdf grid.

    """
    central_values = jnp.array(central_covmat_index.central_values, dtype=float_type)
    covmat = central_covmat_index.covmat

    # Invert the covmat
    inv_covmat = jnp.array(jla.inv(covmat), dtype=float_type)

    @jax.jit
    def chi2(predictions, pdf):
        """ """
        diff = predictions - central_values

        loss = jnp.einsum("i,ij,j", diff, inv_covmat, diff)

        # add penalty term due to positivity
        pos_penalty = _penalty_posdata(pdf, alpha, lambda_positivity)

        loss += jnp.sum(pos_penalty)
        return loss

    if vectorized:
        # jnp.vectorize(chi2, signature="(m,n)->()")
        return jax.vmap(chi2, in_axes=(0, 0), out_axes=0)

    return chi2
