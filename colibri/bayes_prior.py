import jax
import jax.numpy as jnp

from colibri.utils import (
    cast_to_numpy,
    get_full_posterior,
)
from colibri.checks import check_pdf_models_equal


@check_pdf_models_equal
def bayesian_prior(prior_settings):
    """
    Produces a prior transform function.

    Parameters
    ----------
    prior_settings: dict
        The settings for the prior transform.

    Returns
    -------
    prior_transform: @jax.jit CompiledFunction
        The prior transform function.
    """
    if prior_settings.prior_distribution == "uniform_parameter_prior":
        max_val = prior_settings.max_val
        min_val = prior_settings.min_val

        @jax.jit
        def prior_transform(cube):
            return cube * (max_val - min_val) + min_val

    elif prior_settings.prior_distribution == "prior_from_gauss_posterior":
        prior_fit = prior_settings.prior_fit

        df_fit = get_full_posterior(prior_fit)

        # Compute mean and covariance matrix of the posterior
        mean_posterior = jnp.array(df_fit.mean().values)
        cov_posterior = jnp.array(df_fit.cov().values)

        sqrt_cov_posterior = jnp.linalg.cholesky(cov_posterior)

        @cast_to_numpy
        @jax.jit
        def prior_transform(cube):
            # generate independent gaussian with mean 0 and std 1
            independent_gaussian = jax.scipy.stats.norm.ppf(cube)
            return mean_posterior + jnp.einsum(
                "ij,...j->...i", sqrt_cov_posterior, independent_gaussian
            )

    else:
        raise ValueError("Invalid prior distribution.")
    return prior_transform
