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
        prior_specs = prior_settings.prior_distribution_specs

        if "bounds" in prior_specs:
            # Per-parameter bounds
            bounds = jnp.array(list(prior_specs["bounds"].values()))  # shape (n_params, 2)
            mins = bounds[:, 0]
            maxs = bounds[:, 1]

            @jax.jit
            def prior_transform(cube):
                return cube * (maxs - mins) + mins

        elif "min_val" in prior_specs and "max_val" in prior_specs:
            # Global bounds for all parameters
            min_val = prior_specs["min_val"]
            max_val = prior_specs["max_val"]

            @jax.jit
            def prior_transform(cube):
                return cube * (max_val - min_val) + min_val

        else:
            raise ValueError(
                "prior_distribution_specs must define either 'bounds' or 'min_val' and 'max_val'"
            )

    elif prior_settings.prior_distribution == "prior_from_gauss_posterior":
        prior_fit = prior_settings.prior_distribution_specs["prior_fit"]

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
        raise ValueError("Invalid prior type.")
    return prior_transform
