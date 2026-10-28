import jax
import jax.numpy as jnp

from colibri.utils import (
    cast_to_numpy,
    get_full_posterior,
)
from colibri.core import BayesianPrior
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions


def bayesian_prior(prior_settings, forward_map):
    """
    Produces a prior transform function.

    Parameters
    ----------
    prior_settings: dict
        The settings for the prior transform.

    forward_map: ForwardMap
        The forward map of the problem, used to determine parameter names and ordering.


    Returns
    -------
    prior_transform: @jax.jit CompiledFunction
        The prior transform function.
    """
    if prior_settings.prior_distribution == "uniform_parameter_prior":
        prior_specs = prior_settings.prior_distribution_specs

        if "bounds" in prior_specs:
            # Use param names from the forward map to order bounds correctly
            param_names = forward_map.param_names
            bounds_dict = prior_specs["bounds"]
            missing = [p for p in param_names if p not in bounds_dict]
            if missing:
                raise ValueError(f"Missing bounds for parameters: {missing}")

            # Per-parameter bounds
            bounds = jnp.array([bounds_dict[param] for param in param_names])
            mins = bounds[:, 0]
            maxs = bounds[:, 1]

        elif "min_val" in prior_specs and "max_val" in prior_specs:
            # Global bounds for all parameters
            n_params = len(forward_map.param_names)
            mins = jnp.array([float(prior_specs["min_val"])] * n_params)
            maxs = jnp.array([float(prior_specs["max_val"])] * n_params)

        else:
            raise ValueError(
                "prior_distribution_specs must define either 'bounds' or 'min_val' and 'max_val'"
            )

        prior = tfd.Uniform(low=mins, high=maxs)

        def sample(rng_key, n_samples):
            return prior.sample(seed=rng_key, sample_shape=(n_samples,))

        @jax.jit
        def log_prob(x):
            return prior.log_prob(x).sum(axis=-1)

        @jax.jit
        def prior_transform(cube):
            return cube * (maxs - mins) + mins

        @jax.jit
        def from_unconstrained(position):
            return mins + (maxs - mins) * jax.nn.sigmoid(position)

        @jax.jit
        def log_prob_unconstrained(position):
            # For a uniform prior, its normalization cancels the
            # interval-width contribution to the Jacobian.
            return jnp.sum(
                jax.nn.log_sigmoid(position) + jax.nn.log_sigmoid(-position),
                axis=-1,
            )

        def sample_unconstrained(rng_key, n_samples):
            dtype = jnp.asarray(0.0).dtype
            eps = jnp.finfo(dtype).eps
            cube = jax.random.uniform(
                rng_key,
                shape=(n_samples, mins.shape[0]),
                dtype=dtype,
                minval=eps,
                maxval=1 - eps,
            )
            return jnp.log(cube) - jnp.log1p(-cube)

    elif prior_settings.prior_distribution == "prior_from_gauss_posterior":
        from_unconstrained = None
        log_prob_unconstrained = None
        sample_unconstrained = None

        prior_fit = prior_settings.prior_distribution_specs["prior_fit"]

        df_fit = get_full_posterior(prior_fit)

        # Compute mean and covariance matrix of the posterior
        mean_posterior = jnp.array(df_fit.mean().values)
        cov_posterior = jnp.array(df_fit.cov().values)

        sqrt_cov_posterior = jnp.linalg.cholesky(cov_posterior)

        # Define dummy log_prob and sample for now
        @jax.jit
        def log_prob(x):
            raise NotImplementedError("log_prob not implemented for Gaussian prior")

        def sample(rng_key, n_samples):
            raise NotImplementedError("sample not implemented for Gaussian prior")

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

    return BayesianPrior(
        prior_transform=prior_transform,
        log_prob=log_prob,
        sample=sample,
        from_unconstrained=from_unconstrained,
        log_prob_unconstrained=log_prob_unconstrained,
        sample_unconstrained=sample_unconstrained,
    )
