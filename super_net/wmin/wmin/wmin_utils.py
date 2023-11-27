"""
wmin.wmin_utils.py

Module containing util functions for weight minimisation PDF fits.

Author: Mark N. Costantini
Date: 11.11.2023
"""

import logging

import jax
import jax.numpy as jnp

log = logging.getLogger(__name__)


def wmin_fit_name(wminpdfset, set_name=None):
    if set_name:
        return set_name
    return "wmin_fit_" + str(wminpdfset)


def wmin_grid_seed(wmin_grid_index):
    """
    Wmin PRNGKey used for the random choice of wmin replicas to be used
    in the wmin parametrisation and random choice of the central wmin
    replica
    """
    key = jax.random.PRNGKey(wmin_grid_index)
    return key


def weights_initializer_provider(
    weights_initializer="zeros",
    weights_seed=0xABCDEF,
    uniform_minval=-0.1,
    uniform_maxval=0.1,
):
    """
    Function responsible for the initialization of the weights in a weight minimization fit.

    Parameters
    ----------
    weights_initializer: str, default is 'zeros'
            the available options are: ('zeros', 'normal', 'uniform')
            if an unknown option is specified, the 'zeros' will be used

    weights_seed: (Union[int, Array]) – a 64- or 32-bit integer used as the value of the key.

    uniform_minval: see minval of jax.random.uniform

    uniform_maxval: see maxval of jax.random.uniform

    Returns
    -------
    function that takes shape=integer in input and returns array of shape = (shape, )

    """
    if weights_initializer not in ("zeros", "normal", "uniform"):
        log.warning(
            f"weights_initializer {weights_initializer} name not recognized, using default: 'zeros' instead"
        )
        weights_initializer = "zeros"

    if weights_initializer == "zeros":
        return jnp.zeros

    elif weights_initializer == "normal":
        rng = jax.random.PRNGKey(weights_seed)
        initializer = lambda shape: jax.random.normal(key=rng, shape=(shape,))
        return initializer

    elif weights_initializer == "uniform":
        rng = jax.random.PRNGKey(weights_seed)
        initializer = lambda shape: jax.random.uniform(
            key=rng, shape=(shape,), minval=uniform_minval, maxval=uniform_maxval
        )
        return initializer


def weight_minimization_prior(
    n_replicas_wmin,
    prior_type="uniform",
    unif_prior_min_val=-0.7,
    unif_prior_max_val=0.7,
):
    """
    TODO
    """

    if prior_type == "uniform":

        def prior_transform(cube):
            """
            TODO
            """
            params = cube.copy()
            for i in range(n_replicas_wmin - 1):
                params[i] = (
                    cube[i] * (unif_prior_max_val - unif_prior_min_val)
                    + unif_prior_min_val
                )
            return params

        return prior_transform


def resample_from_wmin_posterior(
    samples, n_wmin_posterior_samples=1000, wmin_posterior_resampling_seed=123456
):
    """
    TODO
    """

    current_samples = samples.copy()

    rng = jax.random.PRNGKey(wmin_posterior_resampling_seed)

    resampled_samples = jax.random.choice(
        rng, len(samples), (n_wmin_posterior_samples,), replace=False
    )

    return jnp.array(current_samples[resampled_samples])
