"""
super_net.ns_utils.py

Module containing util functions for nested sampling fits.

Author: James Moore
Date: 18.12.2023
"""

import jax
import jax.numpy as jnp

def resample_from_ns_posterior(
    samples, n_posterior_samples=1000, posterior_resampling_seed=123456
):
    """
    TODO
    """

    current_samples = samples.copy()

    rng = jax.random.PRNGKey(posterior_resampling_seed)

    resampled_samples = jax.random.choice(
        rng, len(samples), (n_posterior_samples,), replace=False
    )

    return jnp.array(current_samples[resampled_samples])
