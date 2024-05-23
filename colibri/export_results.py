"""
colibri.export_results.py

This module contains the functions to export the results of the fit.

"""

from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class BayesianFit:
    """
    Dataclass containing the results and specs of a Bayesian fit.

    Attributes
    ----------
    resampled_posterior: jnp.array
        Array containing the resampled posterior samples.
    full_posterior_samples: jnp.array
        Array containing the full posterior samples.
    bayes_complexity: float
        The Bayesian complexity of the model.
    avg_chi2: float
        The average chi2 of the model.
    min_chi2: float
        The minimum chi2 of the model.
    logz: float
        The log evidence of the model.
    """

    resampled_posterior: jnp.array
    full_posterior_samples: jnp.array
    bayes_complexity: float
    avg_chi2: float
    min_chi2: float
    logz: float
