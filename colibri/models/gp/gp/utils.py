"""
gp.utils.py

Module containing util functions for GP fits.
"""

import jax
import jax.numpy as jnp

from validphys.core import PDF


def bayesian_prior(pdf_model):
    """
    Produces the Bayesian prior for a grid_pdf fit. The options for the
    prior are given in prior_settings, which is a dictionary with required
    key 'type'. The 'type' is one of 'uniform_pdf_prior', 'gaussian_pdf_prior'.

    NOTE: this function overrides the one in colibri.bayes_prior.

    Parameters
    ----------
    pdf_model: pdf_model.PDFModel
        The PDF model to fit.

    prior_settings: dict
        Settings for the prior.

    Returns
    -------
    jit compiled function
        The prior transform function.
    """

    n_grid_points = pdf_model.n_parameters

    @jax.jit
    def prior_transform(cube):
        """
        TODO
        """
        # split cube based on the total number of grid points used to discretize the PDFs
        stacked_pdf_grid, hyperparameters = jnp.split(cube, [n_grid_points])

        mean = jnp.zeros(n_grid_points)
        covmat = jnp.eye(n_grid_points)
        cholesky_covmat = jnp.linalg.cholesky(covmat)

        # generate standard normal random numbers from uniform random numbers
        stacked_pdf_grid = jax.scipy.stats.norm.ppf(stacked_pdf_grid)

        # shift the standard normal random numbers to the mean and scale them by the covariance matrix
        prior = mean + jnp.einsum("ij,j->i", cholesky_covmat, stacked_pdf_grid)

        # concatenate the prior with the hyperparameters
        prior = jnp.concatenate([prior, hyperparameters])

        return prior

    return prior_transform
