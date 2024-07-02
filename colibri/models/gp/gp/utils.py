"""
gp.utils.py

Module containing util functions for GP fits.
"""

import jax
import jax.numpy as jnp


def bayesian_prior(pdf_model, prior_settings, FIT_XGRID):
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

    if prior_settings["type"] == "gibbs_kernel_prior":

        @jax.jit
        def prior_transform(cube):
            """
            TODO: how to put a prior on the hyperparameters?
            """
            # split cube based on the total number of grid points used to discretize the PDFs
            stacked_pdf_grid, hyperparameters = jnp.split(cube, [n_grid_points])

            # Impose constraints on the hyperparameters here
            hyperparameters = (
                jnp.array([0.01, 0.01, -1.0])
                + jnp.array([12.0, 2.0, 1.0]) * hyperparameters
            )

            kernel = gibbs_kernel(
                hyperparameters, jnp.array(FIT_XGRID), jnp.array(FIT_XGRID)
            )

            # TODO: mean prior should be an input too, not hardcoded
            mean = jnp.zeros(n_grid_points)

            # TODO: use svd decomposition for numerical stability
            cholesky_covmat = jnp.linalg.cholesky(kernel)

            # generate standard normal random numbers from uniform random numbers
            stacked_pdf_grid = jax.scipy.stats.norm.ppf(stacked_pdf_grid)

            # shift the standard normal random numbers to the mean and scale them by the covariance matrix
            prior = mean + jnp.einsum("ij,j->i", cholesky_covmat, stacked_pdf_grid)

            # concatenate the prior with the hyperparameters
            prior = jnp.concatenate([prior, hyperparameters])

            return prior

    return prior_transform


def gibbs_kernel(hyperparameters, xgrid1, xgrid2, delta_reg=1e-8):
    """
    Defines a Gibbs kernel with 3 hyperparameters: L0, SIGMA, ALPHA.

    Kernel is defined as:

    k(x, y) = SIGMA^2 * sqrt(2 * l(x) * l(y) / (l(x)^2 + l(y)^2)) * exp(-(x - y)^2 / (l(x)^2 + l(y)^2)) * x^ALPHA * y^ALPHA

        where l(x) = L0 * (x + DELTA)

    Parameters
    ----------
    hyperparameters: list, array, tuple
        hyperparameters [L0, SIGMA, ALPHA]

    xgrid1: array
        x grid points

    xgrid2: array
        y grid points

    delta_reg: float
        regularization parameter

    returns
    -------
    kernel: array
        Gibbs kernel
    """
    L0 = hyperparameters[0]
    SIGMA = hyperparameters[1]
    ALPHA = hyperparameters[2]
    DELTA = delta_reg

    l = lambda x: L0 * (x + DELTA)

    gibbs = (
        lambda x, y: SIGMA**2
        * jnp.sqrt((2.0 * l(x) * l(y)) / (l(x) ** 2 + l(y) ** 2))
        * jnp.exp(-((x - y) ** 2) / (l(x) ** 2 + l(y) ** 2))
        * x**ALPHA
        * y**ALPHA
    )

    kernel = jax.vmap(jax.vmap(gibbs, in_axes=(None, 0)), in_axes=(0, None))(
        jnp.array(xgrid1), jnp.array(xgrid2)
    )

    return kernel
