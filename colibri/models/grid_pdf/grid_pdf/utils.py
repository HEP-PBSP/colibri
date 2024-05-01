"""
grid_pdf.utils.py

Module containing util functions for grid PDF fits.
"""

from validphys import convolution

import jax
import jax.numpy as jnp

import colibri
from validphys.core import PDF

import colibri.bayes_prior


def closure_test_central_pdf_grid(
    closure_test_pdf,
    pdf_model,
    FIT_XGRID,
    reduced_xgrid_data=False,
):
    """
    Computes the central member of the closure_test_pdf grid in the
    evolution basis and only on x points that are specified in xgrids.
    The grid is then interpolated to the full XGRID.

    NOTE: when reduced_xgrid_data=True, this function overrides the one in colibri.utils
    otherwise the one in colibri.utils is used.

    Parameters
    ----------
    closure_test_pdf: validphys.core.PDF

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    pdf_model: colibri.pdf_model.PDFModel
        Specifically, this is the GridPDFModel for this provider.

    reduced_xgrid_data: bool, default is False
        When True the closure_test_central_pdf_grid is overriden.
        When False the closure_test_pdf_grid from colibri.utils is used.


    Returns
    -------
    grid: jnp.array
        grid, is N_fl x N_x
    """

    if not reduced_xgrid_data:
        return colibri.utils.closure_test_pdf_grid(
            closure_test_pdf, FIT_XGRID, Q0=1.65
        )[0]

    # Obtain the PDF values as parameters, then use the model interpolation function
    interpolator = pdf_model.grid_values_func(FIT_XGRID)

    parameters = []
    for fl in pdf_model.xgrids.keys():
        x_vals = pdf_model.xgrids[fl]
        if x_vals:
            parameters += [
                convolution.evolution.grid_values(
                    closure_test_pdf, [fl], x_vals, [1.65]
                )
                .squeeze(-1)[0]
                .squeeze(0)
            ]

    parameters = jnp.concatenate(parameters)
    reduced_pdfgrid = interpolator(parameters)

    return reduced_pdfgrid


def pdf_prior_grid(prior_settings, pdf_model):
    """
    Load the replicas grid for the Bayesian prior.
    """
    # Load the prior PDF
    pdf = PDF(prior_settings["pdf_prior"])

    replicas_grid = jnp.concatenate(
        [
            convolution.evolution.grid_values(
                pdf, [flavour], pdf_model.xgrids[flavour], [1.65]
            )
            for flavour in pdf_model.fitted_flavours
        ],
        axis=2,
    )

    return replicas_grid


def bayesian_prior(prior_settings, pdf_model):
    """
    Produces the Bayesian prior for a grid_pdf fit. The options for the
    prior are given in prior_settings, which is a dictionary with required
    key 'type'. The 'type' is one of 'uniform_pdf_prior', 'gaussian_pdf_prior'.

    NOTE: this function overridest the one in colibri.bayes_prior.

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

    if prior_settings["type"] == "uniform_pdf_prior":
        nsigma = prior_settings["nsigma"]
        pdf_grid = pdf_prior_grid(prior_settings, pdf_model)

        # Remove central replica
        replicas_grid = pdf_grid[1:, :, :, :]

        error68_up = jnp.nanpercentile(replicas_grid, 84.13, axis=0).reshape(-1)
        error68_down = jnp.nanpercentile(replicas_grid, 15.87, axis=0).reshape(-1)

        # Compute the uncertainty on the central grid
        delta = (error68_up - error68_down) / 2
        mean = (error68_up + error68_down) / 2
        error_up = mean + delta * nsigma
        error_down = mean - delta * nsigma

        @jax.jit
        def prior_transform(cube):
            params = error_down + (error_up - error_down) * cube
            return params

    elif prior_settings["type"] == "gaussian_pdf_prior":
        pdf_grid = pdf_prior_grid(prior_settings, pdf_model)

        central_prior_grid = pdf_grid[0, :, :, :].squeeze()
        # Remove central replica
        replicas_grid = pdf_grid[1:, :, :, :]

        pdf_covmat_prior = jnp.cov(
            replicas_grid.reshape((replicas_grid.shape[0], -1)).T
        )

        # note: this matrix could be ill-conditioned when too many xgrids are used
        # this is because xgrid values are not independent
        # in order to avoid this we only keep the diagonal of the covariance matrix
        # and regularise it by cutting off small values
        pdf_diag_covmat_prior = jnp.where(
            jnp.diag(pdf_covmat_prior) < 0, 0, jnp.diag(pdf_covmat_prior)
        )
        cholesky_pdf_covmat = jnp.diag(jnp.sqrt(pdf_diag_covmat_prior))

        @jax.jit
        def prior_transform(cube):
            """
            This currently does not support vectorisation.
            """
            # generate independent gaussian with mean 0 and std 1
            independent_gaussian = jax.scipy.stats.norm.ppf(cube)[:, jnp.newaxis]

            # generate random samples from a multivariate normal distribution
            prior = central_prior_grid + jnp.einsum(
                "ij,kj->ki", independent_gaussian.T, cholesky_pdf_covmat
            ).squeeze(-1)

            return prior

    else:
        return colibri.bayes_prior.bayesian_prior(prior_settings, pdf_model)

    return prior_transform


def mc_initial_parameters(pdf_model, mc_initialiser_settings, replica_index):
    """
    The initial parameters for the Monte Carlo fit.

    NOTE: this function overridest the one in colibri.mc_initialisation.

    Parameters
    ----------
    pdf_model: pdf_model.PDFModel
        The PDF model to fit.

    mc_initialiser_settings: dict
        Settings for the initialiser.

    replica_index: int
        The index of the replica.

    Returns
    -------
    jnp.array
        The initial parameters.
    """

    if mc_initialiser_settings["type"] == "pdf":
        rng = jax.random.PRNGKey(replica_index)
        # Load the PDF
        pdf = PDF(mc_initialiser_settings["pdf_set"])

        replicas_grid = jnp.concatenate(
            [
                convolution.evolution.grid_values(
                    pdf, [flavour], pdf_model.xgrids[flavour], [1.65]
                )
                for flavour in pdf_model.fitted_flavours
            ],
            axis=2,
        )

        central_grid = replicas_grid[0, :, :, :].squeeze()
        # Remove central replica
        replicas_grid = replicas_grid[1:, :, :, :]

        if mc_initialiser_settings["init_type"] == "central":
            return central_grid

        elif mc_initialiser_settings["init_type"] == "uniform":
            nsigma = mc_initialiser_settings["nsigma"]

            error68_up = jnp.nanpercentile(replicas_grid, 84.13, axis=0).reshape(-1)
            error68_down = jnp.nanpercentile(replicas_grid, 15.87, axis=0).reshape(-1)

            # Compute the uncertainty on the central grid
            delta = (error68_up - error68_down) / 2

            # Generate a random number between -nsigma and nsigma
            epsilon = jax.random.uniform(
                rng,
                shape=(pdf_model.n_parameters,),
                minval=-nsigma,
                maxval=nsigma,
            )

            return central_grid + epsilon * delta
    else:
        return colibri.mc_initialisation.mc_initial_parameters(
            pdf_model, mc_initialiser_settings, replica_index
        )
