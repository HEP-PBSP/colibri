"""
grid_pdf.model.py

The grid_pdf model.
"""

import jax
import jax.numpy as jnp
import dill

from validphys import convolution
from validphys.core import PDF

from colibri.pdf_model import PDFModel


def pdf_model(flavour_xgrids, output_path):
    model = GridPDFModel(flavour_xgrids)
    # dump model to output_path using dill
    # this is mainly needed by scripts/ns_resampler.py
    with open(output_path / "pdf_model.pkl", "wb") as file:
        dill.dump(model, file)
    return model


class GridPDFModel(PDFModel):
    """A PDFModel implementation for the grid_pdf module."""

    xgrids: dict
    param_names: list

    def __init__(self, flavour_xgrids):
        self.xgrids = flavour_xgrids

    @property
    def param_names(self):
        """The fitted parameters of the model."""
        return [f"{fl}({x})" for fl in self.fitted_flavours for x in self.xgrids[fl]]

    @property
    def n_parameters(self):
        """The number of parameters of the model."""
        return len(self.param_names)

    @property
    def fitted_flavours(self):
        """The fitted flavours used in the model, in STANDARDISED order,
        according to convolution.FK_FLAVOURS
        """
        flavours = []
        for flavour in convolution.FK_FLAVOURS:
            if self.xgrids[flavour]:
                flavours += [flavour]
        return flavours

    def grid_values_func(self, interpolation_grid):
        """This function should produce a grid values function, which takes
        in the model parameters, and produces the PDF values on the grid xgrid.
        """

        @jax.jit
        def interp_func(params):
            # Perform the interpolation for each flavour in turn
            interpolants = []
            for flavour in convolution.FK_FLAVOURS:
                if flavour in self.fitted_flavours:
                    interpolants += [
                        jnp.interp(
                            jnp.array(interpolation_grid),
                            jnp.array(self.xgrids[flavour]),
                            jnp.array(params[: len(self.xgrids[flavour])]),
                        )
                    ]
                else:
                    interpolants += [jnp.array([0.0] * len(interpolation_grid))]
                params = params[len(self.xgrids[flavour]) :]
            return jnp.array(interpolants)

        return interp_func

    def pred_and_pdf_func(self, xgrid, pdf_pred_func):
        """This function should produce a function that takes in the model
        parameters, and produces the predictions for the data, as well as the
        PDF values on the grid.

        The pdf_pred_func is a function that takes in the PDF defined on the
        xgrid grid. They must therefore be compatible.
        """

        @jax.jit
        def pred_and_pdf(params):
            pdf = self.grid_values_func(xgrid)(params)
            predictions = pdf_pred_func(pdf)
            return predictions, pdf

        return pred_and_pdf


def mc_initial_parameters(pdf_model, mc_initialiser_settings, replica_index):
    """
    The initial parameters for the Monte Carlo fit. The options for the
    initialiser are given in mc_initialiser_settings, which is a dictionary
    with required key 'type'. The 'type' is one of 'zeros', 'uniform', 'pdf'.


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
    if mc_initialiser_settings["type"] == "zeros":
        return jnp.zeros(shape=pdf_model.n_parameters)

    rng = jax.random.PRNGKey(replica_index)

    if mc_initialiser_settings["type"] == "uniform":
        return jax.random.uniform(
            rng,
            shape=(pdf_model.n_parameters,),
            minval=mc_initialiser_settings["minval"],
            maxval=mc_initialiser_settings["maxval"],
        )

    elif mc_initialiser_settings["type"] == "pdf":
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


def bayesian_prior(pdf_model, prior_settings):
    """
    Produces the Bayesian prior for a grid_pdf fit. The options for the
    prior are given in prior_settings, which is a dictionary with required
    key 'type'. The 'type' is one of 'uniform_pdf_prior', 'gaussian_pdf_prior'.

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

    # Load the prior PDF
    pdf = PDF(prior_settings["pdf_prior"])
    nsigma = prior_settings["nsigma"]

    replicas_grid = jnp.concatenate(
        [
            convolution.evolution.grid_values(
                pdf, [flavour], pdf_model.xgrids[flavour], [1.65]
            )
            for flavour in pdf_model.fitted_flavours
        ],
        axis=2,
    )

    central_prior_grid = replicas_grid[0, :, :, :].squeeze()
    # Remove central replica
    replicas_grid = replicas_grid[1:, :, :, :]

    if prior_settings["type"] == "uniform_pdf_prior":
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
        raise ValueError(
            "Invalid prior type. Expected 'uniform_pdf_prior' or 'gaussian_pdf_prior'."
        )

    return prior_transform
