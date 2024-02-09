"""
grid_pdf.model.py

The grid_pdf model.
"""

import jax
import jax.numpy as jnp

from validphys import convolution
from validphys.core import PDF

from colibri.pdf_model import PDFModel

import logging

log = logging.getLogger(__name__)


def pdf_model(flavour_xgrids, flavour_indices):
    return GridPDFModel(flavour_xgrids, flavour_indices)


class GridPDFModel(PDFModel):
    """A PDFModel implementation for the grid_pdf module."""

    xgrids: dict
    param_names: list

    def __init__(self, flavour_xgrids, flavour_indices):
        self.xgrids = flavour_xgrids
        self.flavour_indices = flavour_indices

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

    def grid_values_func(self, interpolation_grid, vectorized=False):
        """
        This function should produce a grid values function, which takes
        in the model parameters, and produces the PDF values on the grid xgrid.

        Parameters
        ----------
        interpolation_grid: list
            The grid to interpolate to.

        vectorized: bool, default is False
            Whether to use vectorized parameters.

        Returns
        -------
        interp_func: @jax.jit CompiledFunction
            The interpolation function.
        """

        if vectorized:
            # Function to perform interpolation for a single grid
            log.warning(
                "grid_pdf model in vectorized mode does not support xgrids of different sizes."
            )

            # check that all xgrids have the same length
            if len(set([len(self.xgrids[fl]) for fl in self.fitted_flavours])) != 1:
                raise ValueError(
                    "grid_pdf model in vectorized mode does not yet support xgrids of different sizes."
                )

            @jax.jit
            def interpolate_flavors(y):
                """
                Interpolates one vectorized flavour.
                """
                reshaped_y = y.reshape(
                    (
                        len(self.flavour_indices),
                        len(
                            self.xgrids[self.fitted_flavours[0]]
                        ),  # only works if all xgrids have the same length
                    )
                )

                out = jnp.array(
                    [
                        jnp.interp(
                            jnp.array(interpolation_grid),
                            jnp.array(self.xgrids[flavour]),
                            reshaped_y[i, :],
                        )
                        for i, flavour in enumerate(self.fitted_flavours)
                    ]
                )
                return out
            
            @jax.jit
            def interpolate_vector(fp, flavour_xgrid):
                """
                Interpolates one vectorized flavour.

                Parameters
                ----------
                fp: jnp.array
                    The PDF values to interpolate.
                
                flavour_xgrid: jnp.array
                    The xgrid for the flavour.
                
                Returns
                -------
                jnp.array
                    The interpolated PDF values.
                """
                return jnp.interp(
                    jnp.array(interpolation_grid),
                    flavour_xgrid,
                    fp,
                )

            @jax.jit
            def interp_func(stacked_pdf_grid):
                """
                Interpolates the stacked PDF grid.

                Parameters
                ----------
                stacked_pdf_grid: jnp.array
                    The vectorized stacked PDF grid.
                
                Returns
                -------
                jnp.array
                    The interpolated vectorized PDF values.
                """
                interpolants = []
                for flavour in convolution.FK_FLAVOURS:

                    if flavour in self.fitted_flavours:
                        
                        # apply vmap to interpolate all vectors of the same flavour at once
                        interpolated_flavour = jax.vmap(interpolate_vector, in_axes=(0, None))(stacked_pdf_grid[:, :len(self.xgrids[flavour])], jnp.array(self.xgrids[flavour]))
                        
                        interpolants += [interpolated_flavour]
                    else:
                        interpolants += [jnp.zeros((stacked_pdf_grid.shape[0], len(interpolation_grid)))]

                    stacked_pdf_grid = stacked_pdf_grid[:, len(self.xgrids[flavour]):]      
                
                return jnp.transpose(jnp.array(interpolants), (1,0,2))


            return interp_func

        else:

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

    # This is needed to avoid ultranest crashing with
    # ValueError: Buffer dtype mismatch, expected 'float_t' but got 'float'
    jax.config.update("jax_enable_x64", True)

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

    return prior_transform
