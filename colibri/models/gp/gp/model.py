"""
gp.model.py

The Gaussian process model.
"""

import jax
import jax.numpy as jnp
import numpy as np

from colibri.pdf_model import PDFModel
from colibri.constants import FLAVOUR_TO_ID_MAPPING
import time
from validphys import convolution
from gp.utils import gibbs_kernel


class GpPDFModel(PDFModel):
    """
    A PDFModel implementation for the Gaussian process module.

    Attributes
    ----------

    """

    fit_xgrid: list
    fitted_flavours: list
    param_names: list
    gp_hyperparams_settings: dict
    prior_settings: dict

    name = "Gaussian process PDF model"

    def __init__(
        self, fit_xgrid, fitted_flavours, gp_hyperparams_settings, prior_settings
    ):
        self.fit_xgrid = fit_xgrid
        self.fitted_flavours = fitted_flavours
        self.gp_hyperparams_settings = gp_hyperparams_settings
        self.prior_settings = prior_settings

    @property
    def param_names(self) -> list[str]:
        """
        The parameters of the model, including hyperparameters of the GP.

        Returns
        -------
        list[str]
            A list of parameter names in the format "flavour(x)" for xgrid parameters
            and "hyperparameter(flavour)" for GP hyperparameters.
        """
        parameter_names = []

        # Ensure necessary attributes are present
        if not hasattr(self, "fitted_flavours"):
            raise AttributeError(
                "The model is missing the 'fitted_flavours' attribute."
            )
        if not hasattr(self, "fit_xgrid"):
            raise AttributeError("The model is missing the 'fit_xgrid' attribute.")
        if not hasattr(self, "gp_hyperparams_settings"):
            raise AttributeError(
                "The model is missing the 'gp_hyperparams_settings' attribute."
            )

        # Generate parameter names for the grid values
        parameter_names.extend(
            f"{fl}({x})" for fl in self.fitted_flavours for x in self.fit_xgrid
        )

        # Generate parameter names for the GP hyperparameters
        for fl in self.fitted_flavours:
            if fl in self.gp_hyperparams_settings:

                parameter_names.extend(
                    f"{hyp_param}({fl})"
                    for hyp_param in self.gp_hyperparams_settings[fl]
                )

        return parameter_names

    @property
    def n_parameters(self):
        """
        The number of parameters of the model.
        This does not include the GP hyperparameters.
        """
        # each flavour has the same number of parameters as the grid
        return int(len(self.fitted_flavours) * len(self.fit_xgrid))

    @property
    def n_hyperparameters(self):
        """
        The number of hyperparameters of the model.
        """
        n_hyperparams = 0
        for _, hyperparams in self.gp_hyperparams_settings.items():
            n_hyperparams += len(hyperparams)
        return n_hyperparams

    def grid_values_func(self, xgrid):
        """This function should produce a grid values function, which takes
        in the model parameters, and produces the PDF values on the grid xgrid.
        """
        if (len(xgrid) != len(self.fit_xgrid)) or (
            not jnp.allclose(jnp.array(xgrid), self.fit_xgrid)
        ):

            if type(xgrid) == list:
                xgrid = jnp.array(xgrid)

            # remove points that are already in the training grid
            common_xgrid_mask = jnp.isin(xgrid, self.fit_xgrid)
            common_xgrid_idx = jnp.where(common_xgrid_mask)[0]
            common_fit_xgrid_idx = jnp.where(
                self.fit_xgrid == xgrid[common_xgrid_mask]
            )[0]

            xgrid = xgrid[~common_xgrid_mask]

            def pdf_func(params):
                """
                Does the conditioning of the GP on the grid values.

                Parameters
                ----------
                params: jnp.array
                    The model parameters.
                """

                # split pdf_grid parameters from GP hyperparameters
                pdf_grid, hyperparameters = jnp.split(params, [self.n_parameters])

                # compute the kernel on the training grid
                training_kernel = self.gp_kernel(
                    hyperparameters, self.fit_xgrid, self.fit_xgrid, self.prior_settings
                )

                # compute the kernel on the test grid
                test_kernel = self.gp_kernel(
                    hyperparameters, xgrid, xgrid, self.prior_settings
                )

                # compute covariance between training and test grid
                cross_kernel = self.gp_kernel(
                    hyperparameters, self.fit_xgrid, xgrid, self.prior_settings
                )

                # compute the inverse of the training kernel
                training_kernel_inv = jnp.linalg.inv(training_kernel)

                # compute the mean of the GP
                mean_gp = self.gp_mean(
                    hyperparameters, xgrid, self.prior_settings
                ) + jnp.dot(
                    cross_kernel.T,
                    jnp.dot(
                        training_kernel_inv,
                        pdf_grid
                        - self.gp_mean(
                            hyperparameters, self.fit_xgrid, self.prior_settings
                        ),
                    ),
                )

                # compute the covariance of the GP
                cov_gp = test_kernel - jnp.dot(
                    cross_kernel.T, jnp.dot(training_kernel_inv, cross_kernel)
                )

                # generate random key with seed that depends on current time
                seed = int(time.time() * 1e9)
                key = jax.random.PRNGKey(seed)

                # sample from the GP
                # note: need to use svd method for numerical stability (cholesky can fail because of numerical negative zeros)
                pdf_grid = jax.random.multivariate_normal(
                    key=key, mean=mean_gp, cov=cov_gp, method="svd"
                )

                # add back the points that were already in the training grid
                pdf_grid = jnp.insert(
                    pdf_grid, common_xgrid_idx + 1, params[common_fit_xgrid_idx]
                )

                lhapdf_grid = np.zeros((len(convolution.FK_FLAVOURS), len(pdf_grid)))

                for flavour in convolution.FK_FLAVOURS:
                    if flavour in self.fitted_flavours:
                        flavour_idx = FLAVOUR_TO_ID_MAPPING[flavour]
                        # does not work when fitting more than one flavour
                        lhapdf_grid[flavour_idx] = pdf_grid

                return lhapdf_grid

        else:

            @jax.jit
            def pdf_func(params):
                """
                Parameters
                ----------
                pdf_grid: jnp.array
                    The PDF grid values, with shape (Nfl, Nx)
                """
                # split pdf_grid parameters from GP hyperparameters
                pdf_grid, _ = jnp.split(params, [self.n_parameters])

                # reshape pdf_grid to (Nfl, Nx)
                pdf_grid = pdf_grid.reshape(
                    len(self.fitted_flavours),
                    int(len(pdf_grid) / len(self.fitted_flavours)),
                )
                return pdf_grid

        return pdf_func

    def gp_kernel(self, hyperparameters, xgrid1, xgrid2, prior_settings):
        """ """
        if prior_settings["type"] == "gibbs_kernel_prior":
            return gibbs_kernel(hyperparameters, xgrid1, xgrid2)

    def gp_mean(self, hyperparameters, xgrid, prior_settings):
        """ """
        if prior_settings["type"] == "gibbs_kernel_prior":
            return jnp.zeros(len(xgrid))
