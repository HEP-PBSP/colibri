"""
gp.model.py

The Gaussian process model.
"""

import jax
import jax.numpy as jnp

from colibri.pdf_model import PDFModel
from colibri.constants import XGRID


class GpPDFModel(PDFModel):
    """
    A PDFModel implementation for the Gaussian process module.

    Attributes
    ----------

    """

    gp_xgrid: list = XGRID
    fitted_flavours: list
    param_names: list
    gp_hyperparams_settings: dict

    name = "Gaussian process PDF model"

    def __init__(self, gp_xgrid, fitted_flavours, gp_hyperparams_settings):
        self.gp_xgrid = gp_xgrid
        self.fitted_flavours = fitted_flavours
        self.gp_hyperparams_settings = gp_hyperparams_settings

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
        if not hasattr(self, "gp_xgrid"):
            raise AttributeError("The model is missing the 'xgrid' attribute.")
        if not hasattr(self, "gp_hyperparams_settings"):
            raise AttributeError(
                "The model is missing the 'gp_hyperparams_settings' attribute."
            )

        # Generate parameter names for the grid values
        parameter_names.extend(
            f"{fl}({x})" for fl in self.fitted_flavours for x in self.gp_xgrid
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
        return int(len(self.fitted_flavours) * len(self.gp_xgrid))

    @property
    def n_hyperparameters(self):
        """
        The number of hyperparameters of the model.
        """
        n_hyperparams = 0
        for _, hyperparams in self.gp_hyperparams_settings.items():
            n_hyperparams += len(hyperparams)
        return n_hyperparams

    def grid_values_func(self, gp_xgrid):
        """This function should produce a grid values function, which takes
        in the model parameters, and produces the PDF values on the grid xgrid.
        """

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
