"""
gp.model.py

The Gaussian process model.
"""

import jax
import jax.numpy as jnp

from validphys import convolution

from colibri.pdf_model import PDFModel


class GpPDFModel(PDFModel):
    """A PDFModel implementation for the Gaussian process module."""

    xgrids: dict
    param_names: list

    name = "Gaussian process PDF model"

    def __init__(self, flavour_xgrids):
        self.xgrids = flavour_xgrids

    @property
    def param_names(self):
        """The parameters of the model."""
        return [f"{fl}({x})" for fl in self.fitted_flavours for x in self.xgrids[fl]]

    @property
    def n_parameters(self):
        """The number of parameters of the model."""
        return len(self.param_names)

    @property
    def fitted_flavours(self):
        """The fitted flavours used in the model, in STANDARDISED order,
        according to convolution.FK_FLAVOURS.
        """
        flavours = []
        for flavour in convolution.FK_FLAVOURS:
            if self.xgrids[flavour]:
                flavours += [flavour]
        return flavours

    def grid_values_func(self, xgrid):
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
