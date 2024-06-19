"""
grid_pdf.model.py

The grid_pdf model.
"""

import jax
import jax.numpy as jnp

from validphys import convolution

from colibri.pdf_model import PDFModel


class GridPDFModel(PDFModel):
    """A PDFModel implementation for the grid_pdf module."""

    xgrids: dict
    param_names: list

    name = "grid_pdf PDF model"

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

    def grid_values_func(self, interpolation_grid, float_type=None):
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
            return jnp.array(interpolants, dtype=float_type)

        return interp_func
