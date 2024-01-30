"""
The grid_pdf model.
"""
import jax
import jax.numpy as jnp

from validphys import convolution
from validphys.core import PDF

from super_net.pdf_model import PDFModel

def pdf_model(flavour_xgrids):
    return GridPDFModel(flavour_xgrids)

def bayesian_prior(pdf_model, prior_settings):

    if prior_settings['type'] == 'uniform_pdf_prior':
        pdf = PDF(prior_settings['pdf_prior'])
        nsigma = prior_settings['nsigma']

        replicas_grid = jnp.concatenate([
            convolution.evolution.grid_values(
                pdf,
                [flavour],
                pdf_model.xgrids[flavour],
                [1.65]
            )
        for flavour in pdf_model.fitted_flavours], axis=2)

        central_prior_grid = replicas_grid[0,:,:,:].squeeze()
        # Remove central replica
        replicas_grid = replicas_grid[1:,:,:,:]

        error68_up = jnp.nanpercentile(replicas_grid, 84.13, axis=0).reshape(-1)
        error68_down = jnp.nanpercentile(replicas_grid, 15.87, axis=0).reshape(-1)

        # This is needed to avoid ultranest crashing with
        # ValueError: Buffer dtype mismatch, expected 'float_t' but got 'float'
        jax.config.update("jax_enable_x64", True)

        # Compute the band for a generic sigma_pdf_prior
        delta = (error68_up - error68_down) / 2
        mean = (error68_up + error68_down) / 2
        error_up = mean + delta * nsigma
        error_down = mean - delta * nsigma

        @jax.jit
        def prior_transform(cube):
            params = error_down + (error_up - error_down) * cube
            return params

        return prior_transform

class GridPDFModel(PDFModel):
    """A PDFModel implementation for the grid_pdf module.
    """
    xgrids: dict
    grid_prior: dict
    grid_init: dict
    param_names: list

    def __init__(self, flavour_xgrids):
        self.xgrids = flavour_xgrids

    @property
    def param_names(self):
        """The fitted parameters of the model.
        """
        return [f"{fl}({x})" for fl in self.fitted_flavours for x in self.xgrids[fl]]

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
            for i, flavour in enumerate(convolution.FK_FLAVOURS):
                if flavour in self.fitted_flavours:
                    interpolants += [jnp.interp(
                        jnp.array(interpolation_grid),
                        jnp.array(self.xgrids[flavour]),
                        jnp.array(params[:len(self.xgrids[flavour])]),
                    )]
                else:
                    interpolants += [jnp.array([0.0]*len(interpolation_grid))]
                params = params[len(self.xgrids[flavour]):]
            return jnp.array(interpolants)

        return interp_func
