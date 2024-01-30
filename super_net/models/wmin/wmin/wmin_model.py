"""
wmin.wmin_model.py

Module containing functions defining the weight minimisation parameterisation.

"""

import jax
import jax.numpy as jnp

from validphys import convolution
from validphys.core import PDF

from super_net.pdf_model import PDFModel


def pdf_model(wmin_settings):
    return WMinPDF(PDF(wmin_settings["wminpdfset"]), wmin_settings["n_basis"])


class WMinPDF(PDFModel):
    def __init__(self, wminpdfset, n_basis):
        self.wminpdfset = wminpdfset
        self.n_basis = n_basis

    @property
    def param_names(self):
        return [f"w_{i+1}" for i in range(self.n_basis)]

    def grid_values_func(self, interpolation_grid):
        """This function should produce a grid values function, which takes
        in the model parameters, and produces the PDF values on the grid xgrid.
        """

        input_grid = jnp.array(
            convolution.evolution.grid_values(
                self.wminpdfset,
                convolution.FK_FLAVOURS,
                interpolation_grid,
                [1.65],
            ).squeeze(-1)
        )[: self.n_basis + 1, :, :]

        @jax.jit
        def interp_func(weights):
            weights = jnp.concatenate((jnp.array([1.0]), jnp.array(weights)))
            pdf = jnp.einsum("i,ijk", weights, input_grid)
            return pdf

        return interp_func


def mc_initial_parameters(pdf_model, mc_initialiser_settings, replica_index):
    if mc_initialiser_settings["type"] == "zeros":
        return [0.0] * pdf_model.n_basis


def bayesian_prior(prior_settings):
    if prior_settings["type"] == "uniform_parameter_prior":
        max_val = prior_settings["max_val"]
        min_val = prior_settings["min_val"]

        def prior_transform(cube):
            return cube * (max_val - min_val) + min_val

        return prior_transform
