"""
wmin.wmin_model.py

Module containing functions defining the weight minimisation parameterisation.

"""

import jax
import jax.numpy as jnp

from validphys import convolution

from colibri.pdf_model import PDFModel

import logging

log = logging.getLogger(__name__)


class WMinPDF(PDFModel):
    """
    A PDFModel implementation for the wmin parameterisation.

    Attributes
    ----------
    wminpdfset: validphys.core.PDF
        The PDF set to use for the wmin parameterisation.

    n_basis: int
        The number of basis functions to use for the wmin parameterisation.

    """

    def __init__(self, wminpdfset, n_basis):
        self.wminpdfset = wminpdfset
        self.n_basis = n_basis

    @property
    def param_names(self):
        return [f"w_{i+1}" for i in range(self.n_basis)]

    def grid_values_func(self, interpolation_grid):
        """
        This function should produce a grid values function, which takes
        in the model parameters, and produces the PDF values on the grid xgrid.

        Weight minimization grid is in the evolution basis.
        The following parametrization is used:

        f_{j,wm} = f_j + sum_i(w_i * (f_i - f_j))

        this has the advantage of automatically satisfying the sum rules.

        Notes:
            - the central replica of the wminpdfset is always included in the
                wmin parametrization

        """

        input_grid = jnp.array(
            convolution.evolution.grid_values(
                self.wminpdfset,
                convolution.FK_FLAVOURS,
                interpolation_grid,
                [1.65],
            ).squeeze(-1)
        )

        if self.n_basis + 1 > input_grid.shape[0]:
            raise ValueError(
                "The number of basis functions is larger than the number of replicas in the wminpdfset."
            )

        # reduce INPUT_GRID to only keep n_replicas_wmin PDF replicas
        wmin_basis_idx = jnp.arange(1, self.n_basis + 1)

        # == generate weight minimization grid so that sum rules are automatically fulfilled == #
        # pick central wmin replica as central replica from PDF set
        wmin_central_replica = 0

        # build wmin input grid so that sum rules are automatically fulfilled
        wmin_input_grid = (
            input_grid[wmin_basis_idx, :, :]
            - input_grid[jnp.newaxis, wmin_central_replica]
        )

        wmin_input_grid = jnp.vstack(
            (input_grid[jnp.newaxis, wmin_central_replica], wmin_input_grid)
        )

        @jax.jit
        def wmin_param(weights):
            weights = jnp.concatenate((jnp.array([1.0]), jnp.array(weights)))
            pdf = jnp.einsum("i,ijk", weights, wmin_input_grid)
            return pdf

        return wmin_param
