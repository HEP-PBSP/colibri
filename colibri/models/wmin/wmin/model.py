"""
wmin.wmin_model.py

Module containing functions defining the weight minimisation parameterisation.

"""

import jax
import jax.numpy as jnp

from validphys import convolution
from validphys.core import PDF

from colibri.pdf_model import PDFModel

import logging

log = logging.getLogger(__name__)


def pdf_model(wmin_settings):
    """
    Weight minimization grid is in the evolution basis.
    The following parametrization is used:

    f_{j,wm} = f_j + sum_i(w_i * (f_i - f_j))

    this has the advantage of automatically satisfying the sum rules.

    Notes:
        - the central replica of the wminpdfset is always included in the
          wmin parametrization
    """

    return WMinPDF(PDF(wmin_settings["wminpdfset"]), wmin_settings["n_basis"])


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

    def grid_values_func(self, interpolation_grid, vectorized=False):
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

        Parameters
        ----------
        interpolation_grid: jnp.array
            The grid used in the fit.

        vectorized: bool, default is False
            Whether to use vectorized weights.

        Returns
        -------
        wmin_param: @jax.jit CompiledFunction
            The wmin parameterisation function.
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

        if vectorized:

            @jax.jit
            def wmin_param(weights):
                """
                Wmin parameterisation for vectorized weights.
                """
                wmin_weights = jnp.c_[jnp.ones(weights.shape[0]), weights]
                pdf = jnp.einsum("ri,ijk -> rjk", wmin_weights, wmin_input_grid)

                return pdf

        else:

            @jax.jit
            def wmin_param(weights):
                weights = jnp.concatenate((jnp.array([1.0]), jnp.array(weights)))
                pdf = jnp.einsum("i,ijk", weights, wmin_input_grid)
                return pdf

        return wmin_param


def mc_initial_parameters(pdf_model, mc_initialiser_settings, replica_index):
    """
    This function initialises the parameters for the weight minimisation
    in a Monte Carlo fit.

    Parameters
    ----------
    pdf_model: pdf_mode.PDFModel
        The PDF model to initialise the parameters for.

    mc_initialiser_settings: dict
        The settings for the initialiser.

    replica_index: int
        The index of the replica.

    Returns
    -------
    initial_values: jnp.array
        The initial values for the parameters.
    """
    if mc_initialiser_settings["type"] not in ("zeros", "normal", "uniform"):
        log.warning(
            f"MC initialiser type {mc_initialiser_settings['type']} not recognised, using default: 'zeros' instead."
        )

        mc_initialiser_settings["type"] = "zeros"

    if mc_initialiser_settings["type"] == "zeros":
        return jnp.array([0.0] * pdf_model.n_basis)

    random_seed = jax.random.PRNGKey(
        mc_initialiser_settings["random_seed"] + replica_index
    )

    if mc_initialiser_settings["type"] == "normal":
        # Currently, only one standard deviation around a zero mean is implemented
        initial_values = jax.random.normal(
            key=random_seed,
            shape=(pdf_model.n_basis,),
        )
        return initial_values

    if mc_initialiser_settings["type"] == "uniform":
        max_val = mc_initialiser_settings["max_val"]
        min_val = mc_initialiser_settings["min_val"]
        initial_values = jax.random.uniform(
            key=random_seed,
            shape=(pdf_model.n_basis,),
            minval=min_val,
            maxval=max_val,
        )
        return initial_values


def bayesian_prior(prior_settings):
    """
    Produces a prior transform function for the weight minimisation parameters.

    Parameters
    ----------
    prior_settings: dict
        The settings for the prior transform.

    Returns
    -------
    prior_transform: @jax.jit CompiledFunction
        The prior transform function.
    """
    if prior_settings["type"] == "uniform_parameter_prior":
        max_val = prior_settings["max_val"]
        min_val = prior_settings["min_val"]

        @jax.jit
        def prior_transform(cube):
            return cube * (max_val - min_val) + min_val

        return prior_transform
