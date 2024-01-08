"""
wmin.wmin_model.py

Module containing functions defining the weight minimisation parameterisation.

Author: Mark N. Costantini
Date: 11.11.2023
"""

import jax
import jax.numpy as jnp

from dataclasses import dataclass, asdict

from validphys import convolution

from super_net.constants import XGRID
from wmin.checks import check_wminpdfset_is_montecarlo
from wmin.wmin_utils import weights_initializer_provider

from super_net.pdf_model import PDFModel

class WMinPDF(PDFModel):

    def __init__(self, weight_minimization_grid, weight_minimization_prior, n_replicas_wmin):
        self.weight_minimization_grid = weight_minimization_grid
        self.weight_minimization_prior = weight_minimization_prior
        self.n_replicas_wmin = n_replicas_wmin

    @property
    def param_names(self):
        return [f"w_{i+1}" for i in range(self.n_replicas_wmin)]

    @property
    def init_params(self):
        return self.weight_minimization_grid.init_wmin_weights

    @property
    def bayesian_prior(self):
        return self.weight_minimization_prior

    def grid_values(self, params):
        wmin_weights = jnp.concatenate((jnp.array([1.0]), params))
        grid = jnp.einsum(
            "i,ijk", wmin_weights, self.weight_minimization_grid.wmin_INPUT_GRID
        )
        return grid

def model(
    weight_minimization_grid,
    weight_minimization_prior,
    n_replicas_wmin,
    ):
    return WMinPDF(weight_minimization_grid, weight_minimization_prior, n_replicas_wmin)

@dataclass(frozen=True)
class WeightMinimizationGrid:
    """
    Represents a weight minimization grid used for a weight minimisation fit.

    Args
    ----
    INPUT_GRID: jnp.array
        PDF grid of the Monte Carlo set used as wmin basis (evolution basis)

    wmin_INPUT_GRID: jnp.array
        grid used for the weight minimisation fit. Defined in such a way as to fulfill the sum rules automatically.

    init_wmin_weights: jnp.array
        initial random weights used in wmin parametrisation

    wmin_basis_idx: jnp.array
        index(s) (of INPUT_GRID replicas) of the weight minimisation basis excluding the
        replica that is used as central replica in the wmin parametrisation

    wmin_central_replica: jnp.array
        index of the central weight minimisation replica.

    """

    INPUT_GRID: jnp.array
    wmin_INPUT_GRID: jnp.array
    init_wmin_weights: jnp.array
    wmin_basis_idx: jnp.array
    wmin_central_replica: jnp.array

    def to_dict(self):
        return asdict(self)


def weight_minimization_grid(
    wminpdfset,
    wmin_grid_seed,
    random_wmin_parametrisation=False,
    n_replicas_wmin=50,
    Q0=1.65,
    weights_initializer="zeros",
    weights_seed=0xABCDEF,
    uniform_minval=-0.1,
    uniform_maxval=0.1,
):
    """
    Weight minimization grid is in the evolution basis.
    The following parametrization is used:

    f_{j,wm} = f_j + sum_i(w_i * (f_i - f_j))

    this has the advantage of automatically satisfying the sum rules.

    Notes:
        - the central replica of the wminpdfset is always included in the
          wmin parametrization
        - the replicas to be used in the parametrization are chosen at random
          within this function if random_wmin_parametrisation is True

    Parameters
    ----------
    wminpdfset: validphys.core.PDF

    weights_initializer_provider: super_net.wmin_utils.weights_initializer_provider
        Function responsible for the initialization of the weights in a weight minimization fit

    n_replicas_wmin: int
        number of replicas from wminpdfset to be used in the weight
        minimization parametrization

    Q0: float, default is 1.65
        scale at which wmin parametrization is done.

    wmin_grid_seed: jax.random.PRNGKey
        wmin_grid_seed is responsible for random wmin parametrisation and random choice of central wmin replica

    Returns
    -------
    WeightMinimizationGrid dataclass

    """

    # dimensions here are N_rep x N_fl x N_x
    INPUT_GRID = jnp.array(
        convolution.evolution.grid_values(
            wminpdfset, convolution.FK_FLAVOURS, XGRID, [Q0]
        ).squeeze(-1)
    )

    if n_replicas_wmin + 1 > INPUT_GRID.shape[0]:
        raise (
            f"n_replicas_wmin should be <= than the number of replicas contained in the PDF set {wminpdfset}"
        )

    if random_wmin_parametrisation:
        # reduce INPUT_GRID to only keep n_replicas_wmin PDF replicas
        wmin_basis_idx = jax.random.choice(
            wmin_grid_seed, INPUT_GRID.shape[0], shape=(n_replicas_wmin,), replace=False
        )
        # == generate weight minimization grid so that sum rules are automatically fulfilled == #
        # pick central wmin replica at random
        wmin_central_replica = jax.random.permutation(
            wmin_grid_seed, wmin_basis_idx, independent=True
        )[0]

        # discard central wmin replica from wmin basis
        wmin_basis_idx = wmin_basis_idx[jnp.where(wmin_basis_idx != wmin_central_replica)]

    else:
        # reduce INPUT_GRID to only keep n_replicas_wmin PDF replicas
        wmin_basis_idx = jnp.arange(1, n_replicas_wmin + 1)

        # == generate weight minimization grid so that sum rules are automatically fulfilled == #
        # pick central wmin replica as central replica from PDF set
        wmin_central_replica = 0

    wmin_INPUT_GRID = (
        INPUT_GRID[wmin_basis_idx, :, :] - INPUT_GRID[jnp.newaxis, wmin_central_replica]
    )

    wmin_INPUT_GRID = jnp.vstack(
        (INPUT_GRID[jnp.newaxis, wmin_central_replica], wmin_INPUT_GRID)
    )

    # initial weights for weight minimization
    init_wmin_weights = weights_initializer_provider(
        weights_initializer=weights_initializer,
        weights_seed=weights_seed,
        uniform_minval=uniform_minval,
        uniform_maxval=uniform_maxval,
    )(wmin_INPUT_GRID.shape[0] - 1)

    return WeightMinimizationGrid(
        INPUT_GRID=INPUT_GRID,
        wmin_INPUT_GRID=wmin_INPUT_GRID,
        init_wmin_weights=init_wmin_weights,
        wmin_basis_idx=wmin_basis_idx,
        wmin_central_replica=wmin_central_replica,
    )
