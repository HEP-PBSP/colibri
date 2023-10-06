import jax
import jax.numpy as jnp

from dataclasses import dataclass

from validphys.convolution import evolution
from validphys import convolution

from super_net.constants import XGRID
from super_net.checks import check_wminpdfset_is_montecarlo
from super_net.wmin_utils import weights_initializer_provider


@dataclass(frozen=True)
class WeightMinimizationGrid:
    pass


@check_wminpdfset_is_montecarlo
def weight_minimization_grid(
    wminpdfset,
    n_replicas_wmin=50,
    Q0=1.65,
    rng_jax=0xDEAFBEEF,
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
          within this function

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

    rng_jax: seed
        Create a pseudo-random number generator (PRNG) key given an integer seed.


    Returns
    -------
    5D tuple:

        - INPUT_GRID: PDF grid of the Monte Carlo set used as wmin basis

        - wmin_INPUT_GRID: grid used for the weight minimisation fit. Defined in such a way
                           as to fulfill the sum rules automatically.

        - weight_base_num: initial random weights

        - wmin_basis_idx: index (of INPUT_GRID replicas) of the weight minimisation basis excluding the
                          replica that is used as central replica in the wmin parametrisation

        - rep1_idx: index of central wmin replica

    """

    rng = jax.random.PRNGKey(rng_jax)

    # dimensions here are N_rep x N_fl x N_x
    INPUT_GRID = evolution.grid_values(
        wminpdfset, convolution.FK_FLAVOURS, XGRID, [Q0]
    ).squeeze(-1)

    if n_replicas_wmin > INPUT_GRID.shape[0]:
        raise (
            f"n_replicas_wmin should be <= than the number of replicas contained in the PDF set {wminpdfset}"
        )

    # reduce INPUT_GRID to only keep n_replicas_wmin PDF replicas
    wmin_basis_idx = jax.random.choice(
        rng, INPUT_GRID.shape[0], shape=(n_replicas_wmin,), replace=False
    )

    # == generate weight minimization grid so that sum rules are automatically fulfilled == #
    # pick central wmin replica at random
    rep1_idx = jax.random.permutation(rng, wmin_basis_idx, independent=True)[0]

    # discard central wmin replica from wmin basis
    wmin_basis_idx = wmin_basis_idx[jnp.where(wmin_basis_idx != rep1_idx)]
    
    wmin_INPUT_GRID = INPUT_GRID[wmin_basis_idx,:,:] - INPUT_GRID[jnp.newaxis, rep1_idx]
    
    wmin_INPUT_GRID = jnp.vstack((INPUT_GRID[jnp.newaxis, rep1_idx], wmin_INPUT_GRID))

    # initial weights for weight minimization
    weight_base_num = weights_initializer_provider(
        weights_initializer=weights_initializer,
        weights_seed=weights_seed,
        uniform_minval=uniform_minval,
        uniform_maxval=uniform_maxval,
    )(wmin_INPUT_GRID.shape[0] - 1)
    # output should be dataclass
    return INPUT_GRID, wmin_INPUT_GRID, weight_base_num, wmin_basis_idx, rep1_idx
