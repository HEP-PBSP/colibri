import jax
import jax.numpy as jnp

from validphys.convolution import evolution
from validphys import convolution

from super_net.constants import XGRID
from super_net.checks import check_wminpdfset_is_montecarlo





@check_wminpdfset_is_montecarlo
def weight_minimization_grid(wminpdfset, n_replicas_wmin=50, Q0=1.65, rng_jax=0xDEAFBEEF):
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
    wminpdfset : 'validphys.core.PDF'

    n_replicas_wmin : int
                number of replicas from wminpdfset to be used in the weight
                minimization parametrization

    Q0 : float, default is 1.65
        scale at which wmin parametrization is done.

    rng_jax : 
    
    Returns
    -------

    TODO
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
    random_replicas = jax.random.choice(
        rng, INPUT_GRID.shape[0], shape=(n_replicas_wmin,), replace=False
    )
    
    # include central replica
    if not jnp.any(random_replicas == 0):
        random_replicas = jnp.append(random_replicas, 0)
    
    # shuffle random replicas
    random_replicas = jax.random.permutation(rng, random_replicas, independent=True)
    
    INPUT_GRID = INPUT_GRID[random_replicas]

    # generate weight minimization grid so that sum rules are automatically fulfilled

    rep1_idx = jax.random.choice(rng, INPUT_GRID.shape[0])
    wmin_INPUT_GRID = (
        jnp.delete(INPUT_GRID, rep1_idx, axis=0) - INPUT_GRID[jnp.newaxis, rep1_idx]
    )
    wmin_INPUT_GRID = jnp.vstack((INPUT_GRID[jnp.newaxis, rep1_idx], wmin_INPUT_GRID))

    # initial weights for weight minimization
    
    weight_base_num = jnp.zeros(INPUT_GRID.shape[0] - 1) #jax.random.normal(rng, shape=(INPUT_GRID.shape[0] - 1,))

    return INPUT_GRID, wmin_INPUT_GRID, weight_base_num



