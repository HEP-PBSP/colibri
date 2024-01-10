"""
TODO
"""

import jax.numpy as jnp

from validphys import convolution

from super_net.constants import XGRID
from super_net.utils import FLAVOURS_ID_MAPPINGS


def closure_test_central_pdf_grid(closure_test_pdf, xgrids, Q0=1.65):
    """
    Computes the central member of the closure_test_pdf grid in the 
    evolution basis and only on x points that are specified in xgrids. 
    The grid is then interpolated to the full XGRID.

    NOTE: this function overrides the one in super_net.utils

    Parameters
    ----------
    closure_test_pdf: validphys.core.PDF

    xgrids: dict
        defines the reduced grid, keys are flavour names and values are x values.
        Each flavour needs to have a reduced grid assigned to in xgrids and
        all flavours need to have the same number of x values.

    Q0: float, default is 1.65

    Returns
    -------
    grid: jnp.array
        grid, is N_fl x N_x
    """

    # Every flavour (even unused ones like photon) needs to have a reduced grid assigned to in xgrids,
    # the flavour selection/mapping is then done by flavour_mapping (flavour_indices)

    reduced_xgrid = jnp.array(
        [
            convolution.evolution.grid_values(
                closure_test_pdf, [fl], x_vals, [Q0]
            ).squeeze(-1)[0].squeeze(0)
            for fl, x_vals in xgrids.items()
        ],
    )
    
    interpolated_xgrid = jnp.zeros(
        (reduced_xgrid.shape[0], len(XGRID))
    )

    
    for fl_idx in range(reduced_xgrid.shape[1]):
        interpolated_xgrid = interpolated_xgrid.at[fl_idx, :].set(
            jnp.interp(
                jnp.array(XGRID),
                jnp.array(xgrids[FLAVOURS_ID_MAPPINGS[fl_idx]]),
                reduced_xgrid[fl_idx, :],
            )
        )
    
    return interpolated_xgrid