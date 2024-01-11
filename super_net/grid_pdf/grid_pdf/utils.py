"""
TODO
"""

import jax.numpy as jnp

from validphys import convolution

import super_net
from super_net.constants import XGRID
from super_net.utils import FLAVOURS_ID_MAPPINGS


def closure_test_central_pdf_grid(
    closure_test_pdf, xgrids, length_reduced_xgrids, reduced_xgrid_data=True, Q0=1.65
):
    """
    Computes the central member of the closure_test_pdf grid in the
    evolution basis and only on x points that are specified in xgrids.
    The grid is then interpolated to the full XGRID.

    NOTE: when reduced_xgrid_data=True, this function overrides the one in super_net.utils
    otherwise the one in super_net.utils is used.

    Parameters
    ----------
    closure_test_pdf: validphys.core.PDF

    xgrids: dict
        defines the reduced grid, keys are flavour names and values are x values.
        All flavours need to have the same number of x values.
        Flavours with no x values are assigned a zero grid.

    length_reduced_xgrids: int
        lenght of the reduced xgrids
    
    reduced_xgrid_data: bool, default is True
        When True the closure_test_central_pdf_grid is overriden.
        When False the closure_test_pdf_grid from super_net.utils is used.
    
    Q0: float, default is 1.65

    Returns
    -------
    grid: jnp.array
        grid, is N_fl x N_x
    """
    
    if not reduced_xgrid_data:
        return super_net.utils.closure_test_pdf_grid(closure_test_pdf, Q0=Q0)[0]
    
    # fill with zeros the xgrids that are not used
    xgrids = {
        fl: jnp.array(x_vals) if x_vals else jnp.zeros(length_reduced_xgrids)
        for fl, x_vals in xgrids.items()
    }

    # the flavour selection/mapping is then done by flavour_mapping (flavour_indices)
    reduced_xgrid = jnp.array(
        [
            convolution.evolution.grid_values(closure_test_pdf, [fl], x_vals, [Q0])
            .squeeze(-1)[0]
            .squeeze(0)
            for fl, x_vals in xgrids.items()
        ],
    )

    interpolated_xgrid = jnp.zeros((reduced_xgrid.shape[0], len(XGRID)))

    for fl_idx in range(reduced_xgrid.shape[1]):
        interpolated_xgrid = interpolated_xgrid.at[fl_idx, :].set(
            jnp.interp(
                jnp.array(XGRID),
                jnp.array(xgrids[FLAVOURS_ID_MAPPINGS[fl_idx]]),
                reduced_xgrid[fl_idx, :],
            )
        )

    return interpolated_xgrid
