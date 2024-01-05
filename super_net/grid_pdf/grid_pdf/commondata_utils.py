"""
grid_pdf.commondata_utils.py

Module containing commondata and central covmat index functions.

Author: Mark N. Costantini
Date: 05.02.2024
"""

import jax.numpy as jnp

from validphys import convolution

from super_net.constants import XGRID
from super_net.utils import FLAVOURS_ID_MAPPINGS
from super_net.commondata_utils import closuretest_commondata_tuple, pseudodata_commondata_tuple

from reportengine import collect


def closure_test_pdf_grid_interpolated(closure_test_pdf, xgrids, Q0=1.65):
    """
    Computes the closure_test_pdf grid in the evolution basis and only on
    x points that are specified in xgrids. The grid is then interpolated
    to the full XGRID.

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
        grid, is N_rep x N_fl x N_x
    """

    # Every flavour (even unused ones like photon) needs to have a reduced grid assigned to in xgrids,
    # the flavour selection/mapping is then done by flavour_mapping (flavour_indices)

    reduced_xgrid = jnp.concatenate(
        [
            convolution.evolution.grid_values(
                closure_test_pdf, [fl], x_vals, [Q0]
            ).squeeze(-1)
            for fl, x_vals in xgrids.items()
        ],
        axis=1,
    )

    interpolated_xgrid = jnp.zeros((reduced_xgrid.shape[0], reduced_xgrid.shape[1], len(XGRID)))

    for rep_idx in range(reduced_xgrid.shape[0]):

        for fl_idx in range(reduced_xgrid.shape[1]):

                interpolated_xgrid = interpolated_xgrid.at[rep_idx, fl_idx, :].set(
                    jnp.interp(
                        jnp.array(XGRID),
                        jnp.array(xgrids[FLAVOURS_ID_MAPPINGS[fl_idx]]),
                        reduced_xgrid[rep_idx, fl_idx, :],
                    )
                )
    
    return interpolated_xgrid 
    

def grid_pdf_closuretest_commondata_tuple(
    data,
    experimental_commondata_tuple,
    closure_test_pdf_grid_interpolated,
    flavour_indices=None,
):
    """
    Like super_net.commondata_utils.closuretest_commondata_tuple but for a 
    closure_test_pdf_grid_interpolated instead of a closure_test_pdf_grid.
    """
    return closuretest_commondata_tuple(data, experimental_commondata_tuple, closure_test_pdf_grid_interpolated, flavour_indices)


def grid_pdf_closuretest_pseudodata_commondata_tuple(data, grid_pdf_closuretest_commondata_tuple, replica_seed):
    """
    Like super_net.commondata_utils.pseudodata_commondata_tuple but for a grid_pdf_closuretest_commondata_tuple.
    """
    return pseudodata_commondata_tuple(data, grid_pdf_closuretest_commondata_tuple, replica_seed)


"""
Collect over multiple random seeds so as to generate multiple commondata instances.
To be used in a Monte Carlo closure test fit.
"""
collect_grid_pdf_closuretest_pseudodata_commondata_tuple = collect(
    "grid_pdf_closuretest_pseudodata_commondata_tuple",
    ("replica_indices",),
)