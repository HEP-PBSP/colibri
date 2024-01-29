"""
grid_pdf.utils.py

Module containing util functions for grid PDF fits.

Author: Luca Mantani
Date: 18.12.2023
"""

import logging
from datetime import datetime
import jax
import jax.numpy as jnp

from validphys import convolution

import super_net
from super_net.constants import XGRID
from super_net.utils import FLAVOURS_ID_MAPPINGS
from grid_pdf.grid_pdf_model import pdf_prior_grid

log = logging.getLogger(__name__)


def closure_test_central_pdf_grid(
    closure_test_pdf,
    xgrids,
    length_reduced_xgrids,
    flavour_indices,
    reduced_xgrid_data=False,
    Q0=1.65,
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
        Defines the reduced grid, keys are flavour names and values are x values.
        All flavours need to have the same number of x values.
        Flavours with no x values are assigned a zero grid.

    length_reduced_xgrids: int
        lenght of the reduced xgrids

    flavour_indices: list
        Specifies the ids of the flavours to include in a fit.

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

    # the flavour selection/mapping is then done by flavour_mapping (flavour_indices)
    reduced_pdfgrid = jnp.array(
        [
            (
                convolution.evolution.grid_values(closure_test_pdf, [fl], x_vals, [Q0])
                .squeeze(-1)[0]
                .squeeze(0)
                if x_vals
                else jnp.zeros(length_reduced_xgrids)
            )
            for fl, x_vals in xgrids.items()
        ],
    )

    interpolated_pdfgrid = jnp.zeros((reduced_pdfgrid.shape[0], len(XGRID)))

    for fl_idx in flavour_indices:
        interpolated_pdfgrid = interpolated_pdfgrid.at[fl_idx, :].set(
            jnp.interp(
                jnp.array(XGRID),
                jnp.array(xgrids[FLAVOURS_ID_MAPPINGS[fl_idx]]),
                reduced_pdfgrid[fl_idx, :],
            )
        )
    return interpolated_pdfgrid


def gridpdf_fit_name(set_name=None):
    if set_name:
        return set_name
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return f"{current_time}_grid_fit"


def pdf_prior_grid_initializer(
    grid_initializer, reduced_xgrids, flavour_indices, length_stackedpdf, replica_index
):
    """
    Returns the pdf prior grid initializer.
    """
    init_pdf = pdf_prior_grid(
        grid_initializer["pdf_prior"], reduced_xgrids, flavour_indices
    )

    # if init_type is central, return the central pdf
    if grid_initializer["init_type"] == "central":
        return init_pdf.stacked_pdf_grid_prior

    # if init_type is uniform, return a random pdf
    # with a uniform distribution around the central pdf
    elif grid_initializer["init_type"] == "uniform":
        rng = jax.random.PRNGKey(replica_index)

        sigma_pdf_init = grid_initializer["sigma_pdf_init"]

        error68_up = init_pdf.error68_up
        error68_down = init_pdf.error68_down

        # Compute the delta between the central and the upper/lower error
        delta = (error68_up - error68_down) / 2

        # Generate a random number between -sigma_pdf_init and sigma_pdf_init
        epsilon = jax.random.uniform(
            rng,
            shape=(length_stackedpdf,),
            minval=-sigma_pdf_init,
            maxval=sigma_pdf_init,
        )

        return init_pdf.stacked_pdf_grid_prior + epsilon * delta


def init_stacked_pdf_grid(
    grid_initializer,
    length_stackedpdf,
    replica_index,
    reduced_xgrids,
    flavour_indices,
    multiple_initiators=False,
):
    if grid_initializer["type"] == "zeros":
        return jnp.zeros(shape=length_stackedpdf)

    elif grid_initializer["type"] == "uniform":
        rng = jax.random.PRNGKey(replica_index)

        return jax.random.uniform(
            rng,
            shape=(length_stackedpdf,),
            minval=grid_initializer["minval"],
            maxval=grid_initializer["maxval"],
        )

    elif grid_initializer["type"] == "pdf":

        if not multiple_initiators:
            init_grid = pdf_prior_grid_initializer(
                grid_initializer,
                reduced_xgrids,
                flavour_indices,
                length_stackedpdf,
                replica_index,
            )
            return init_grid

        # if multiple_initiators is True, return a list of multiple initializers
        rng = jax.random.PRNGKey(replica_index)
        multiple_init_grid = [
            pdf_prior_grid_initializer(
                grid_initializer, reduced_xgrids, flavour_indices, length_stackedpdf, k
            )
            for k in jax.random.randint(rng, minval=0, maxval=1000, shape=(100,))
        ]
        return multiple_init_grid
