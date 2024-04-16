"""
colibri.utils.py

Module containing several utils for PDF fits.

Author: Mark N. Costantini
Date: 11.11.2023
"""

import jax
import jax.numpy as jnp
import numpy as np
from validphys import convolution

from colibri.constants import XGRID


def fill_dis_fkarr_with_zeros(fktable):
    """
    Fills the FK array with zeros so as to get array of shape
    (Ndat, Nfl, 50)
    """

    new_fkarr = np.zeros(
        (
            fktable.get_np_fktable().shape[0],
            fktable.get_np_fktable().shape[1],
            len(XGRID),
        )
    )
    indices = np.where(np.isclose(np.array(XGRID), fktable.xgrid[:, np.newaxis]))[1]
    new_fkarr[:, :, indices] = fktable.get_np_fktable()

    return new_fkarr


def fill_had_fkarr_with_zeros(fktable):
    """
    Fills the FK array with zeros so as to get array of shape
    (Ndat, Nfl, 50, 50)
    """

    new_fkarr = np.zeros(
        (
            fktable.get_np_fktable().shape[0],
            fktable.get_np_fktable().shape[1],
            len(XGRID),
            len(XGRID),
        )
    )

    indices = np.where(np.isclose(np.array(XGRID), fktable.xgrid[:, np.newaxis]))[1]
    new_fkarr[:, :, indices[:, None], indices] = fktable.get_np_fktable()

    return new_fkarr


def t0_pdf_grid(t0pdfset, Q0=1.65):
    """
    Computes the t0 pdf grid in the evolution basis.

    Parameters
    ----------
    t0pdfset: validphys.core.PDF

    Q0: float, default is 1.65

    Returns
    -------
    t0grid: jnp.array
        t0 grid, is N_rep x N_fl x N_x
    """

    t0grid = jnp.array(
        convolution.evolution.grid_values(
            t0pdfset, convolution.FK_FLAVOURS, XGRID, [Q0]
        ).squeeze(-1)
    )
    return t0grid


def closure_test_pdf_grid(closure_test_pdf, Q0=1.65):
    """
    Computes the closure_test_pdf grid in the evolution basis.

    Parameters
    ----------
    closure_test_pdf: validphys.core.PDF

    Q0: float, default is 1.65

    Returns
    -------
    grid: jnp.array
        grid, is N_rep x N_fl x N_x
    """

    grid = jnp.array(
        convolution.evolution.grid_values(
            closure_test_pdf, convolution.FK_FLAVOURS, XGRID, [Q0]
        ).squeeze(-1)
    )
    return grid


def resample_from_ns_posterior(
    samples, n_posterior_samples=1000, posterior_resampling_seed=123456
):
    """
    TODO
    """

    current_samples = samples.copy()

    rng = jax.random.PRNGKey(posterior_resampling_seed)

    resampled_samples = jax.random.choice(
        rng, current_samples, (n_posterior_samples,), replace=False
    )

    return resampled_samples


def closure_test_central_pdf_grid(closure_test_pdf_grid):
    """
    Returns the central replica of the closure test pdf grid.
    """
    return closure_test_pdf_grid[0]
