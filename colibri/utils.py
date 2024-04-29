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
import pathlib
import sys
import os
import pandas as pd


def fill_dis_fkarr_with_zeros(fktable, FIT_XGRID):
    """
    Fills the FK array with zeros so as to get array of shape
    (Ndat, Nfl, N_FIT_XGRID)

    Parameters
    ----------
    fktable: validphys.coredata.FKTableData

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    Returns
    -------
    new_fkarr: np.ndarray
    """

    new_fkarr = np.zeros(
        (
            fktable.get_np_fktable().shape[0],
            fktable.get_np_fktable().shape[1],
            len(FIT_XGRID),
        )
    )
    indices = np.where(np.isclose(FIT_XGRID, fktable.xgrid[:, np.newaxis]))[1]
    new_fkarr[:, :, indices] = fktable.get_np_fktable()

    return new_fkarr


def fill_had_fkarr_with_zeros(fktable, FIT_XGRID):
    """
    Fills the FK array with zeros so as to get array of shape
    (Ndat, Nfl, N_FIT_XGRID, N_FIT_XGRID)

    Parameters
    ----------
    fktable: validphys.coredata.FKTableData

    Returns
    -------
    new_fkarr: np.ndarray
    """

    new_fkarr = np.zeros(
        (
            fktable.get_np_fktable().shape[0],
            fktable.get_np_fktable().shape[1],
            len(FIT_XGRID),
            len(FIT_XGRID),
        )
    )

    indices = np.where(np.isclose(FIT_XGRID, fktable.xgrid[:, np.newaxis]))[1]
    new_fkarr[:, :, indices[:, None], indices] = fktable.get_np_fktable()

    return new_fkarr


def t0_pdf_grid(t0pdfset, FIT_XGRID, Q0=1.65):
    """
    Computes the t0 pdf grid in the evolution basis.

    Parameters
    ----------
    t0pdfset: validphys.core.PDF

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    Q0: float, default is 1.65

    Returns
    -------
    t0grid: jnp.array
        t0 grid, is N_rep x N_fl x N_x
    """

    t0grid = jnp.array(
        convolution.evolution.grid_values(
            t0pdfset, convolution.FK_FLAVOURS, FIT_XGRID, [Q0]
        ).squeeze(-1)
    )
    return t0grid


def closure_test_pdf_grid(closure_test_pdf, FIT_XGRID, Q0=1.65):
    """
    Computes the closure_test_pdf grid in the evolution basis.

    Parameters
    ----------
    closure_test_pdf: validphys.core.PDF

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    Q0: float, default is 1.65

    Returns
    -------
    grid: jnp.array
        grid, is N_rep x N_fl x N_x
    """

    grid = jnp.array(
        convolution.evolution.grid_values(
            closure_test_pdf, convolution.FK_FLAVOURS, FIT_XGRID, [Q0]
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


def get_fit_path(fit):
    fit_path = pathlib.Path(sys.prefix) / "share/colibri/results" / fit
    if not os.path.exists(fit_path):
        raise FileNotFoundError(
            "Could not find a fit " + fit + " in the colibri/results directory."
        )
    return str(fit_path)


def get_full_posterior(colibri_fit):
    """
    Given a colibri fit, returns the pandas dataframe with the results of the fit
    at the parameterisation scale.

    Parameters
    ----------
    colibri_fit : str
        The name of the fit to read.


    Returns
    -------
    pandas dataframe
    """

    fit_path = get_fit_path(colibri_fit)

    csv_path = fit_path + "/full_posterior_sample.csv"
    # check that file exist
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Could not find the full posterior sample for the fit " + colibri_fit
        )

    df = pd.read_csv(csv_path, index_col=0)

    return df
