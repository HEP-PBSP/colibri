"""Plotting utilities."""

import os
import re
import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import logging

import corner
import matplotlib.pyplot as plt
from reportengine.figure import figure, figuregen

from validphys import convolution
from validphys.core import PDF

from colibri.plots_and_tables.fit_reader import (
    get_fit_path,
    get_csv_file_posterior,
    get_pdf_model,
)
from colibri.constants import FLAVOUR_TO_ID_MAPPING, GRID_MAPPING

log = logging.getLogger(__name__)

color_key = ["#66C2A5", "#FC8D62", "#8DA0CB"]


@figure
def plot_corner(colibri_fits):
    """Plots comparison corner plot for the fits in colibri_fits"""
    fit_dfs = []

    for fit in colibri_fits:
        fit_path = get_fit_path(fit)
        if os.path.exists(fit_path + "/ns_result.csv"):
            fit_dfs += [pd.read_csv(fit_path + "/ns_result.csv", index_col=0)]
        elif os.path.exists(fit_path + "/mc_result.csv"):
            fit_dfs += [pd.read_csv(fit_path + "/mc_result.csv", index_col=0)]
        else:
            raise FileNotFoundError(
                "Could not find the results of an NS or MC fit for fit " + fit
            )

    # Check that the parameters for the fits are all the same
    for fit_df in fit_dfs:
        if len(fit_df.columns) != len(fit_dfs[0].columns):
            raise ValueError(
                "The supplied fits do not have the same number of parameters."
            )
        if not all(fit_df.columns == fit_dfs[0].columns):
            raise ValueError("The supplied fits do not have the same parameters.")

    fig = corner.corner(fit_dfs[0], color=color_key[0])
    if len(fit_dfs) > 1:
        for i in range(len(fit_dfs[1:])):
            fig = corner.corner(fit_dfs[i], fig=fig, color=color_key[i + 1])

    return fig


@figuregen
def plot_histograms(colibri_fits):
    """Plots comparison histograms for the fits in colibri_fits"""

    fit_dfs = []
    for fit in colibri_fits:
        fit_path = get_fit_path(fit)
        if os.path.exists(fit_path + "/ns_result.csv"):
            fit_dfs += [pd.read_csv(fit_path + "/ns_result.csv", index_col=0)]
        elif os.path.exists(fit_path + "/mc_result.csv"):
            fit_dfs += [pd.read_csv(fit_path + "/mc_result.csv", index_col=0)]
        else:
            raise FileNotFoundError(
                "Could not find the results of an NS or MC fit for fit " + fit
            )

    # Check that the parameters for the fits are all the same
    for fit_df in fit_dfs:
        if len(fit_df.columns) != len(fit_dfs[0].columns):
            raise ValueError(
                "The supplied fits do not have the same number of parameters."
            )
        if not all(fit_df.columns == fit_dfs[0].columns):
            raise ValueError("The supplied fits do not have the same parameters.")

    fig, ax = plt.subplots()
    params = fit_dfs[0].columns

    for param in params:
        for i, fit_df in enumerate(fit_dfs):
            ax.hist(fit_df[param], label=colibri_fits[i])
            ax.set_xlabel(param)
        yield fig
        ax.cla()


@figure
def plot_ultranest_results(ultranest_results_path):
    """
    Plots the results of a Bayesian fit performed using the
    Ultranest package.

    Parameters
    ----------
    ultranest_results_path: str
        Path to the output folder of an Ultranest fit.
    """

    fig, _ = plt.subplots()

    with open(
        ultranest_results_path + "/ultranest_results/ultranest_results.json"
    ) as f:
        results = json.loads(json.load(f))

    paramnames = results["paramnames"]
    data = np.array(results["samples"])

    fig = corner.corner(
        data,
        labels=paramnames,
        show_titles=True,
        quiet=True,
    )

    return fig


class ColibriFitsPlotter:

    def __init__(
        self,
        colibri_fit,
        underlyinglaw=None,
    ):

        self.colibri_fit = colibri_fit
        self.underlyinglaw = underlyinglaw

    @property
    def posterior_from_csv(self):
        """
        Reads the *result.csv file containing posterior samples of PDF parameters
        into a pandas dataframe.
        """
        return get_csv_file_posterior(self.colibri_fit["id"])

    @property
    def pdf_model(self):
        """
        Loads the .pkl file into an instance of pdf model class.
        """
        return get_pdf_model(self.colibri_fit["id"])

    def underlyinglaw_fl_grid(self, flavour, interp_grid):
        """
        TODO
        """
        # if interpolation grid is not empty return underlying law
        if len(interp_grid) != 0:
            underlyinglaw_fl = (
                convolution.evolution.grid_values(
                    PDF(self.underlyinglaw), [flavour], interp_grid, [1.65]
                )
                .squeeze(-1)[0]
                .squeeze(0)
            )

            return underlyinglaw_fl
        else:
            return []

    def pdf_values(self, flavour, interpolation_grid):
        """
        Returns a 2D array with shape Number of posterior samples and shape
        of interpolation_grid.

        Parameters
        ----------
        flavour: str

        Returns
        -------
        np.array
        """
        grid_values_func = self.pdf_model.grid_values_func(interpolation_grid)

        pdf_values = np.array(
            [
                grid_values_func(
                    params=self.posterior_from_csv.values[replica_index, :]
                )[FLAVOUR_TO_ID_MAPPING[flavour]]
                for replica_index in range(self.posterior_from_csv.shape[0])
            ]
        )

        return pdf_values

    def stats_68_cl(self, flavour, interpolation_grid, stats_68_cl_settings=None):
        """
        TODO
        """
        pdf_values = self.pdf_values(flavour, interpolation_grid)

        upper_band = np.nanpercentile(pdf_values, 84.13, axis=0)
        lower_band = np.nanpercentile(pdf_values, 15.87, axis=0)

        mean = pdf_values.mean(axis=0)

        if stats_68_cl_settings["spline_interpolation"]:
            interp_ub = interp1d(
                interpolation_grid,
                upper_band,
                **{"bounds_error": False, **stats_68_cl_settings["interp1d_settings"]},
            )
            interp_lb = interp1d(
                interpolation_grid,
                lower_band,
                **{"bounds_error": False, **stats_68_cl_settings["interp1d_settings"]},
            )
            interp_mean = interp1d(
                interpolation_grid,
                mean,
                **{"bounds_error": False, **stats_68_cl_settings["interp1d_settings"]},
            )

            if stats_68_cl_settings["type"] == "linear":
                x_new = np.linspace(
                    min(interpolation_grid),
                    max(interpolation_grid),
                    stats_68_cl_settings["n_new_points"],
                )
            elif stats_68_cl_settings["type"] == "log":
                x_new = np.logspace(
                    np.log10(min(interpolation_grid)),
                    np.log10(max(interpolation_grid)),
                    stats_68_cl_settings["n_new_points"],
                )

            upper_band = interp_ub(x_new)
            lower_band = interp_lb(x_new)
            mean = interp_mean(x_new)

            return x_new, upper_band, lower_band, mean

        return interpolation_grid, upper_band, lower_band, mean


@figuregen
def plot_pdf_from_csv_colibrifit(
    colibri_fits,
    underlyinglaw=None,
    flavours=None,
    interpolation_grid=None,
    xscale="log",
    stats_68_cl_settings=None,
):
    """

    Parameters
    ----------
    colibri_fits: list
        list of dict containing fit Id and label

    flavours: list, default is None
        when None all flavours are used

    interpolation_grid: str, default is None, has three possible options
        if None then grid_pdf model is assumed and the pdf_model.xgrids is used
        if xgrid: constants.XGRID is used
        if lhapdf: constants.LHAPDF_XGRID is used

    xscale: str, default is log
        can be either log or linear

    cubic_spline_interp: Bool, default is False
        whether to interpolate within xgrid values.
        Only works for grid_pdf model

    n_interp_points: int, default is 100
        number of interpolation points

    Yields
    ------
    matplotlib figure
    """
    if interpolation_grid and (interpolation_grid not in GRID_MAPPING):
        raise KeyError(
            f"interpolation_grid has to be set to either 'xgrid' or 'lhapdf_grid', if interpolation_grid is None, then pdf_model has to have 'xgrids' attribute"
        )

    # use all flavours per default
    if not flavours:
        flavours = FLAVOUR_TO_ID_MAPPING.keys()

    for fl in flavours:

        fig, ax = plt.subplots()

        for fit in colibri_fits:

            colibri_plotter = ColibriFitsPlotter(
                fit,
                underlyinglaw,
            )

            pdf_model = colibri_plotter.pdf_model

            # if interpolation grid is either 'grid' or 'lhapdf_grid' then take corresponding
            # grids in GRID_MAPPING, otherwise use the model xgrid (available for grid_pdf model only)
            interp_grid = GRID_MAPPING.get(interpolation_grid, pdf_model.xgrids[fl])

            interp_grid, upper_band, lower_band, mean = colibri_plotter.stats_68_cl(
                fl, interp_grid, stats_68_cl_settings
            )

            ax.plot(
                interp_grid,
                mean,
                linestyle="-",
                label=f"{fit['label']}, {fl}",
            )

            ax.fill_between(
                interp_grid,
                lower_band,
                upper_band,
                alpha=0.5,
            )

            # plot the underlying law only once
            if underlyinglaw and (fit == colibri_fits[0]):
                # this bit here is probably quite inefficient
                ax.plot(
                    interp_grid,
                    colibri_plotter.underlyinglaw_fl_grid(
                        fl,
                        interp_grid,
                    ),
                    linestyle="--",
                    label=f"Underlying law",
                )

            ax.set_xscale(xscale)
            ax.legend()

        yield fig


@figuregen
def plot_pdf_ratio_from_csv_colibrifit(
    colibri_fits,
    normalize_to,
    underlyinglaw=None,
    flavours=None,
    interpolation_grid=None,
    xscale="log",
    stats_68_cl_settings=None,
):
    """

    Parameters
    ----------
    colibri_fits: list
        list of dict containing fit Id and label

    flavour: list, default is None
        when None all flavours are used

    interpolation_grid: str, default is None, has three possible options
        if None then grid_pdf model is assumed and the pdf_model.xgrids is used
        if xgrid: constants.XGRID is used
        if lhapdf: constants.LHAPDF_XGRID is used

    xscale: str, default is log
        can be either log or linear

    cubic_spline_interp: Bool, default is False
        whether to interpolate within xgrid values.
        Only works for grid_pdf model

    n_interp_points: int, default is 100
        number of interpolation points

    Yields
    ------
    matplotlib figure
    """
    if interpolation_grid and (interpolation_grid not in GRID_MAPPING):
        raise KeyError(
            f"interpolation_grid has to be set to either 'xgrid' or 'lhapdf_grid', if interpolation_grid is None, then pdf_model has to have 'xgrids' attribute"
        )

    # get the fit id from the normalize_to parameter
    if isinstance(normalize_to, int):
        normalize_to = colibri_fits[normalize_to - 1]

    # use all flavours per default
    if not flavours:
        flavours = FLAVOUR_TO_ID_MAPPING.keys()

    # get normalize to pdf:
    colibri_plotter_normto = ColibriFitsPlotter(
        normalize_to,
        underlyinglaw,
    )

    pdf_model_normto = colibri_plotter_normto.pdf_model

    for fl in flavours:

        fig, ax = plt.subplots()

        normto_interp_grid = GRID_MAPPING.get(
            interpolation_grid, pdf_model_normto.xgrids[fl]
        )
        _, _, _, mean_normto = colibri_plotter_normto.stats_68_cl(
            fl,
            normto_interp_grid,
            stats_68_cl_settings,
        )

        for fit in colibri_fits:

            colibri_plotter = ColibriFitsPlotter(
                fit,
                underlyinglaw,
            )

            pdf_model = colibri_plotter.pdf_model

            # if interpolation grid is either 'grid' or 'lhapdf_grid' then take corresponding
            # grids in GRID_MAPPING, otherwise use the model xgrid (available for grid_pdf model only)
            interp_grid = GRID_MAPPING.get(interpolation_grid, pdf_model.xgrids[fl])

            interp_grid, upper_band, lower_band, mean = colibri_plotter.stats_68_cl(
                fl, interp_grid, stats_68_cl_settings
            )

            ax.plot(
                interp_grid,
                mean / mean_normto,
                linestyle="-",
                label=f"{fit['label']}, {fl}",
            )

            ax.fill_between(
                interp_grid,
                lower_band / mean_normto,
                upper_band / mean_normto,
                alpha=0.5,
            )

            # plot the underlying law only once
            if underlyinglaw and (fit == colibri_fits[0]):
                ax.plot(
                    interp_grid,
                    colibri_plotter.underlyinglaw_fl_grid(
                        fl,
                        (interp_grid),
                    )
                    / mean_normto,
                    linestyle="--",
                    label=f"Underlying law",
                )

            ax.set_xscale(xscale)
            ax.legend()

        yield fig
