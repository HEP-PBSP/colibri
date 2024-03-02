"""Plotting utilities."""

import os
import re
import json
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
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
from colibri.constants import FLAVOUR_TO_ID_MAPPING, XGRID, LHAPDF_XGRID

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

    def underlyinglaw_fl_grid(self, flavour, interpolation_grid):
        """
        TODO
        """
        underlyinglaw_fl = (
            convolution.evolution.grid_values(
                PDF(self.underlyinglaw), [flavour], interpolation_grid, [1.65]
            )
            .squeeze(-1)[0]
            .squeeze(0)
        )
        return underlyinglaw_fl

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

    def stats_68_cl(self, flavour, interpolation_grid):
        """
        TODO
        """
        pdf_values = self.pdf_values(flavour, interpolation_grid)

        upper_band = np.nanpercentile(pdf_values, 84.13, axis=0)
        lower_band = np.nanpercentile(pdf_values, 15.87, axis=0)

        mean = pdf_values.mean(axis=0)

        return upper_band, lower_band, mean

    def spline_interpolator(self, flavour, interpolation_grid):
        """
        TODO
        """

        upper_band, lower_band, mean = self.stats_68_cl(flavour, interpolation_grid)

        # do some interpolation stuff

        return upper_band, lower_band, mean
        # cs_mean = CubicSpline(x_vals, mean)
        # cs_upper = CubicSpline(x_vals, upper_band)
        # cs_lower = CubicSpline(x_vals, lower_band)

        # mean, upper_band, lower_band = (
        #     cs_mean(x_new),
        #     cs_upper(x_new),
        #     cs_lower(x_new),
        # )


@figuregen
def plot_pdf_from_csv_colibrifit(
    colibri_fits,
    underlyinglaw=None,
    flavours=None,
    interpolation_grid=None,
    xscale="log",
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

    # use all flavours per default
    if not flavours:
        flavours = FLAVOUR_TO_ID_MAPPING.keys()

    # choose interpolation grid
    if interpolation_grid == "xgrid":
        interpolation_grid = XGRID
    elif interpolation_grid == "lhapdf":
        interpolation_grid = LHAPDF_XGRID

    for fl in flavours:

        fig, ax = plt.subplots()

        for fit in colibri_fits:

            colibri_plotter = ColibriFitsPlotter(
                fit,
                underlyinglaw,
            )

            pdf_model = colibri_plotter.pdf_model

            upper_band, lower_band, mean = colibri_plotter.spline_interpolator(
                fl, interpolation_grid if interpolation_grid else pdf_model.xgrids[fl]
            )

            x_vals = interpolation_grid if interpolation_grid else pdf_model.xgrids[fl]

            ax.plot(
                x_vals,
                mean,
                linestyle="-",
                label=f"{fit['label']}, {fl}",
            )

            ax.fill_between(
                x_vals,
                lower_band,
                upper_band,
                alpha=0.5,
            )

            # plot the underlying law only once
            if underlyinglaw and (fit == colibri_fits[0]):
                ax.plot(
                    interpolation_grid if interpolation_grid else pdf_model.xgrids[fl],
                    colibri_plotter.underlyinglaw_fl_grid(
                        fl,
                        (
                            interpolation_grid
                            if interpolation_grid
                            else pdf_model.xgrids[fl]
                        ),
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
    # get the fit id from the normalize_to parameter
    if isinstance(normalize_to, int):
        normalize_to = colibri_fits[normalize_to - 1]

    # use all flavours per default
    if not flavours:
        flavours = FLAVOUR_TO_ID_MAPPING.keys()

    # choose interpolation grid
    if interpolation_grid == "xgrid":
        interpolation_grid = XGRID
    elif interpolation_grid == "lhapdf":
        interpolation_grid = LHAPDF_XGRID

    # get normalize to pdf:
    colibri_plotter_normto = ColibriFitsPlotter(
        normalize_to,
        underlyinglaw,
    )

    pdf_model_normto = colibri_plotter_normto.pdf_model

    for fl in flavours:

        fig, ax = plt.subplots()

        upper_band_normto, lower_band_normto, mean_normto = (
            colibri_plotter_normto.spline_interpolator(
                fl,
                (
                    interpolation_grid
                    if interpolation_grid
                    else pdf_model_normto.xgrids[fl]
                ),
            )
        )
        for fit in colibri_fits:

            colibri_plotter = ColibriFitsPlotter(
                fit,
                underlyinglaw,
            )

            pdf_model = colibri_plotter.pdf_model

            upper_band, lower_band, mean = colibri_plotter.spline_interpolator(
                fl, interpolation_grid if interpolation_grid else pdf_model.xgrids[fl]
            )

            x_vals = interpolation_grid if interpolation_grid else pdf_model.xgrids[fl]

            ax.plot(
                x_vals,
                mean / mean_normto,
                linestyle="-",
                label=f"{fit['label']}, {fl}",
            )

            ax.fill_between(
                x_vals,
                lower_band / lower_band_normto,
                upper_band / upper_band_normto,
                alpha=0.5,
            )

            # plot the underlying law only once
            if underlyinglaw and (fit == colibri_fits[0]):
                ax.plot(
                    interpolation_grid if interpolation_grid else pdf_model.xgrids[fl],
                    colibri_plotter.underlyinglaw_fl_grid(
                        fl,
                        (
                            interpolation_grid
                            if interpolation_grid
                            else pdf_model.xgrids[fl]
                        ),
                    )
                    / mean_normto,
                    linestyle="--",
                    label=f"Underlying law",
                )

            ax.set_xscale(xscale)
            ax.legend()

        yield fig


@figuregen
def plot_gridpdf_from_csv_colibrifit(colibri_fits, underlyinglaw=None, xscale="log"):
    """
    Plots the grid pdf from the csv file of a colibri fit.
    The fits must have been performed using the grid pdf model.

    Parameters
    ----------
    colibri_fits : list
        List of colibri fits to plot.

    underlyinglaw : str, default is None
        The underlying law used to generate data in the fits.

    xscale : str, default is 'log'
        The scale of the x-axis. Can be 'log' or 'linear'.

    yields
    ------
    fig : matplotlib.figure.Figure
        A figure object with the plot.
    """
    # load the pdf model from the first fit
    pdf_model = csv_file_reader(colibri_fits[0]["id"], load_pdf_model=True)["pdf_model"]

    # get the underlying law values on the xgrid points of the pdf model of the first fit
    underlyinglaw_dict = {}
    for fl in pdf_model.xgrids.keys():
        x_vals = pdf_model.xgrids[fl]
        if x_vals:

            underlyinglaw_dict[fl] = (
                convolution.evolution.grid_values(
                    PDF(underlyinglaw), [fl], x_vals, [1.65]
                )
                .squeeze(-1)[0]
                .squeeze(0)
            )

    # nested dictionary, first key is the colibri fit, second key is the flavour
    dict_posterior_samples = {}

    # save relevant posterior samples into dictionary
    for colibri_fit_dict in colibri_fits:
        colibri_fit = colibri_fit_dict["id"]

        colibri_fit_label = colibri_fit_dict["label"]

        csv_info = csv_file_reader(colibri_fit, load_pdf_model=True)

        dict_posterior_samples[colibri_fit] = {"label": colibri_fit_label}

        for fl in csv_info["pdf_model"].fitted_flavours:
            cols = [
                col
                for col in csv_info["posterior_samples"].columns
                if col.startswith(fl)
            ]

            dict_posterior_samples[colibri_fit][fl] = csv_info["posterior_samples"][
                cols
            ]

    # loop over flavours and fits
    for fl in csv_info["pdf_model"].fitted_flavours:

        fig, ax = plt.subplots()

        for colibri_fit in colibri_fits:

            # get the posterior samples for the fit and flavour
            df = dict_posterior_samples[colibri_fit["id"]][fl]

            upper_band = np.nanpercentile(df.values, 84.13, axis=0)
            lower_band = np.nanpercentile(df.values, 15.87, axis=0)

            mean = df.values.mean(axis=0)

            # get x labels and round them
            pattern = r"\d+\.\d+"
            x_labels = [float(re.findall(pattern, x)[0]) for x in df.iloc[0].index]

            ax.plot(
                x_labels,
                mean,
                linestyle="-",
                label=f"{dict_posterior_samples[colibri_fit['id']]['label']}, {fl}",
            )

            ax.fill_between(
                x_labels,
                lower_band,
                upper_band,
                alpha=0.5,
            )

            # plot the underlying law only once
            if underlyinglaw and colibri_fit == colibri_fits[0]:
                ax.plot(
                    x_labels,
                    underlyinglaw_dict[fl],
                    linestyle="--",
                    label=f"Underlying law",
                )

            ax.set_xscale(xscale)
            ax.legend()

        yield fig


@figuregen
def plot_ratio_gridpdf_from_csv_colibrifit(
    colibri_fits, normalize_to, underlyinglaw=None, xscale="log"
):
    """
    Plots the grid pdf ratio from the csv file of a colibri fit.
    The fits must have been performed using the grid pdf model.

    Parameters
    ----------
    colibri_fits : list
        List of colibri fits to plot.

    normalize_to : str or int
        can be either a string with the fit id or an integer with the index of the fit in colibri_fits.

    underlyinglaw : str, default is None
        The underlying law used to generate data in the fits.

    xscale : str, default is 'log'
        The scale of the x-axis. Can be 'log' or 'linear'.

    yields
    ------
    fig : matplotlib.figure.Figure
        A figure object with the plot.
    """
    # get the fit id from the normalize_to parameter
    if isinstance(normalize_to, int):
        normalize_to = colibri_fits[normalize_to]["id"]

    # load the pdf model from the first fit
    pdf_model = csv_file_reader(colibri_fits[0]["id"], load_pdf_model=True)["pdf_model"]

    # get the underlying law values on the xgrid points of the pdf model of the first fit
    underlyinglaw_dict = {}
    for fl in pdf_model.xgrids.keys():
        x_vals = pdf_model.xgrids[fl]
        if x_vals:

            underlyinglaw_dict[fl] = (
                convolution.evolution.grid_values(
                    PDF(underlyinglaw), [fl], x_vals, [1.65]
                )
                .squeeze(-1)[0]
                .squeeze(0)
            )

    # nested dictionary, first key is the colibri fit, second key is the flavour
    dict_posterior_samples = {}

    # save relevant posterior samples into dictionary
    for colibri_fit_dict in colibri_fits:
        colibri_fit = colibri_fit_dict["id"]

        colibri_fit_label = colibri_fit_dict["label"]

        csv_info = csv_file_reader(colibri_fit, load_pdf_model=True)

        dict_posterior_samples[colibri_fit] = {"label": colibri_fit_label}

        for fl in csv_info["pdf_model"].fitted_flavours:
            cols = [
                col
                for col in csv_info["posterior_samples"].columns
                if col.startswith(fl)
            ]

            dict_posterior_samples[colibri_fit][fl] = csv_info["posterior_samples"][
                cols
            ]

    # loop over flavours and fits
    for fl in csv_info["pdf_model"].fitted_flavours:

        # get the posterior samples for the fit and flavour to normalize to
        mean_normalize_to = dict_posterior_samples[normalize_to][fl].mean(axis=0)

        fig, ax = plt.subplots()

        for colibri_fit in colibri_fits:

            # get the posterior samples for the fit and flavour
            df = dict_posterior_samples[colibri_fit["id"]][fl] / mean_normalize_to

            upper_band = np.nanpercentile(df.values, 84.13, axis=0)
            lower_band = np.nanpercentile(df.values, 15.87, axis=0)

            mean = df.values.mean(axis=0)

            # get x labels and round them
            pattern = r"\d+\.\d+"
            x_labels = [float(re.findall(pattern, x)[0]) for x in df.iloc[0].index]

            ax.plot(
                x_labels,
                mean,
                linestyle="-",
                label=f"{dict_posterior_samples[colibri_fit['id']]['label']}, {fl}",
            )

            ax.fill_between(
                x_labels,
                lower_band,
                upper_band,
                alpha=0.5,
            )

            # plot the underlying law only once
            if underlyinglaw and colibri_fit == colibri_fits[0]:
                ax.plot(
                    x_labels,
                    underlyinglaw_dict[fl] / mean_normalize_to,
                    linestyle="--",
                    label=f"Underlying law",
                )

            ax.set_xscale(xscale)
            ax.legend()

        yield fig
