"""Plotting utilities."""

import os
import re
import json
import numpy as np
import pandas as pd
import logging

import corner
import matplotlib.pyplot as plt
from reportengine.figure import figure, figuregen

from colibri.plots_and_tables.fit_reader import get_fit_path, csv_file_reader

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


@figuregen
def plot_pdf_from_csv_colibrifit(
    colibri_fits, underlyinglaw=None, load_pdf_model=True, xscale="log"
):
    """

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
    # TODO: implement underlyinglaw
    # check that pdf model is the same for all fits
    # TODO

    # nested dictionary, first key is the colibri fit, second key is the flavour
    dict_posterior_samples = {}

    # save relevant posterior samples into dictionary
    for colibri_fit_dict in colibri_fits:
        colibri_fit = colibri_fit_dict["id"]

        colibri_fit_label = colibri_fit_dict["label"]

        csv_info = csv_file_reader(colibri_fit, load_pdf_model=load_pdf_model)

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

            ax.set_xscale(xscale)
            ax.legend()

        yield fig
