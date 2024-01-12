"""Plotting utilities."""

import matplotlib.pyplot as plt
from reportengine.figure import figure
import json
import corner
import numpy as np

import pathlib
from reportengine.figure import figuregen
import os, sys

import pandas as pd

def get_fit_path(fit):
    fit_path = pathlib.Path(sys.prefix) / "share/super_net/results" / fit
    if not os.path.exists(fit_path):
        raise FileNotFoundError("Could not find a fit " + fit + " in the super_net/results directory.")
    return str(fit_path)

@figuregen
def plot_histograms(super_net_fits):
    """Plots comparison histograms for the fits in super_net_fits
    """

    fit_dfs = []
    for fit in super_net_fits:
        fit_path = get_fit_path(fit)
        if os.path.exists(fit_path + '/ns_result.csv'):
            fit_dfs += [pd.read_csv(fit_path + '/ns_result.csv', index_col=0)]
        elif os.path.exists(fit_path + '/mc_result.csv'):
            fit_dfs += [pd.read_csv(fit_path + '/mc_result.csv', index_col=0)]
        else:
            raise FileNotFoundError('Could not find the results of an NS or MC fit for fit ' + fit)

    # Check that the parameters for the fits are all the same
    for fit_df in fit_dfs:
        if not all(fit_df.columns == fit_dfs[0].columns):
            raise ValueError('The supplied fits do not have the same parameters.')

    fig, ax = plt.subplots()
    params = fit_dfs[0].columns

    for param in params:
        for i, fit_df in enumerate(fit_dfs):
            ax.hist(fit_df[param], label=super_net_fits[i])
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

    fig, ax = plt.subplots()

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
