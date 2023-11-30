"""
A plotting module for the weight-minimisation fits.
"""

import sys
import pathlib
import os
import json
import corner

from reportengine.figure import figuregen, figure

import matplotlib.pyplot as plt
import numpy as np

color_key = ['#66C2A5', '#FC8D62', '#8DA0CB']

def get_fit_mc_weights_path(fit):
    fit_path = pathlib.Path(sys.prefix) / "share/super_net/results" / fit
    if not os.path.exists(fit_path):
        raise FileNotFoundError("Could not find a fit " + fit + " in the super_net/results directory.")

    mc_weights_path = fit_path / "monte_carlo_results/monte_carlo_results.json"
    if not os.path.exists(mc_weights_path):
        raise FileNotFoundError("Could not find monte_carlo_results for " + fit + ".")

    return mc_weights_path

def get_mc_fits_info(super_net_fits):
    mc_weights = {}
    for fit in super_net_fits:
        mc_weights_path = get_fit_mc_weights_path(fit)

        with open(mc_weights_path) as f:
            d = json.load(f)

        d = np.array(d)
        nreps, nweights = d.shape
        mc_weights[fit] = {'nweights': nweights, 'nreps': nreps, 'weights': d.T}

    return mc_weights

def mc_weights(super_net_fits):
    return get_mc_fits_info(super_net_fits)

@figure
def plot_mc_weights_corner_plot(mc_weights):
    """
    TODO
    """

    fit_list = [fit for fit in mc_weights]
    fig = corner.corner(mc_weights[fit_list[0]]['weights'].T, color=color_key[0], labels=[rf"$w_{i+1}$" for i in range(mc_weights[fit_list[0]]['nweights'])])
    if len(fit_list) > 1:
        for i, fit in enumerate(fit_list[1:]):
            fig = corner.corner(mc_weights[fit]['weights'].T, fig=fig, color=color_key[i+1])
    
    return fig

@figuregen
def plot_mc_weights_histograms(mc_weights):
    """
    TODO
    """
    max_weights = max([mc_weights[fit]['nweights'] for fit in mc_weights]) 

    for i in range(max_weights):
       fig, ax = plt.subplots()
       for fit in mc_weights:
           if i < mc_weights[fit]['nweights']:
               ax.hist(mc_weights[fit]['weights'][i,:], label=fit, alpha=0.5)
           ax.set_ylabel("Frequency")
           ax.set_xlabel(rf"$w_{i+1}$")
           x_left, x_right = ax.get_xlim()
           y_low, y_high = ax.get_ylim()
           #set aspect ratio
           ax.set_aspect(abs((x_right-x_left)/(y_low-y_high)))
           ax.legend()

       yield fig
