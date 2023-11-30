"""
A plotting module for the weight-minimisation fits.
"""

import sys
import pathlib
import os
import json

from reportengine.figure import figuregen

import matplotlib.pyplot as plt
import numpy as np

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

@figuregen
def plot_mc_weights_histograms(super_net_fits):
    """
    TODO
    """

    mc_weights = get_mc_fits_info(super_net_fits)
    max_weights = max([mc_weights[fit]['nweights'] for fit in mc_weights]) 

    for i in range(max_weights):
       fig, ax = plt.subplots()
       for fit in mc_weights:
           if i < mc_weights[fit]['nweights']:
               ax.hist(mc_weights[fit]['weights'][i,:], label=fit, alpha=0.5)
           ax.set_ylabel("Frequency")
           ax.set_xlabel(rf"$w_{i+1}$")
           ax.legend()

       yield fig
