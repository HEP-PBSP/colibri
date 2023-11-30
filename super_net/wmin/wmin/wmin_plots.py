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

@figuregen
def plot_mc_weights_histograms(super_net_fit):
    """
    TODO
    """

    fit = super_net_fit    

    fit_path = pathlib.Path(sys.prefix) / "share/super_net/results" / fit
    if not os.path.exists(fit_path):
        raise FileNotFoundError("Could not find a fit " + fit + " in the super_net/results directory.")

    mc_weights_path = fit_path / "monte_carlo_results/monte_carlo_results.json"
    if not os.path.exists(mc_weights_path):
        raise FileNotFoundError("Could not find monte_carlo_results for " + fit + ".")

    with open(mc_weights_path) as f:
        d = json.load(f)

    d = np.array(d)
    d = d.T

    for i, col in enumerate(d):
       fig, ax = plt.subplots()
       ax.hist(col)
       ax.set_ylabel("Frequency")
       ax.set_xlabel(rf"$w_{i+1}$")

       yield fig    
