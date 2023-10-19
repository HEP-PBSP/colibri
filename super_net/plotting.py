"""Plotting utilities."""

import matplotlib.pyplot as plt
from reportengine.figure import figure
import json
import corner
import numpy as np

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

    with open(ultranest_results_path + '/ultranest_results.json') as f:
        results = json.loads(json.load(f))

    paramnames = results['paramnames']
    data = np.array(results['weighted_samples']['points'])
    weights = np.array(results['weighted_samples']['weights'])
    cumsumweights = np.cumsum(weights)

    mask = cumsumweights > 1e-4

    fig = corner.corner(data[mask,:], weights=weights[mask], labels=paramnames, show_titles=True, quiet=True)

    return fig 
