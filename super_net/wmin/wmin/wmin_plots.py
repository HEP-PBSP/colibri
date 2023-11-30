"""
A plotting module for the weight-minimisation fits.
"""

import sys
import pathlib
import os
import json
import corner
import yaml

from reportengine.figure import figuregen, figure
from validphys.core import PDF

import matplotlib.pyplot as plt
import numpy as np

color_key = ['#66C2A5', '#FC8D62', '#8DA0CB']

def print_parametrisations(super_net_fits):
    """
    Prints the parametric form that is used in each of the super_net_fits.
    """

    for fit in super_net_fits:
        runcard = get_fit_runcard(fit)

    if len(super_net_fits) == 1:
        text = "This report displays the super_net fit performed in the parametrisation:\n\n"
    else:
        text = "This report compares super_net fits performed in the following " + str(len(super_net_fits)) + " parametrisations:\n\n"

    for fit in super_net_fits:
        runcard = get_fit_runcard(fit)
        wminpdfset = runcard['wminpdfset']
        n_replicas_wmin = runcard['n_replicas_wmin']
        use_same_wmin_param_per_replica = runcard['use_same_wmin_param_per_replica']
        if 'random_wmin_parametrisation' in runcard.keys():
            random_wmin_parametrisation = runcard[random_wmin_parametrisation]
        else:
            random_wmin_parametrisation = False

        text += "- " + fit + " has the parametrisation: " 

        if not random_wmin_parametrisation:
            if n_replicas_wmin == 2:
                text += f"<div style=\"text-align:center\"> $\\textbf{{f}}(\\textbf{{w}}) = \\textbf{{f}}_0 + w_1 (\\textbf{{f}}_1 - \\textbf{{f}}_0) + w_{n_replicas_wmin} (\\textbf{{f}}_{n_replicas_wmin} - \\textbf{{f}}_0),$ </div>"

            else:
                text += f"<div style=\"text-align:center\"> $\\textbf{{f}}(\\textbf{{w}}) = \\textbf{{f}}_0 + w_1 (\\textbf{{f}}_1 - \\textbf{{f}}_0) + \cdots + w_{n_replicas_wmin} (\\textbf{{f}}_{n_replicas_wmin} - \\textbf{{f}}_0),$ </div>"

            text += " where $\\textbf{f}_k$ is the $k$th replica drawn from the PDF set " + wminpdfset + " and $\\textbf{f}_0$ is the central replica of the same PDF set.\n\n"

        else:
            if use_same_wmin_param_per_replica:
                if n_replicas_wmin == 2:
                    text += f"<div style=\"text-align:center\"> $\\textbf{{f}}(\\textbf{{w}}) = \\textbf{f}_j + w_1 (\\textbf{{f}}_1 - \\textbf{{f}}_j) + w_{n_replicas_wmin} (\\textbf{{f}}_{n_replicas_wmin} - \\textbf{{f}}_j),$ </div>"

                else:
                    text += f"<div style=\"text-align:center\"> $\\textbf{{f}}(\\textbf{{w}}) = \\textbf{f}_j + w_1 (\\textbf{{f}}_1 - \\textbf{{f}}_j) + \cdots + w_{n_replicas_wmin} (\\textbf{{f}}_{n_replicas_wmin} - \\textbf{{f}}_j),$ </div>"

                text += " where $\\textbf{f}_k$ is the $k$th replica drawn from the PDF set " + wminpdfset + " and $\\textbf{f}_j$ is a fixed replica, drawn at random from the same PDF set.\n\n"

            else:
                if n_replicas_wmin == 2:
                    text += f"<div style=\"text-align:centre\"> $\\textbf{{f}}(\\\textbf{{w}}) = \\textbf{{f}}_{{j_i}} + w_1 (\\textbf{{f}}_1 - \\textbf{{f}}_{{j_i}}) + w_{n_replicas_wmin} (\\textbf{{f}}_{n_replcas_wmin} - \\textbf{{f}}_{{j_i}}),$ </div>"
                else:
                    text += f"<div style=\"text-align:centre\"> $\\textbf{{f}}(\\\textbf{{w}}) = \\textbf{{f}}_{{j_i}} + w_1 (\\textbf{{f}}_1 - \\textbf{{f}}_{{j_i}}) + \cdots + w_{n_replicas_wmin} (\\textbf{{f}}_{n_replcas_wmin} - \\textbf{{f}}_{{j_i}}),$ </div>"

                text += " where $\\textbf{f}_k$ is the $k_j$th replica drawn from the PDF set " + wminpdfset + " and $\\textbf{f}_{j_i}$ is the $j_i$th replica drawn from the same PDF set randomly, for the $i$th Monte Carlo replica considered in the fit.\n\n"

    return text

def get_fit_path(fit):
    fit_path = pathlib.Path(sys.prefix) / "share/super_net/results" / fit
    if not os.path.exists(fit_path):
        raise FileNotFoundError("Could not find a fit " + fit + " in the super_net/results directory.")

    return fit_path

def get_fit_runcard(fit):
    fit_path = get_fit_path(fit)
    yaml_path = fit_path / "input/runcard.yaml"
    if not os.path.exists(yaml_path):
        raise FileNotFoundError("Could not find runcard for fit " + fit + " in the super_net/results directory.")

    with open(yaml_path, 'r') as file:
        fit_runcard = yaml.safe_load(file)

    return fit_runcard

def get_fit_mc_weights_path(fit):
    fit_path = get_fit_path(fit)
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

def plot_pdf_pulls(super_net_fits):
    """
    Plots the directions in which the PDFs pull for each of a collection of super_net_fits.
    """

    for fit in super_net_fits:
        fit_data = get_fit_runcard(fit)

        wminpdfset = fit_data['wminpdfset']
        nweights = fit_data['n_replicas_wmin'] 

        pdf = PDF(wminpdfset)
#        print(pdf)
#        print(dir(pdf))
