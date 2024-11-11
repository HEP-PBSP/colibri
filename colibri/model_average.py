"""
colibri.model_average.py

Module containing functions for performing Bayesian model average.
"""

import os
import logging
import dill

import numpy as np
import pandas as pd

from colibri.utils import resample_from_ns_posterior
from colibri.export_results import write_exportgrid
from colibri.constants import LHAPDF_XGRID, EXPORT_LABELS

from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

log = logging.getLogger()


def selected_fits(fits, delta_logz=26.6):
    """
    Select a subset, A, of the models the set of models is defined as

    A = {M_k : log(Zk) >= log(Zmax) - delta_logz}

    where delta_logz can be chosen, for instance, to be a value from Jeffreys scales.

    Parameters
    ----------
    fits: list
        list of ColibriFitSpecs

    delta_logz: float
        Model selection tolerance constant

    Returns
    -------
    list
        a list of ColibriFitSpecs with len <= of the original
    """
    log_z_max = np.max([fit.bayesian_metrics["logz"] for fit in fits])
    selected_models = [
        fit for fit in fits if (fit.bayesian_metrics["logz"] >= log_z_max - delta_logz)
    ]
    return selected_models


def selected_fits_with_weights(selected_fits):
    """
    If ∆ is the quantity of interest, or parameters of interest, then when performing model
    average we can compute its posterior distribution given data D as

    p(∆|D) = sum_k p(D|∆,M_k) p(M_k|D)

    - M_k are the different models over which we average.
    - p(M_k|D) is the posterior model probability

    Assuming that the prior probability is the same for each model we get

    p(M_k|D) = p(D|M_k) / (sum_l p(D|M_l))

    hence multiplying by 1 = exp(-log(Z_avg))/exp(-log(Z_avg)) we get

    p(D|M_k) = exp(log(Z_k) - log(Z_avg)) / (1 + sum_{l != 1} exp(log(Z_l)-log(Z_avg)))

    """
    logz_values = np.array([fit.bayesian_metrics["logz"] for fit in selected_fits])
    mean_logZ = np.mean(logz_values)

    for fit, logz in zip(selected_fits, logz_values):
        p_k = np.exp(logz - mean_logZ) / (np.sum(np.exp(logz_values - mean_logZ)))

        fit.bayesian_metrics["bayesian_weight"] = p_k
    import IPython

    IPython.embed()
    return selected_fits


def model_combination(selected_fits_with_weights, n_samples):
    """ """
    # get fraction of number of replicas for each fit
    for fit in selected_fits_with_weights:
        n_frac_samples = int(fit.bayesian_metrics["bayesian_weight"] * n_samples)
        fit.bayesian_metrics["n_frac_samples"] = n_frac_samples
