"""
Module to read the results of the fits
"""

import pathlib
import os, sys
import pandas as pd
import dill

import logging

log = logging.getLogger(__name__)



def get_fit_path(fit):
    fit_path = pathlib.Path(sys.prefix) / "share/colibri/results" / fit
    if not os.path.exists(fit_path):
        raise FileNotFoundError(
            "Could not find a fit " + fit + " in the colibri/results directory."
        )
    return str(fit_path)


def csv_file_reader(colibri_fit, load_pdf_model=False):
    """
    Given a colibri fit, returns the pandas dataframe with the results of the fit
    differentiating between Monte Carlo and Bayesian fits.

    Parameters
    ----------
    colibri_fit : str
        The name of the fit to read.
    
    load_pdf_model : bool, default=False
        Useful for grid pdf model

    Returns
    -------
    csv_file_info : dict
        A dictionary with the type of fit and the posterior samples.
    """

    fit_path = get_fit_path(colibri_fit)
    csv_file_info = {}

    if os.path.exists(fit_path + "/ns_result.csv"):
        log.info(f"Reading {fit_path}/ns_result.csv for a Bayesian fit.")
        df = pd.read_csv(fit_path + "/ns_result.csv", index_col=0)
        csv_file_info["type"] = "ns"
        csv_file_info["posterior_samples"] = df
        

    elif os.path.exists(fit_path + "/mc_result.csv"):
        log.info(f"Reading {fit_path}/mc_result.csv for a Monte Carlo fit.")
        df = pd.read_csv(fit_path + "/mc_result.csv", index_col=0)
        csv_file_info["type"] = "mc"
        csv_file_info["posterior_samples"] = df
    
    elif os.path.exists(fit_path + "/analytic_result.csv"):
        log.info(f"Reading {fit_path}/analytic_result.csv for an analytic fit.")
        df = pd.read_csv(fit_path + "/analytic_result.csv", index_col=0)
        csv_file_info["type"] = "analytic"
        csv_file_info["posterior_samples"] = df
        
    else:
        raise FileNotFoundError(
            "Could not find the results of an NS or MC fit for fit " + colibri_fit
        )
    
    if load_pdf_model:
        if not os.path.exists(fit_path + "/pdf_model.pkl"):
            raise FileNotFoundError(
                "Could not find the pdf model for fit " + colibri_fit
            )
        
        with open(fit_path + "/pdf_model.pkl", "rb") as file:
            pdf_model = dill.load(file)
            
        csv_file_info["pdf_model"] = pdf_model

    return csv_file_info
