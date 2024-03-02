"""
Module to read the results of the fits
"""

import pathlib
import os, sys
import pandas as pd
import dill
import glob

import logging

log = logging.getLogger(__name__)


def get_fit_path(fit):
    fit_path = pathlib.Path(sys.prefix) / "share/colibri/results" / fit
    if not os.path.exists(fit_path):
        raise FileNotFoundError(
            "Could not find a fit " + fit + " in the colibri/results directory."
        )
    return str(fit_path)


def get_csv_file_posterior(colibri_fit):
    """
    Given a colibri fit, returns the pandas dataframe with the results of the fit
    at the parameterisation scale.

    Parameters
    ----------
    colibri_fit : str
        The name of the fit to read.


    Returns
    -------
    pandas dataframe
    """

    fit_path = get_fit_path(colibri_fit)

    if not glob.glob(fit_path + "/*result.csv"):
        raise FileNotFoundError("Could not find the csv results of fit " + colibri_fit)

    csv_path = glob.glob(fit_path + "/*result.csv")[0]
    df = pd.read_csv(csv_path, index_col=0)
    
    return df


def get_pdf_model(colibri_fit):
    """
    Given a colibri fit, returns the stored pdf model class.

    Parameters
    ----------
    colibri_fit : str
        The name of the fit to read.


    Returns
    -------
    pdf model class
    """

    fit_path = get_fit_path(colibri_fit)

    if not os.path.exists(fit_path + "/pdf_model.pkl"):
        raise FileNotFoundError(
            "Could not find the pdf model for fit " + colibri_fit
        )

    with open(fit_path + "/pdf_model.pkl", "rb") as file:
        pdf_model = dill.load(file)

    return pdf_model
