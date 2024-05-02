"""
An executable to perform bayesian model selection on a set of fits.
"""

import argparse
import logging

from reportengine import colors
import os
from colibri.utils import get_pdf_model, get_fit_path
from colibri.api import API as colibri_api
import jax
import yaml
import json
import pandas as pd

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(colors.ColorHandler())


def check_chi2_functions(
    chi2_functions, pdf_model1, post_samples, FIT_XGRID, pred_data
):
    test_pred, _ = pdf_model1.pred_and_pdf_func(FIT_XGRID, forward_map=pred_data)(
        post_samples[0].values[0]
    )

    chi2_1 = chi2_functions[0](test_pred)
    chi2_2 = chi2_functions[1](test_pred)

    if chi2_1 != chi2_2:
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Script to compare fits.")

    parser.add_argument("fit_names", nargs=2, help="List of colibri fits to compare.")

    args = parser.parse_args()

    # Load the pdf models
    pdf_model1, pdf_model2 = [get_pdf_model(fit) for fit in args.fit_names]

    # Get fit paths
    fit_paths = [get_fit_path(fit) for fit in args.fit_names]

    # Read runcard.yaml in input folder
    runcard_path = fit_paths[0] + "/input/runcard.yaml"
    with open(runcard_path, "r") as file:
        runcard_1 = yaml.safe_load(file)

    runcard_path = fit_paths[1] + "/input/runcard.yaml"
    with open(runcard_path, "r") as file:
        runcard_2 = yaml.safe_load(file)

    # Check that the key dataset_inputs is the same
    if runcard_1["dataset_inputs"] != runcard_2["dataset_inputs"]:
        raise ValueError(
            "The dataset_inputs are not the same. Model comparison would be meaningless."
        )

    # Read full_posterior_sample.csv
    post_samples = [
        pd.read_csv(fit_path + "/full_posterior_sample.csv", index_col=0)
        for fit_path in fit_paths
    ]

    # Produce chi2 functions
    chi2_functions = [
        colibri_api.make_chi2(**runcard) for runcard in [runcard_1, runcard_2]
    ]

    # Produce FIT_XGRID and pred_data
    FIT_XGRID = colibri_api.FIT_XGRID(**runcard_1)
    pred_data = colibri_api.make_pred_data(**runcard_1)

    # Check that chi2 functions are the same, i.e. for the same pdf and predictions
    # they produce the same chi2. This is important for the Bayes factor to be meaningful.
    if not check_chi2_functions(
        chi2_functions, pdf_model1, post_samples, FIT_XGRID, pred_data
    ):
        raise ValueError(
            "The chi2 functions are not the same. Model comparison would be meaningless."
        )

    # Compute average chi2 for each fit
    avg_chi2 = []
    for pdf_model, chi2, post_sample in zip(
        [pdf_model1, pdf_model2], chi2_functions, post_samples
    ):
        pred_func = pdf_model.pred_and_pdf_func(FIT_XGRID, forward_map=pred_data)
        pred_func = jax.vmap(pred_func, in_axes=(0,), out_axes=(0, 0))
        pred = pred_func(post_sample.values)[0]
        avg_chi2.append(jax.vmap(chi2)(pred).mean())
    log.info(f"Average chi2 for fit 1: {avg_chi2[0]}")
    log.info(f"Average chi2 for fit 2: {avg_chi2[1]}")

    # Load results.json or evidence.csv for each fit
    logz = []
    max_logl = []
    for fit_path in fit_paths:
        if not os.path.exists(fit_path + "/ultranest_logs/info/results.json"):
            if os.path.exists(fit_path + "/evidence.csv"):
                with open(fit_path + "/evidence.csv", "r") as file:
                    logz.append(pd.read_csv(file)["LogZ"].values[0])
            else:
                raise FileNotFoundError(
                    "Could not find the results.json nor the evidence.csv file for the fit "
                    + fit_path
                )
        else:
            results_path = fit_path + "/ultranest_logs/info/results.json"
            with open(results_path, "r") as file:
                results = json.load(file)
                logz.append(results["logz"])
                max_logl.append(results["maximum_likelihood"]["logl"])

    log.info(f"LogZ for fit 1: {logz[0]}")
    log.info(f"LogZ for fit 2: {logz[1]}")

    # Compute the Bayes factor
    log_bayes_factor = logz[0] - logz[1]
    log.info(f"Log Bayes factor: {log_bayes_factor}")

    # Interpret the Bayes factor
    if log_bayes_factor > 0:
        if log_bayes_factor > 5:
            log.info("The first model is strongly favored.")
        elif log_bayes_factor > 2.5:
            log.info("The first model is moderately favored.")
        elif log_bayes_factor > 1:
            log.info("The first model is weakly favored.")
        else:
            log.info("The evidence test is inconclusive.")
    else:
        if log_bayes_factor < -5:
            log.info("The second model is strongly favored.")
        elif log_bayes_factor < -2.5:
            log.info("The second model is moderately favored.")
        elif log_bayes_factor < -1:
            log.info("The second model is weakly favored.")
        else:
            log.info("The evidence test is inconclusive.")

    # Compute the bayesian complexity
    Cb1 = avg_chi2[0] + 2 * max_logl[0]
    Cb2 = avg_chi2[1] + 2 * max_logl[1]
    log.info(f"Bayesian complexity for fit 1: {Cb1}")
    log.info(f"Bayesian complexity for fit 2: {Cb2}")
