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


def produce_test_predictions(pdf_model, runcard, fit_path):
    # Read full_posterior_sample.csv
    post_sample = pd.read_csv(fit_path + "/full_posterior_sample.csv", index_col=0)
    # Produce FIT_XGRID and pred_data
    FIT_XGRID = colibri_api.FIT_XGRID(**runcard)
    pred_data = colibri_api.make_pred_data(**runcard)

    test_pred, _ = pdf_model.pred_and_pdf_func(FIT_XGRID, forward_map=pred_data)(
        post_sample.values[0]
    )

    return test_pred


def check_chi2_functions(chi2_functions, predictions):

    chi2_1 = chi2_functions[0](predictions)
    chi2_2 = chi2_functions[1](predictions)

    if chi2_1 != chi2_2:
        return False

    return True


def compute_average_chi2(chi2, fit_path, pdf_model, runcard):
    # Read full_posterior_sample.csv
    post_sample = pd.read_csv(fit_path + "/full_posterior_sample.csv", index_col=0)

    # Produce FIT_XGRID and pred_data
    FIT_XGRID = colibri_api.FIT_XGRID(**runcard)
    pred_data = colibri_api.make_pred_data(**runcard)

    # Compute predictions
    pred_func = pdf_model.pred_and_pdf_func(FIT_XGRID, forward_map=pred_data)
    pred_func = jax.vmap(pred_func, in_axes=(0,), out_axes=(0, 0))
    pred = pred_func(post_sample.values)[0]

    # Compute average chi2
    chi2 = jax.vmap(chi2, in_axes=(0,), out_axes=0)
    avg_chi2 = chi2(pred).mean()

    return avg_chi2


def main():
    parser = argparse.ArgumentParser(description="Script to compare fits.")

    parser.add_argument("fit_names", nargs=2, help="List of colibri fits to compare.")

    args = parser.parse_args()

    # Load the pdf models
    pdf_models = [get_pdf_model(fit) for fit in args.fit_names]

    # Get fit paths
    fit_paths = [get_fit_path(fit) for fit in args.fit_names]

    # Read runcard.yaml in input folder
    runcards = []
    for fit_path in fit_paths:
        if not os.path.exists(fit_path + "/input/runcard.yaml"):
            raise FileNotFoundError(
                "Could not find the runcard.yaml file in the input folder of the fit "
                + fit_path
            )

        runcard_path = fit_path + "/input/runcard.yaml"
        with open(runcard_path, "r") as file:
            runcards.append(yaml.safe_load(file))

    # Check that the key dataset_inputs is the same
    if runcards[0]["dataset_inputs"] != runcards[1]["dataset_inputs"]:
        raise ValueError(
            "The dataset_inputs are not the same. Model comparison would be meaningless."
        )

    # Load results.json or results.csv for each fit
    logz = []
    max_logl = []
    for fit_path in fit_paths:
        if not os.path.exists(fit_path + "/ultranest_logs/info/results.json"):
            if os.path.exists(fit_path + "/results.csv"):
                with open(fit_path + "/results.csv", "r") as file:
                    results = pd.read_csv(file)
                    logz.append(results["logz"].values[0])
                    max_logl.append(results["logl"].values[0])
            else:
                raise FileNotFoundError(
                    "Could not find the results.json nor the results.csv file for the fit "
                    + fit_path
                )
        else:
            results_path = fit_path + "/ultranest_logs/info/results.json"
            with open(results_path, "r") as file:
                results = json.load(file)
                logz.append(results["logz"])
                max_logl.append(results["maximum_likelihood"]["logl"])

    # Produce chi2 functions
    chi2_functions = [colibri_api.make_chi2(**runcard) for runcard in runcards]

    # Check that chi2 functions are the same, i.e. for the same pdf and predictions
    # they produce the same chi2. This is important for the Bayes factor to be meaningful.
    if not check_chi2_functions(
        chi2_functions,
        produce_test_predictions(pdf_models[0], runcards[0], fit_paths[0]),
    ):
        raise ValueError(
            "The chi2 functions are not the same. Model comparison would be meaningless."
        )

    avg_chi2 = [
        compute_average_chi2(chi2, fit_path, pdf_model, runcard)
        for chi2, fit_path, pdf_model, runcard in zip(
            chi2_functions, fit_paths, pdf_models, runcards
        )
    ]

    # Print average chi2
    log.info(f"Average chi2 for fit 1: {avg_chi2[0]}")
    log.info(f"Average chi2 for fit 2: {avg_chi2[1]}")

    # Compute the bayesian complexity
    Cb1 = avg_chi2[0] + 2 * max_logl[0]
    Cb2 = avg_chi2[1] + 2 * max_logl[1]
    # Print the results
    log.info(f"Bayesian complexity for fit 1: {Cb1}")
    log.info(f"Bayesian complexity for fit 2: {Cb2}")

    # Print the logZ values
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
