"""
An executable to perform bayesian model selection on a set of fits.
"""

import argparse
import logging

from reportengine import colors
import os
from colibri.utils import get_pdf_model, get_fit_path
import yaml
import json
import pandas as pd

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(colors.ColorHandler())


def main():
    parser = argparse.ArgumentParser(description="Script to compare fits.")

    parser.add_argument("fit_names", nargs=2, help="List of colibri fits to compare.")

    args = parser.parse_args()

    # Load the pdf models
    pdf_models = [get_pdf_model(fit) for fit in args.fit_names]

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

    # Load results.json or evidence.csv for each fit
    logz = []
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
                logz.append(json.load(file)["logz"])

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
