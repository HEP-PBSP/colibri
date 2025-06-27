"""
An executable for assessing the quality of the Monte Carlo replicas produced
in an MC fit, and rejecting them if their chi2 exceeds a particular threshold.
"""

import os
import shutil
import pandas as pd
import argparse
import logging
import pathlib
import jax.numpy as jnp

from reportengine import colors

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(colors.ColorHandler())


def main():
    parser = argparse.ArgumentParser(
        description="Script to select MC replicas post-fit"
    )
    parser.add_argument("fit_name", help="The colibri fit to perform post-fit on.")
    parser.add_argument(
        "--chi2_threshold",
        "-c",
        type=float,
        default=1.5,
        help="The chi2 threshold, above which an MC replica will be rejected.",
    )
    parser.add_argument(
        "--nsigma",
        type=float,
        default=5,
        help="The nsigma threshold above which replicas are rejected.",
    )
    parser.add_argument(
        "--target_replicas",
        "-t",
        type=int,
        default=100,
        help="The target number of replicas to be produced by postfit.",
    )
    args = parser.parse_args()

    # Convert fit_path to a pathlib.Path object
    fit_path = pathlib.Path(args.fit_name)

    # Give names to other arguments
    chi2_threshold = args.chi2_threshold

    nsigma_threshold = args.nsigma

    # Check that the folder fit_replicas exists
    if not os.path.exists(fit_path / "fit_replicas"):
        raise FileNotFoundError(
            f"{fit_path}/fit_replicas does not exist; please run the Monte Carlo fit first."
        )

    # Filter out only the directories
    replicas_path = fit_path / "fit_replicas"
    # Create the directory for the replicas if it does not exist
    # else delete it and create it again
    if not os.path.exists(fit_path / "replicas"):
        os.mkdir(fit_path / "replicas")
    else:
        shutil.rmtree(fit_path / "replicas")
        os.mkdir(fit_path / "replicas")

    replicas_list = sorted(list(replicas_path.iterdir()))

    final_losses = jnp.array([])
    # Loop over the replicas and check their final loss
    for replica in replicas_list:
        # Get last iteration from the mc_loss.csv file
        final_loss = pd.read_csv(replica / "mc_loss.csv").iloc[-1]["training_loss"]
        final_losses = jnp.concatenate((final_losses, jnp.array([final_loss])), axis=0)

    mean_loss = jnp.mean(final_losses)
    std_loss = jnp.std(final_losses)

    # List of replicas to keep
    good_replicas = []

    # We will copy the replicas and order them starting with 0
    # and increasing the index for each good replica we find
    i = 0
    for replica, loss in zip(replicas_list, final_losses):

        index = int(replica.name.split("_")[1])

        chi2_pass = loss <= chi2_threshold
        nsigma_pass = loss - mean_loss <= nsigma_threshold * std_loss

        # Check if final loss is above the threshold
        if chi2_pass and nsigma_pass:
            # We found a good replica
            good_replicas.append(index)
            # Increase replica index
            i += 1
            # Copy the replica to the fit directory
            shutil.copytree(replica, fit_path / f"replicas/replica_{i}")

        if i == args.target_replicas:
            break

    if i < args.target_replicas:
        log.critical(
            f"You asked for {args.target_replicas} replicas, but only {i} replicas pass postfit selection.\n"
            f"You could consider increasing the threshold for the final training loss.",
        )

    else:
        log.info(
            f"Target number of replicas reached, {i} replicas pass postfit selection"
        )

    fit_dfs = []
    if good_replicas:
        for i in good_replicas:
            fit_dfs += [
                pd.read_csv(
                    replicas_path / f"replica_{i}" / f"mc_result_replica_{i}.csv",
                    index_col=0,
                )
            ]
    else:
        raise ValueError("No replicas pass the postfit selection.")

    # Keep only the replicas with index in good_replicas
    postfit_df = pd.concat(fit_dfs)
    postfit_df.index = [i + 1 for i in range(len(good_replicas))]

    # Save the postfit dataframe
    postfit_df.to_csv(fit_path / "mc_result.csv", float_format="%.5e")

    log.info("Postfit completed")
