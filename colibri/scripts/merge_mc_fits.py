"""
An executable for merging identical MC fits, keeping the best fit replica by replica.
"""

import os
import shutil
import pandas as pd
import argparse
import logging

from reportengine import colors

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(colors.ColorHandler())


def main():
    parser = argparse.ArgumentParser(description="Script to merge MC fits.")

    parser.add_argument("fit_names", nargs="+", help="List of colibri fits to merge")

    parser.add_argument(
        "--target_replicas",
        "-t",
        type=int,
        default=100,
        help="The target number of replicas to be produced by the merge.",
    )

    parser.add_argument(
        "--merged_fit_name",
        "-newfit",
        type=str,
        default="Merged_fit",
        help="The name of the merged fit.",
    )

    args = parser.parse_args()

    # Create folder for the merged fit
    if not os.path.exists(args.merged_fit_name):
        os.mkdir(args.merged_fit_name)
    else:
        shutil.rmtree(args.merged_fit_name)
        os.mkdir(args.merged_fit_name)

    # Copy the pdf_model.pkl file from the first fit
    shutil.copy(args.fit_names[0] + "/pdf_model.pkl", args.merged_fit_name)

    # Create the directory for the replicas if it does not exist
    # else delete it and create it again
    if not os.path.exists(args.merged_fit_name + "/fit_replicas"):
        os.mkdir(args.merged_fit_name + "/fit_replicas")
    else:
        shutil.rmtree(args.merged_fit_name + "/fit_replicas")
        os.mkdir(args.merged_fit_name + "/fit_replicas")

    for i in range(args.target_replicas):
        losses = []
        for fit in args.fit_names:
            if not os.path.exists(fit + "/fit_replicas/replica_" + str(i + 1)):
                raise FileNotFoundError(
                    f"{fit}/fit_replicas/replica_{i + 1} does not exist."
                )
            path = os.path.join(fit, "fit_replicas", f"replica_{i + 1}")
            df = pd.read_csv(path + "/mc_loss.csv")

            # take the last value of the loss and compare, keeping the smallest
            loss = df["training_loss"].iloc[-1]

            losses.append(loss)

        # Get the smallest loss and copy the corresponding replica in the merged fit folder
        best_fit = args.fit_names[losses.index(min(losses))]
        shutil.copytree(
            best_fit + "/fit_replicas/replica_" + str(i + 1),
            args.merged_fit_name + "/fit_replicas/replica_" + str(i + 1),
        )

    log.info("Merge completed.")
