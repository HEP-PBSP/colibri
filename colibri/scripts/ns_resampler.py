"""
An executable for sampling and storing exportgrids from the stored ultranest logs posterior samples
without having to re run the entire fit.
"""

import os
import pandas as pd
import argparse
import logging
import pathlib
import dill

from reportengine import colors

from colibri.utils import resample_from_ns_posterior
from colibri.lhapdf import write_exportgrid

from mpi4py import MPI


log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(colors.ColorHandler())


def main():
    parser = argparse.ArgumentParser(description="Script to resample from NS posterior")
    parser.add_argument("fit_name", help="The colibri fit from which to sample.")
    parser.add_argument(
        "--nreplicas",
        "-nrep",
        type=int,
        default=100,
        help="The number of samples.",
    )
    parser.add_argument(
        "--resampling_seed",
        "-seed",
        type=int,
        default=1,
        help="The random seed to be used to sample from the posterior.",
    )

    parser.add_argument(
        "--resampled_fit_name",
        "-newfit",
        type=str,
        default=None,
        help="The name of the resampled fit.",
    )

    args = parser.parse_args()
    if args.resampled_fit_name is None:
        args.resampled_fit_name = "resampled_" + args.fit_name

    # Convert fit_path to a pathlib.Path object
    fit_path = pathlib.Path(args.fit_name)

    # path of resampled fit
    resampled_fit_path = pathlib.Path(args.resampled_fit_name)

    # Give names to other arguments
    nreplicas = args.nreplicas
    resampling_seed = args.resampling_seed

    # check whether fit path exists

    log.info(f"Loading pdf model from {fit_path}")
    # load pdf_model from fit using dill
    with open(fit_path / "pdf_model.pkl", "rb") as file:
        pdf_model = dill.load(file)

    # Check that the .txt file with posterior samples exists
    if not os.path.exists(fit_path / "ultranest_logs/chains/equal_weighted_post.txt"):
        raise FileNotFoundError(
            f"{fit_path}/ultranest_logs/chains/equal_weighted_post.txt does not exist;"
            "please run the bayesian fit first."
        )

    equal_weight_post_path = fit_path / "ultranest_logs/chains/equal_weighted_post.txt"

    samples = pd.read_csv(equal_weight_post_path, sep="\s+", dtype=float).values

    if nreplicas > samples.shape[0]:
        nreplicas = samples.shape[0]
        log.warning(
            f"The chosen number of posterior samples exceeds the number of posterior"
            "samples computed by ultranest. Setting the number of resampled posterior"
            f"samples to {nreplicas}"
        )

    resampled_posterior = resample_from_ns_posterior(
        samples,
        nreplicas,
        resampling_seed,
    )

    # copy old fit to resampled fit
    os.system(f"cp -r {fit_path} {resampled_fit_path}")

    # remove old replicas from resampled fit
    os.system(f"rm -r {resampled_fit_path}/replicas")

    # overwrite old ns_result.csv with resampled posterior
    parameters = pdf_model.param_names
    df = pd.DataFrame(resampled_posterior, columns=parameters)
    df.to_csv(str(resampled_fit_path) + "/ns_result.csv")

    # Initialize MPI communication
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Distribute exportgrid writing tasks across processes
    start_index = rank * resampled_posterior.shape[0] // size
    end_index = (rank + 1) * resampled_posterior.shape[0] // size

    for i in range(start_index, end_index):
        log.info(f"Writing exportgrid for replica {i+1} on rank {rank}")
        write_exportgrid(
            resampled_posterior[i],
            pdf_model,
            i + 1,
            resampled_fit_path,
        )

    log.info(f"Resampling completed. Resampled fit stored in {resampled_fit_path}")
