"""
An executable for sampling and storing exportgrids from the stored ultranest logs posterior samples
without having to re run the entire fit.
"""

import argparse
import logging
import pathlib

from reportengine import colors

from colibri.utils import ns_fit_resampler


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

    parser.add_argument(
        "--parametrisation_scale",
        "-Q",
        type=float,
        default=1.65,
        help="The scale at which the PDFs are fitted.",
    )

    args = parser.parse_args()

    if args.resampled_fit_name is None:
        args.resampled_fit_name = "resampled_" + args.fit_name

    # Convert fit_path to a pathlib.Path object
    fit_path = pathlib.Path(args.fit_name)

    # path of resampled fit
    resampled_fit_path = pathlib.Path(args.resampled_fit_name)

    ns_fit_resampler(
        fit_path=fit_path,
        resampled_fit_path=resampled_fit_path,
        n_replicas=args.nreplicas,
        resampling_seed=args.resampling_seed,
        resampled_fit_name=args.resampled_fit_name,
        parametrisation_scale=args.parametrisation_scale,
    )
