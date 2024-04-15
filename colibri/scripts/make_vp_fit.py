"""
This script makes a colibri fit compatible with the validphys fit format.
In order to do so it needs to 
- copy the `input/runcard.yaml` file into a `filter.yml` file
- generate an md5 hash of the `filter.yml` file
- copy the fit to the approprate results directory

Note that such a format is useful for compatibility with the `validphys` library.
In particular, it allows on to upload the fit with `vp-upload` but also to nake use of the
parse rules for FitSpec.
"""

import argparse
import hashlib
import logging
import os
import pathlib
import sys

from reportengine import colors

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(colors.ColorHandler())


def main():
    parser = argparse.ArgumentParser(
        description="Script to make a colibri fit compatible with validphys standards."
    )
    parser.add_argument("fit_name", help="The colibri fit to standardise.")

    parser.add_argument(
        "--copy_fit",
        "-copy",
        type=bool,
        default=True,
        help="Whether to copy the fit to the results directory.",
    )

    parser.add_argument(
        "--remove_fit",
        "-remove",
        type=bool,
        default=False,
        help="Whether to remove the fit from the original directory and only keep it in the results directory.",
    )

    args = parser.parse_args()

    # Convert fit_path to a pathlib.Path object
    fit_path = pathlib.Path(args.fit_name)

    # Check that the input directory exists
    if not os.path.exists(fit_path / "input"):
        raise FileNotFoundError(
            f"{fit_path}/input does not exist; Make sure that the fit was runned correctly."
        )

    # Copy input/runcard.yaml to filter.yml
    os.system(f"cp {fit_path}/input/runcard.yaml {fit_path}/filter.yml")
    log.info(f"input/runcard.yaml copied to filter.yml")

    # Generate md5 hash of the filter.yml file
    output_filename = fit_path / "md5"
    with open(fit_path / "filter.yml", "rb") as f:
        hash_md5 = hashlib.md5(f.read()).hexdigest()
    with open(output_filename, "w") as g:
        g.write(hash_md5)

    log.info(f"md5 {hash_md5} stored in {output_filename}")

    # Copy the fit to the results directory
    results_fit_path = pathlib.Path(sys.prefix) / "share/colibri/results"

    if args.remove_fit:
        os.system(f"mv {fit_path} {results_fit_path}")
        log.info(f"Fit {fit_path} moved to {results_fit_path}")

    elif args.copy_fit:
        os.system(f"cp -r {fit_path} {results_fit_path}")
        log.info(f"Fit {fit_path} copied to {results_fit_path}")
