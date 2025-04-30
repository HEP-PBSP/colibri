"""
A wrapper around n3fit/scripts/evolven3fit.py.

"""

import argparse
import logging
import os
import pathlib
import shutil
import sys
from glob import glob

import evolven3fit
import lhapdf
from evolven3fit.utils import read_runcard
from n3fit.scripts.evolven3fit import main as evolven3fit_main
from reportengine import colors
from validphys import lhio
from validphys.core import PDF
from validphys.scripts.postfit import PostfitError, relative_symlink, set_lhapdf_info

# Clear any existing handlers from root logger
root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.setLevel(logging.WARNING)  # Set higher threshold globally

# Set up module-specific logger
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(colors.ColorHandler())


def my_custom_get_theoryID_from_runcard(usr_path):
    """
    Does the same as `evolven3fit.utils.get_theoryID_from_runcard`
    but assumes that `theoryid` is defined in the runcard.
    """
    my_runcard = read_runcard(usr_path)
    return my_runcard["theoryid"]


# override (monkey patch) the function
evolven3fit.utils.get_theoryID_from_runcard = my_custom_get_theoryID_from_runcard


def _postfit_emulator(fit_path):
    """
    Emulates the postfit script from validphys/scripts/postfit.py
    by creating the symlinks, central replica and LHAPDF set
    within the postfit directory.

    It does not perform any selection of replicas, so it is
    equivalent to the postfit script but without the selection
    of replicas.
    """
    fitname = fit_path.name

    # Paths
    postfit_path = fit_path / "postfit"
    LHAPDF_path = postfit_path / fitname  # Path for LHAPDF grid output
    replicas_path = fit_path / "replicas"  # Path for replicas output

    # Generate postfit and LHAPDF directory
    if postfit_path.is_dir():
        log.warning(f"Removing existing postfit directory: {postfit_path}")
        shutil.rmtree(postfit_path)
    os.makedirs(LHAPDF_path, exist_ok=True)

    # Perform dummy postfit selection
    all_replicas = sorted(glob(f"{replicas_path}/replica_*/"))
    selected_paths = all_replicas

    # Copy info file
    info_source_path = replicas_path.joinpath(f"{fitname}.info")
    info_target_path = LHAPDF_path.joinpath(f"{fitname}.info")
    shutil.copy2(info_source_path, info_target_path)
    set_lhapdf_info(info_target_path, len(selected_paths))

    # Generate symlinks
    for drep, source_path in enumerate(selected_paths, 1):
        # Symlink results to postfit directory
        source_dir = pathlib.Path(source_path).resolve()
        target_dir = postfit_path.joinpath(f"replica_{drep}")
        relative_symlink(source_dir, target_dir)

        # Symlink results to pdfset directory
        source_grid = source_dir.joinpath(fitname + ".dat")
        target_file = f"{fitname}_{drep:04d}.dat"
        target_grid = LHAPDF_path.joinpath(target_file)
        relative_symlink(source_grid, target_grid)

    log.info(f"{len(selected_paths)} replicas written to the postfit folder")

    # Generate final PDF with replica 0
    log.info("Beginning construction of replica 0")
    # It's important that this is prepended, so that any existing instance of
    # `fitname` is not read from some other path
    lhapdf.pathsPrepend(str(postfit_path))
    generatingPDF = PDF(fitname)
    lhio.generate_replica0(generatingPDF)

    # Test replica 0
    try:
        lhapdf.mkPDF(fitname, 0)
    except RuntimeError as e:
        raise PostfitError("CRITICAL ERROR: Failure in reading replica zero") from e
    log.info("\n\n*****************************************************************\n")
    log.info("Postfit complete")
    log.info(f"Your LHAPDF set can be found in: {LHAPDF_path}")
    log.info("Please upload your results with:")
    log.info(f"\tvp-upload {fit_path}\n")
    log.info("and install with:")
    log.info(f"\tvp-get fit {fitname}\n")
    log.info(
        "\n\n*****************************************************************\n\n"
    )


def main():
    """
    Before running `evolven3fit` from n3fit/scripts/evolven3fit.py,
    creates a symlink called `nnfit` to replicas folder.
    """

    parser = argparse.ArgumentParser(
        description="A wrapper around n3fit/scripts/evolven3fit.py.\n"
        "Usage for evolution: `evolve_fit [evolve] <fit_name>`\n"
        "For more details, run `evolven3fit --help`.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Make `action` optional (defaults to “evolve”)
    parser.add_argument(
        "action",
        nargs="?",
        default="evolve",
        choices=["evolve"],
        help="The action to run (defaults to 'evolve').",
    )
    parser.add_argument("name_fit", help="The name of the fit directory")
    args = parser.parse_args()

    FIT_DIR = args.name_fit
    FIT_PATH = pathlib.Path(FIT_DIR).resolve()

    # Check if the action is 'evolve'
    if args.action != "evolve":
        raise ValueError("Invalid action. Only 'evolve' is supported.")

    # Check if the fit name is provided
    if not args.name_fit:
        raise ValueError("Please provide a fit name.")

    replicas_path = os.path.join(FIT_DIR, "replicas")
    symlink_path = os.path.join(FIT_DIR, "nnfit")

    if not os.path.exists(replicas_path):
        raise FileNotFoundError(f"Error: replicas folder not found at {replicas_path}")

    try:
        os.symlink("replicas", symlink_path)
    except FileExistsError:
        log.warning(f"Warning: symlink {symlink_path} already exists")

    # Run evolven3fit: invoke the underlying CLI with both args
    sys.argv = ["evolven3fit", args.action, args.name_fit]
    evolven3fit_main()

    # Run postfit emulator only for bayesian fits
    if "bayes_metrics.csv" in os.listdir(FIT_PATH):
        log.info("Running postfit emulator")
        _postfit_emulator(FIT_PATH)
    else:
        log.info("Skipping postfit emulator")
