"""
A wrapper around n3fit/scripts/evolven3fit.py.

"""

import logging
import os
import pathlib
import shutil
import sys
from glob import glob

import lhapdf
from n3fit.scripts.evolven3fit import main as evolven3fit_main
from reportengine import colors
from validphys import lhio
from validphys.core import PDF
from validphys.scripts.postfit import PostfitError, relative_symlink, set_lhapdf_info

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(colors.ColorHandler())


if len(sys.argv) != 3:
    log.error("Usage: evolve_fit <command> name_fit")
    sys.exit(1)

FIT_DIR = sys.argv[2]
FIT_PATH = pathlib.Path(FIT_DIR).resolve()


def _postfit_emulator():
    """
    Emulates the postfit script from validphys/scripts/postfit.py
    by creating the symlinks, central replica and LHAPDF set
    within the postfit directory.

    It does not perform any selection of replicas, so it is
    equivalent to the postfit script but without the selection
    of replicas.
    """
    fitname = FIT_PATH.name

    # Paths
    postfit_path = FIT_PATH / "postfit"
    LHAPDF_path = postfit_path / fitname  # Path for LHAPDF grid output
    replicas_path = FIT_PATH / "replicas"  # Path for replicas output

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
    log.info("Please upload your results with:")
    log.info(f"\tvp-upload {FIT_PATH}\n")
    log.info("and install with:")
    log.info(f"\tvp-get fit {fitname}\n")
    log.info("*****************************************************************\n\n")


def main():
    """
    Before running `evolven3fit` from n3fit/scripts/evolven3fit.py,
    creates a symlink called `nnfit` to replicas folder.
    """
    replicas_path = os.path.join(FIT_DIR, "replicas")
    symlink_path = os.path.join(FIT_DIR, "nnfit")

    if not os.path.exists(replicas_path):
        print(f"Error: replicas folder not found at {replicas_path}")
        sys.exit(1)

    try:
        os.symlink("replicas", symlink_path)
    except FileExistsError:
        print(f"Warning: symlink {symlink_path} already exists")

    # try:
    evolven3fit_main()

    # Run postfit emulator
    _postfit_emulator()

    # TODO: option to create postfit folder in the case of a Bayesian fit

    # TODO: option on whether to generate central replica or not (useful for wmin basis generation)
    # TODO: symlink fit folder to environment NNPDF results folder

    # TODO: symlinking of lhapdf folder

    # finally:
    #     if os.path.islink(symlink_path):
    #         os.remove(symlink_path)

    # symlink evolved fit to lhapdf repository
