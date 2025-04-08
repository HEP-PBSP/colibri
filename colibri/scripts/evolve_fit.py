"""
A wrapper around n3fit/scripts/evolven3fit.py.

"""

import logging
import os
import sys
import pathlib
import shutil
from glob import glob

from n3fit.scripts.evolven3fit import main as evolven3fit_main
from validphys.scripts.postfit import set_lhapdf_info, relative_symlink

from reportengine import colors

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
    Emulates the postfit script from validphys/scripts/postfit.py.
    What it does:
    - Creates a postfit folder in the root of the fit directory.
    - Generates the central replica.
    - Creates an LHAPDF folder in the standard LHAPDF location of the environment.
    """
    fitname = FIT_PATH.name
    
    # Paths
    postfit_path = FIT_PATH / 'postfit'
    LHAPDF_path  = postfit_path/fitname     # Path for LHAPDF grid output
    replicas_path = FIT_PATH / 'replicas'   # Path for replicas output

    # Generate postfit and LHAPDF directory
    if postfit_path.is_dir():
        log.warning(f"Removing existing postfit directory: {postfit_path}")
        shutil.rmtree(postfit_path)
    os.mkdir(LHAPDF_path)

    # Perform dummy postfit selection
    all_replicas   = sorted(glob(f"{replicas_path}/replica_*/"))
    selected_paths = all_replicas
    
    # Copy info file
    info_source_path = replicas_path.joinpath(f'{fitname}.info')
    info_target_path = LHAPDF_path.joinpath(f'{fitname}.info')
    shutil.copy2(info_source_path, info_target_path)
    set_lhapdf_info(info_target_path, len(selected_paths))

    # Generate symlinks
    for drep, source_path in enumerate(selected_paths, 1):
        # Symlink results to postfit directory
        source_dir = pathlib.Path(source_path).resolve()
        target_dir = postfit_path.joinpath(f'replica_{drep}')
        relative_symlink(source_dir, target_dir)

        # Symlink results to pdfset directory
        source_grid = source_dir.joinpath(fitname+'.dat')
        target_file = f'{fitname}_{drep:04d}.dat'
        target_grid = LHAPDF_path.joinpath(target_file)
        relative_symlink(source_grid, target_grid)
    


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

    # TODO: option to create postfit folder in the case of a Bayesian fit

    # TODO: option on whether to generate central replica or not (useful for wmin basis generation)
    # TODO: symlink fit folder to environment NNPDF results folder

    # TODO: symlinking of lhapdf folder

    # finally:
    #     if os.path.islink(symlink_path):
    #         os.remove(symlink_path)

    # symlink evolved fit to lhapdf repository
