"""
A wrapper around n3fit/scripts/evolven3fit.py.

"""

import logging
import os
import sys

from n3fit.scripts.evolven3fit import main as evolven3fit_main
from reportengine import colors

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(colors.ColorHandler())


def main():
    """
    Before running `evolven3fit` from n3fit/scripts/evolven3fit.py,
    creates a symlink called `nnfit` to replicas folder.
    """
    if len(sys.argv) != 3:
        log.error("Usage: evolve_fit <command> name_fit")
        sys.exit(1)

    fit_dir = sys.argv[2]
    replicas_path = os.path.join(fit_dir, "replicas")
    symlink_path = os.path.join(fit_dir, "nnfit")

    if not os.path.exists(replicas_path):
        print(f"Error: replicas folder not found at {replicas_path}")
        sys.exit(1)

    try:
        os.symlink("replicas", symlink_path)
    except FileExistsError:
        print(f"Warning: symlink {symlink_path} already exists")

    # try:
    evolven3fit_main()

    # TODO: symlink fit folder to environment NNPDF results folder

    # TODO: symlinking of lhapdf folder

    # finally:
    #     if os.path.islink(symlink_path):
    #         os.remove(symlink_path)

    # symlink evolved fit to lhapdf repository
