"""
An executable for performing evolution of exportgrids, producing
an LHAPDF grid.
"""

import os
import argparse
import yaml
import logging
import numpy as np

from pathlib import Path

from scipy.interpolate import interp1d

import eko
from eko import basis_rotation
from ekobox import info_file, genpdf, apply
from validphys.pdfbases import PIDS_DICT

import lhapdf

from validphys.loader import Loader
from colibri.constants import LHAPDF_XGRID

from collections import defaultdict

from validphys.lhio import generate_replica0

from reportengine import colors

from mpi4py import MPI

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(colors.ColorHandler())


def lhapdf_path():
    """Returns the path to the share/LHAPDF directory"""
    return lhapdf.paths()[0]


def process_replica(replica, pdf_data, eko_op, lhapdf_destination):
    evolved_blocks = evolve_exportgrid(pdf_data, eko_op, LHAPDF_XGRID)
    replica_num = replica.removeprefix("replica_")
    genpdf.export.dump_blocks(
        Path(lhapdf_destination),
        int(replica_num),
        evolved_blocks,
        pdf_type=f"PdfType: replica\nFromMCReplica: {replica_num}\n",
    )

    log.info(f"Evolved replica {replica_num}.")


def main():
    parser = argparse.ArgumentParser(description="Script to evolve PDF exportgrids")
    parser.add_argument("fit_name", help="The colibri fit to evolve.")
    args = parser.parse_args()

    # Read theory from fit runcard, and load eko
    with open(args.fit_name + "/input/runcard.yaml", "r") as file:
        runcard = yaml.safe_load(file)
    theoryid = runcard["theoryid"]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    eko_path = (Loader().check_theoryID(theoryid).path) / "eko.tar"

    with eko.EKO.edit(eko_path) as eko_op:
        x_grid_obj = eko.interpolation.XGrid(LHAPDF_XGRID)
        eko.io.manipulate.xgrid_reshape(
            eko_op, targetgrid=x_grid_obj, inputgrid=x_grid_obj
        )

    # Load all replicas into a dictionary
    initial_PDFs_dict = {}
    replicas_path = args.fit_name + "/replicas"
    for yaml_file in Path(replicas_path).glob(f"replica_*/{args.fit_name}.exportgrid"):
        data = yaml.safe_load(yaml_file.read_text(encoding="UTF-8"))
        initial_PDFs_dict[yaml_file.parent.stem] = data

    # Apply the evolution
    with eko.EKO.read(eko_path) as eko_op:
        # Read the cards directly from the eko to make sure they are consistent
        theory = eko_op.theory_card
        op = eko_op.operator_card

        # Modify the info file with the fit-specific info
        info = info_file.build(theory, op, 1, info_update={})
        info["NumMembers"] = len(initial_PDFs_dict)
        info["ErrorType"] = "replicas"  # Needs to be modified if Hessian!
        info["XMin"] = float(LHAPDF_XGRID[0])
        info["XMax"] = float(LHAPDF_XGRID[-1])
        info["Flavors"] = basis_rotation.flavor_basis_pids

        # If no LHAPDF folder exists, create one
        lhapdf_destination = lhapdf_path() + "/" + args.fit_name
        if not os.path.exists(lhapdf_destination):
            os.mkdir(lhapdf_destination)

        genpdf.export.dump_info(lhapdf_destination, info)

        # Distribute replicas among processes
        local_replicas = [
            replica
            for i, replica in enumerate(initial_PDFs_dict.items())
            if i % size == rank
        ]

        # Process local replicas
        for replica, pdf_data in local_replicas:
            process_replica(replica, pdf_data, eko_op, lhapdf_destination)

        # Synchronize to ensure all processes have finished
        comm.Barrier()
        if rank == 0:
            log.info(
                f"Evolution complete. Evolved grids can be found in {lhapdf_destination}."
            )
            # Produce the central replica
            log.info("Producing central replica.")
            l = Loader()
            pdf = l.check_pdf(args.fit_name)
            generate_replica0(pdf)


# This class is copied directly from evolven3fit_new
class LhapdfLike:
    """
    Class which emulates lhapdf but only for an initial condition PDF (i.e. with only one q2 value).

    Q20 is the fitting scale fo the pdf and it is the only available scale for the objects of this class.

    X_GRID is the grid of x values on top of which the pdf is interpolated.

    PDF_GRID is a dictionary containing the pdf grids at fitting scale for each pid.
    """

    def __init__(self, pdf_grid, q20, x_grid):
        self.pdf_grid = pdf_grid
        self.q20 = q20
        self.x_grid = x_grid
        self.funcs = [
            interp1d(self.x_grid, self.pdf_grid[pid], kind="cubic")
            for pid in range(len(PIDS_DICT))
        ]

    def xfxQ2(self, pid, x, q2):
        """Return the value of the PDF for the requested pid, x value and, whatever the requested
        q2 value, for the fitting q2.

        Parameters
        ----------

            pid: int
                pid index of particle
            x: float
                x-value
            q2: float
                Q square value

        Returns
        -------
            : float
            x * PDF value
        """
        return self.funcs[list(PIDS_DICT.values()).index(PIDS_DICT[pid])](x)

    def hasFlavor(self, pid):
        """Check if the requested pid is in the PDF."""
        return pid in PIDS_DICT


# This function is copied directly from evolven3fit_new
def evolve_exportgrid(exportgrid, eko, x_grid):
    """
    Evolves the provided exportgrid for the desired replica with the eko and returns the evolved block

    Parameters
    ----------
        exportgrid: dict
            exportgrid of pdf at fitting scale
        eko: eko object
            eko operator for evolution
        xgrid: list
            xgrid to be used as the targetgrid
    Returns
    -------
        : list(np.array)
        list of evolved blocks
    """
    # construct LhapdfLike object
    pdf_grid = np.array(exportgrid["pdfgrid"]).transpose()
    pdf_to_evolve = LhapdfLike(pdf_grid, exportgrid["q20"], x_grid)
    # evolve pdf
    evolved_pdf = apply.apply_pdf(eko, pdf_to_evolve)
    # generate block to dump
    targetgrid = eko.bases.targetgrid.tolist()

    # Finally separate by nf block (and order per nf/q)
    by_nf = defaultdict(list)
    for q, nf in sorted(eko.evolgrid, key=lambda ep: ep[1]):
        by_nf[nf].append(q)
    q2block_per_nf = {nf: sorted(qs) for nf, qs in by_nf.items()}

    blocks = []
    for nf, q2grid in q2block_per_nf.items():

        def pdf_xq2(pid, x, Q2):
            x_idx = targetgrid.index(x)
            return x * evolved_pdf[(Q2, nf)]["pdfs"][pid][x_idx]

        block = genpdf.generate_block(
            pdf_xq2,
            xgrid=targetgrid,
            sorted_q2grid=q2grid,
            pids=basis_rotation.flavor_basis_pids,
        )
        blocks.append(block)

    return blocks
