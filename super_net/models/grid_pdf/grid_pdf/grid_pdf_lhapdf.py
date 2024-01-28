"""
grid_pdf.grid_pdf_lhapdf.py

Module containing functions used to write a grid PDF fit to an LHAPDF grid.

Author: James Moore
Date: 18.12.2023
"""

from grid_pdf.grid_pdf_model import interpolate_grid
from super_net.utils import FLAVOURS_ID_MAPPINGS
from validphys.loader import Loader
from validphys.lhio import generate_replica0
from super_net.constants import LHAPDF_XGRID

from pathlib import Path

import lhapdf
import numpy as np
import os
import yaml

import logging
from scipy.interpolate import interp1d

import eko
from ekobox import info_file, genpdf, apply
from eko import basis_rotation
from validphys.pdfbases import PIDS_DICT

from collections import defaultdict

log = logging.getLogger(__name__)


def lhapdf_path():
    """Returns the path to the share/LHAPDF directory"""
    return lhapdf.paths()[0]


EXPORT_LABELS = [
    "TBAR",
    "BBAR",
    "CBAR",
    "SBAR",
    "UBAR",
    "DBAR",
    "GLUON",
    "D",
    "U",
    "S",
    "C",
    "B",
    "T",
    "PHT",
]

export_to_evolution = {
    "\Sigma": {
        "U": 1,
        "UBAR": 1,
        "D": 1,
        "DBAR": 1,
        "S": 1,
        "SBAR": 1,
        "C": 1,
        "CBAR": 1,
        "B": 1,
        "BBAR": 1,
        "T": 1,
        "TBAR": 1,
    },
    "V": {
        "U": 1,
        "UBAR": -1,
        "D": 1,
        "DBAR": -1,
        "S": 1,
        "SBAR": -1,
        "C": 1,
        "CBAR": -1,
        "B": 1,
        "BBAR": -1,
        "T": 1,
        "TBAR": -1,
    },
    "T3": {"U": 1, "UBAR": 1, "D": -1, "DBAR": -1},
    "V3": {"U": 1, "UBAR": -1, "d": -1, "DBAR": 1},
    "T8": {"U": 1, "UBAR": 1, "D": 1, "DBAR": 1, "S": -2, "SBAR": -2},
    "V8": {"U": 1, "UBAR": -1, "D": 1, "DBAR": -1, "S": -2, "SBAR": +2},
    "T15": {
        "U": 1,
        "UBAR": 1,
        "D": 1,
        "DBAR": 1,
        "S": 1,
        "SBAR": 1,
        "c": -3,
        "CBAR": -3,
    },
    "V15": {
        "U": 1,
        "UBAR": -1,
        "D": 1,
        "DBAR": -1,
        "S": 1,
        "SBAR": -1,
        "c": -3,
        "CBAR": +3,
    },
    "T24": {
        "U": 1,
        "UBAR": 1,
        "D": 1,
        "DBAR": 1,
        "S": 1,
        "SBAR": 1,
        "C": 1,
        "CBAR": 1,
        "B": -4,
        "BBAR": -4,
    },
    "V24": {
        "U": 1,
        "UBAR": -1,
        "D": 1,
        "DBAR": -1,
        "S": 1,
        "SBAR": -1,
        "C": 1,
        "CBAR": -1,
        "B": -4,
        "BBAR": +4,
    },
    "T35": {
        "U": 1,
        "UBAR": 1,
        "D": 1,
        "DBAR": 1,
        "S": 1,
        "SBAR": 1,
        "C": 1,
        "CBAR": 1,
        "B": 1,
        "BBAR": 1,
        "T": -5,
        "TBAR": -5,
    },
    "V35": {
        "U": 1,
        "UBAR": -1,
        "D": 1,
        "DBAR": -1,
        "S": 1,
        "SBAR": -1,
        "C": 1,
        "CBAR": -1,
        "B": 1,
        "BBAR": -1,
        "T": -5,
        "TBAR": +5,
    },
    "g": {"GLUON": 1},
    "photon": {"PHT": 1},
}

# Construct the inverse transformation from evolution to export
num_flav = len(FLAVOURS_ID_MAPPINGS)
export_to_evolution_matrix = np.zeros((num_flav, num_flav))
for i in range(num_flav):
    j = 0
    for flav in EXPORT_LABELS:
        if flav in export_to_evolution[FLAVOURS_ID_MAPPINGS[i]].keys():
            export_to_evolution_matrix[i, j] = export_to_evolution[
                FLAVOURS_ID_MAPPINGS[i]
            ][flav]
        else:
            export_to_evolution_matrix[i, j] = 0
        j += 1

evolution_to_export_matrix = np.linalg.inv(export_to_evolution_matrix)


def write_exportgrid_from_fit_samples(
    samples,
    n_posterior_samples,
    reduced_xgrids,
    length_reduced_xgrids,
    flavour_indices,
    output_path=None,
):
    """
    Writes an exportgrid for each of the replicas in the posterior sample.
    The exportgrids are written to a folder called "replicas" in the output_path.
    The exportgrids are written in the format required by EKO, but are not yet
    evolved.

    Parameters
    ----------
    samples: list
        List of posterior samples.
    
    n_posterior_samples: int
        Number of posterior samples.
    
    reduced_xgrids: dict
        The reduced x-grids used in the fit, organised by flavour.
    
    length_reduced_xgrids: int
        The length of the reduced x-grids.
    
    flavour_indices: list
        The indices of the flavours used in the fit.

    output_path: pathlib.PosixPath
        Path to the output folder.

    Returns
    -------
    None
        
    """

    # Write an export grid at the initial scale for each of the replicas in the posterior
    # sample.
    replicas_path = str(output_path) + "/replicas"
    if not os.path.exists(replicas_path):
        os.mkdir(replicas_path)

    fit_name = str(output_path).split("/")[-1]

    for i in range(n_posterior_samples):
        rep_path = replicas_path + "/replica_" + str(i + 1)
        if not os.path.exists(rep_path):
            os.mkdir(rep_path)
        exportgrid = write_exportgrid(
            samples, reduced_xgrids, length_reduced_xgrids, flavour_indices, i
        )
        with open(rep_path + "/" + fit_name + ".exportgrid", "w") as outfile:
            yaml.dump(exportgrid, outfile)

    return None


def evolution_of_exportgrid(fit_path, fit_name, theoryid, n_posterior_samples, folder=lhapdf_path):
    """
    Evolves the exportgrids stored in the replicas folder of the fit_path.
    The evolved grids are written to the folder specified by folder which is
    assumed to be the share/LHAPDF folder.

    Once the evolved grids are written, the central replica is generated.
    Note: for a successful generation of the central replica, the n_posterior_samples
    must be equal to the number of replicas in the replicas folder.


    Parameters
    ----------
    fit_path: str
        Path to the fit folder.

    fit_name: str
        Name of the fit (is the same name of the runcard used for the fit).
    
    theoryid: validphys.core.TheoryIDSpec
        TheoryID of the theory used for the fit.
    
    n_posterior_samples: int
        Number of posterior samples.
    
    folder: str, default=lhapdf_path
        Path to the LHAPDF folder.
    
    Returns
    -------
    None
        
    """
    replicas_path = str(fit_path) + "/replicas"

    # Now run EKO on the exportgrids to complete the PDF evolution
    log.info(f"Loading eko from theory {theoryid.id}")
    eko_path = (Loader().check_theoryID(theoryid.id).path) / "eko.tar"

    with eko.EKO.edit(eko_path) as eko_op:
        x_grid_obj = eko.interpolation.XGrid(LHAPDF_XGRID)
        eko.io.manipulate.xgrid_reshape(
            eko_op, targetgrid=x_grid_obj, inputgrid=x_grid_obj
        )

    # Load the export grids into a dictionary
    initial_PDFs_dict = {}
    for yaml_file in Path(replicas_path).glob(
        f"replica_*/{fit_name}.exportgrid"
    ):
        data = yaml.safe_load(yaml_file.read_text(encoding="UTF-8"))
        initial_PDFs_dict[yaml_file.parent.stem] = data

    with eko.EKO.read(eko_path) as eko_op:
        # Read the cards directly from the eko to make sure they are consistent
        theory = eko_op.theory_card
        op = eko_op.operator_card

        # Modify the info file with the fit-specific info
        info = info_file.build(theory, op, 1, info_update={})
        info["NumMembers"] = n_posterior_samples
        info["ErrorType"] = "replicas"
        info["XMin"] = float(LHAPDF_XGRID[0])
        info["XMax"] = float(LHAPDF_XGRID[-1])
        # Save the PIDs in the info file in the same order as in the evolution
        info["Flavors"] = basis_rotation.flavor_basis_pids
        # info["NumFlavors"] = theory.heavy.num_flavs_max_pdf

        # If no LHAPDF folder exists, create one
        lhapdf_destination = folder() + "/" + fit_name
        if not os.path.exists(lhapdf_destination):
            os.mkdir(lhapdf_destination)

        genpdf.export.dump_info(lhapdf_destination, info)

        progress = 1
        for replica, pdf_data in initial_PDFs_dict.items():
            log.info(
                "Evolving replica " + str(progress) + " of " + str(n_posterior_samples)
            )
            evolved_blocks = evolve_exportgrid(pdf_data, eko_op, LHAPDF_XGRID)
            replica_num = replica.removeprefix("replica_")
            genpdf.export.dump_blocks(
                Path(lhapdf_destination),
                int(replica_num),
                evolved_blocks,
                pdf_type=f"PdfType: replica\nFromMCReplica: {replica_num}\n",
            )

            progress += 1
    
    log.info("Generating central replica")
    # Produce the central replica
    l = Loader()
    pdf = l.check_pdf(fit_name)
    generate_replica0(pdf)

    return None


def lhapdf_grid_pdf_from_samples(
    samples,
    reduced_xgrids,
    flavour_indices,
    length_reduced_xgrids,
    n_posterior_samples,
    theoryid,
    folder=lhapdf_path,
    output_path=None,
):
    """
    TODO
    """

    # Write an export grid at the initial scale for each of the replicas in the posterior
    # sample.
    replicas_path = str(output_path) + "/replicas"
    if not os.path.exists(replicas_path):
        os.mkdir(replicas_path)

    fit_name = str(output_path).split("/")[-1]

    for i in range(n_posterior_samples):
        rep_path = replicas_path + "/replica_" + str(i + 1)
        if not os.path.exists(rep_path):
            os.mkdir(rep_path)
        exportgrid = write_exportgrid(
            samples, reduced_xgrids, length_reduced_xgrids, flavour_indices, i
        )
        with open(rep_path + "/" + fit_name + ".exportgrid", "w") as outfile:
            yaml.dump(exportgrid, outfile)

    # Now run EKO on the exportgrids to complete the PDF evolution
    log.info(f"Loading eko from theory {theoryid.id}")
    eko_path = (Loader().check_theoryID(theoryid.id).path) / "eko.tar"

    with eko.EKO.edit(eko_path) as eko_op:
        x_grid_obj = eko.interpolation.XGrid(LHAPDF_XGRID)
        eko.io.manipulate.xgrid_reshape(
            eko_op, targetgrid=x_grid_obj, inputgrid=x_grid_obj
        )

    # Load the export grids into a dictionary
    initial_PDFs_dict = {}
    for yaml_file in Path(replicas_path).glob(
        f"replica_*/{output_path.name}.exportgrid"
    ):
        data = yaml.safe_load(yaml_file.read_text(encoding="UTF-8"))
        initial_PDFs_dict[yaml_file.parent.stem] = data

    with eko.EKO.read(eko_path) as eko_op:
        # Read the cards directly from the eko to make sure they are consistent
        theory = eko_op.theory_card
        op = eko_op.operator_card

        # Modify the info file with the fit-specific info
        info = info_file.build(theory, op, 1, info_update={})
        info["NumMembers"] = n_posterior_samples
        info["ErrorType"] = "replicas"
        info["XMin"] = float(LHAPDF_XGRID[0])
        info["XMax"] = float(LHAPDF_XGRID[-1])
        # Save the PIDs in the info file in the same order as in the evolution
        info["Flavors"] = basis_rotation.flavor_basis_pids
        # info["NumFlavors"] = theory.heavy.num_flavs_max_pdf

        # If no LHAPDF folder exists, create one
        lhapdf_destination = folder + "/" + fit_name
        if not os.path.exists(lhapdf_destination):
            os.mkdir(lhapdf_destination)

        genpdf.export.dump_info(lhapdf_destination, info)

        progress = 1
        for replica, pdf_data in initial_PDFs_dict.items():
            log.info(
                "Evolving replica " + str(progress) + " of " + str(n_posterior_samples)
            )
            evolved_blocks = evolve_exportgrid(pdf_data, eko_op, LHAPDF_XGRID)
            replica_num = replica.removeprefix("replica_")
            genpdf.export.dump_blocks(
                Path(lhapdf_destination),
                int(replica_num),
                evolved_blocks,
                pdf_type=f"PdfType: replica\nFromMCReplica: {replica_num}\n",
            )

            progress += 1


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


def write_exportgrid(
    df,
    reduced_xgrids,
    length_reduced_xgrids,
    flavour_indices,
    replica,
    Q0=1.65,
    xgrid=LHAPDF_XGRID,
):
    """

    Parameters
    ---------
    xgrid:
      A list of x points.

    Q0:
      The initial scale at which the grids are written.

    replica:
      An integer which indexes the replica.
    """

    # Interpolate on the xgrid
    interpolate = interpolate_grid(
        reduced_xgrids, length_reduced_xgrids, flavour_indices, interpolation_grid=xgrid
    )
    grid_for_writing = interpolate(df[replica])

    # Rotate the grid from the evolution basis into the export grid basis
    grid_for_writing = np.array(grid_for_writing)
    grid_for_writing = evolution_to_export_matrix @ grid_for_writing
    grid_for_writing = grid_for_writing.T.tolist()

    # Prepare a dictionary for the exportgrid
    export_grid = {}

    # Set the initial Q2 value, which will always be the same.
    export_grid["q20"] = Q0**2
    export_grid["xgrid"] = xgrid
    export_grid["replica"] = int(replica)
    export_grid["labels"] = EXPORT_LABELS

    export_grid["pdfgrid"] = grid_for_writing

    return export_grid
