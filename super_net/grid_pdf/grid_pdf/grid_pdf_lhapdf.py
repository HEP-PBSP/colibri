"""
grid_pdf.grid_pdf_lhapdf.py

Module containing functions used to write a grid PDF fit to an LHAPDF grid.

Author: James Moore
Date: 18.12.2023
"""

from grid_pdf.grid_pdf_model import interpolate_grid
from super_net.utils import FLAVOURS_ID_MAPPINGS
from validphys.loader import Loader

from pathlib import Path

import lhapdf
import numpy as np
import os
import yaml
import shutil
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

EXPORT_LABELS = ['TBAR', 'BBAR', 'CBAR', 'SBAR', 'UBAR', 'DBAR', 'GLUON', 'D', 'U', 'S', 'C', 'B', 'T', 'PHT']

export_to_evolution = {
    '\Sigma'  : {'U': 1, 'UBAR': 1, 'D': 1, 'DBAR': 1, 'S': 1, 'SBAR': 1, 'C': 1, 'CBAR': 1 ,'B':1, 'BBAR': 1, 'T': 1, 'TBAR': 1},
    'V'        : {'U': 1, 'UBAR':-1, 'D': 1, 'DBAR':-1, 'S': 1, 'SBAR':-1, 'C': 1, 'CBAR':-1 ,'B':1, 'BBAR':-1, 'T': 1, 'TBAR':-1},

    'T3'       : {'U': 1, 'UBAR': 1, 'D':-1, 'DBAR':-1},
    'V3'       : {'U': 1, 'UBAR':-1, 'd':-1, 'DBAR': 1},

    'T8'       : {'U': 1, 'UBAR': 1, 'D': 1, 'DBAR': 1, 'S':-2, 'SBAR':-2},
    'V8'       : {'U': 1, 'UBAR':-1, 'D': 1, 'DBAR':-1, 'S':-2, 'SBAR':+2},

    'T15'      : {'U': 1, 'UBAR': 1, 'D': 1, 'DBAR': 1, 'S': 1, 'SBAR': 1, 'c':-3, 'CBAR':-3},
    'V15'      : {'U': 1, 'UBAR':-1, 'D': 1, 'DBAR':-1, 'S': 1, 'SBAR':-1, 'c':-3, 'CBAR':+3},


    'T24'      : {'U': 1, 'UBAR': 1, 'D': 1, 'DBAR': 1, 'S': 1, 'SBAR': 1, 'C': 1, 'CBAR': 1, 'B':-4, 'BBAR':-4},
    'V24'      : {'U': 1, 'UBAR':-1, 'D': 1, 'DBAR':-1, 'S': 1, 'SBAR':-1, 'C': 1, 'CBAR':-1, 'B':-4, 'BBAR':+4},

    'T35'      : {'U': 1, 'UBAR': 1, 'D': 1, 'DBAR': 1, 'S': 1, 'SBAR': 1, 'C': 1, 'CBAR': 1, 'B': 1, 'BBAR': 1, 'T':-5, 'TBAR':-5},
    'V35'      : {'U': 1, 'UBAR':-1, 'D': 1, 'DBAR':-1, 'S': 1, 'SBAR':-1, 'C': 1, 'CBAR':-1, 'B': 1, 'BBAR':-1, 'T':-5, 'TBAR':+5},

    'g'        : {'GLUON':1},
    'photon'   : {'PHT':1},
    }

# Construct the inverse transformation from evolution to export
num_flav = len(FLAVOURS_ID_MAPPINGS)
export_to_evolution_matrix = np.zeros((num_flav, num_flav))
for i in range(num_flav):
    j = 0
    for flav in EXPORT_LABELS:
        if flav in export_to_evolution[FLAVOURS_ID_MAPPINGS[i]].keys():
            export_to_evolution_matrix[i,j] = export_to_evolution[FLAVOURS_ID_MAPPINGS[i]][flav]
        else:
            export_to_evolution_matrix[i,j] = 0
        j += 1

evolution_to_export_matrix = np.linalg.inv(export_to_evolution_matrix)    

STANDARD_XGRID = [1e-09, 1.29708482343957e-09, 1.68242903474257e-09, 2.18225315420583e-09, 2.83056741739819e-09,
  3.67148597892941e-09, 4.76222862935315e-09, 6.1770142737618e-09, 8.01211109898438e-09,
  1.03923870607245e-08, 1.34798064073805e-08, 1.74844503691778e-08, 2.26788118881103e-08,
  2.94163370300835e-08, 3.81554746595878e-08, 4.94908707232129e-08, 6.41938295708371e-08,
  8.32647951986859e-08, 1.08001422993829e-07, 1.4008687308113e-07, 1.81704331793772e-07,
  2.35685551545377e-07, 3.05703512595323e-07, 3.96522309841747e-07, 5.1432125723657e-07,
  6.67115245136676e-07, 8.65299922973143e-07, 1.12235875241487e-06, 1.45577995547683e-06,
  1.88824560514613e-06, 2.44917352454946e-06, 3.17671650028717e-06, 4.12035415232797e-06,
  5.3442526575209e-06, 6.93161897806315e-06, 8.99034258238145e-06, 1.16603030112258e-05,
  1.51228312288769e-05, 1.96129529349212e-05, 2.54352207134502e-05, 3.29841683435992e-05,
  4.27707053972016e-05, 5.54561248105849e-05, 7.18958313632514e-05, 9.31954227979614e-05,
  0.00012078236773133, 0.000156497209466554, 0.000202708936328495, 0.000262459799331951,
  0.000339645244168985, 0.000439234443000422, 0.000567535660104533, 0.000732507615725537,
  0.000944112105452451, 0.00121469317686978, 0.00155935306118224, 0.00199627451141338,
  0.00254691493736552, 0.00323597510213126, 0.00409103436509565, 0.00514175977083962,
  0.00641865096062317, 0.00795137940306351, 0.009766899996241, 0.0118876139251364,
  0.0143298947643919, 0.0171032279460271, 0.0202100733925079, 0.0236463971369542,
  0.0274026915728357, 0.0314652506132444, 0.0358174829282429, 0.0404411060163317,
  0.0453171343973807, 0.0504266347950069, 0.0557512610084339, 0.0612736019390519,
  0.0669773829498255, 0.0728475589986517, 0.0788703322292727, 0.0850331197801452,
  0.0913244910278679, 0.0977340879783772, 0.104252538208639, 0.110871366547237, 0.117582909372878,
  0.124380233801599, 0.131257062945031, 0.138207707707289, 0.145227005135651, 0.152310263065985,
  0.159453210652156, 0.166651954293987, 0.173902938455578, 0.181202910873333, 0.188548891679097,
  0.195938145999193, 0.203368159629765, 0.210836617429103, 0.218341384106561, 0.225880487124065,
  0.233452101459503, 0.241054536011681, 0.248686221452762, 0.256345699358723, 0.264031612468684,
  0.271742695942783, 0.279477769504149, 0.287235730364833, 0.295015546847664, 0.302816252626866,
  0.310636941519503, 0.318476762768082, 0.326334916761672, 0.334210651149156, 0.342103257303627,
  0.350012067101685, 0.357936449985571, 0.365875810279643, 0.373829584735962, 0.381797240286494,
  0.389778271981947, 0.397772201099286, 0.40577857340234, 0.413796957540671, 0.421826943574548,
  0.429868141614175, 0.437920180563205, 0.44598270695699, 0.454055383887562, 0.462137890007651,
  0.470229918607142, 0.478331176755675, 0.486441384506059, 0.494560274153348, 0.502687589545177,
  0.510823085439086, 0.518966526903235, 0.527117688756998, 0.535276355048428, 0.543442318565661,
  0.551615380379768, 0.559795349416641, 0.5679820420558, 0.576175281754088, 0.584374898692498,
  0.59258072944444, 0.60079261666395, 0.609010408792398, 0.61723395978245, 0.625463128838069,
  0.633697780169485, 0.641937782762089, 0.650183010158361, 0.658433340251944, 0.666688655093089,
  0.674948840704708, 0.683213786908386, 0.691483387159697, 0.699757538392251, 0.708036140869916,
  0.716319098046733, 0.724606316434025, 0.732897705474271, 0.741193177421404, 0.749492647227008,
  0.757796032432224, 0.766103253064927, 0.774414231541921, 0.782728892575836, 0.791047163086478,
  0.799368972116378, 0.807694250750291, 0.816022932038457, 0.824354950923382, 0.832690244169987,
  0.841028750298844, 0.8493704095226, 0.857715163684985, 0.866062956202683, 0.874413732009721,
  0.882767437504206, 0.891124020497459, 0.899483430165226, 0.907845617001021, 0.916210532771399,
  0.924578130473112, 0.932948364292029, 0.941321189563734, 0.949696562735755, 0.958074441331298,
  0.966454783914439, 0.974837550056705, 0.983222700304978, 0.991610196150662, 1.0]

def lhapdf_grid_pdf_ultranest_result(
        ultranest_grid_fit,
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
    ns_replicas_path = str(output_path) + '/ns_replicas'
    if not os.path.exists(ns_replicas_path):
        os.mkdir(ns_replicas_path)

    fit_name = str(output_path).split("/")[-1]

    for i in range(n_posterior_samples):
        rep_path = ns_replicas_path + '/replica_' + str(i+1)
        if not os.path.exists(rep_path):
            os.mkdir(rep_path)
        exportgrid = write_exportgrid(ultranest_grid_fit, reduced_xgrids, length_reduced_xgrids, flavour_indices, i)
        with open(rep_path+'/'+fit_name+'.exportgrid', 'w') as outfile:
            yaml.dump(exportgrid, outfile)

    # Now run EKO on the exportgrids to complete the PDF evolution
    log.info(f"Loading eko from theory {theoryid.id}")
    eko_path = (Loader().check_theoryID(theoryid.id).path) / "eko.tar"

    with eko.EKO.edit(eko_path) as eko_op:
        x_grid_obj = eko.interpolation.XGrid(STANDARD_XGRID)
        eko.io.manipulate.xgrid_reshape(eko_op, targetgrid=x_grid_obj, inputgrid=x_grid_obj)

    # Load the export grids into a dictionary
    initial_PDFs_dict = {}
    for yaml_file in Path(ns_replicas_path).glob(f"replica_*/{output_path.name}.exportgrid"):
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
        info["XMin"] = float(STANDARD_XGRID[0])
        info["XMax"] = float(STANDARD_XGRID[-1])
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
            log.info("Evolving replica " + str(progress) + " of " + str(n_posterior_samples))
            evolved_blocks = evolve_exportgrid(pdf_data, eko_op, STANDARD_XGRID)
            replica_num = replica.removeprefix("replica_")
            genpdf.export.dump_blocks(
                Path(lhapdf_destination),
                int(replica_num),
                evolved_blocks,
                pdf_type=f"PdfType: replica\nFromMCReplica: {replica_num}\n"
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
            interp1d(self.x_grid, self.pdf_grid[pid], kind="cubic") for pid in range(len(PIDS_DICT))
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
            pdf_xq2, xgrid=targetgrid, sorted_q2grid=q2grid, pids=basis_rotation.flavor_basis_pids
        )
        blocks.append(block)

    return blocks

def write_exportgrid(df, reduced_xgrids, length_reduced_xgrids, flavour_indices, replica, Q0=1.65, xgrid=STANDARD_XGRID):
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
    interpolate = interpolate_grid(reduced_xgrids, length_reduced_xgrids, flavour_indices, interpolation_grid=xgrid)
    grid_for_writing = interpolate(df[replica])

    # Rotate the grid from the evolution basis into the export grid basis
    grid_for_writing = np.array(grid_for_writing)
    grid_for_writing = evolution_to_export_matrix @ grid_for_writing
    grid_for_writing = grid_for_writing.T.tolist()

    # Prepare a dictionary for the exportgrid
    export_grid = {}

    # Set the initial Q2 value, which will always be the same.
    export_grid['q20'] = Q0**2
    export_grid['xgrid'] = xgrid
    export_grid['replica'] = int(replica)
    export_grid['labels'] = EXPORT_LABELS

    export_grid['pdfgrid'] = grid_for_writing

    return export_grid 
