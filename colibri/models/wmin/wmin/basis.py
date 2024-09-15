"""
wmin.basis.py

This module contains the functions that allow to construct a basis for the wmin parametrisation.
The main target used for the construction of the basis at this stage is the simultaneous 
satisfaction of the momentum, u-valence and d-valence sum rules.
"""

import numpy as np
import logging
import os
import yaml
import pathlib
import shutil

from reportengine import collect

from validphys.sumrules import sum_rules, KNOWN_SUM_RULES_EXPECTED
from validphys import convolution, lhaindex
from validphys.core import PDF
from validphys.mc2hessian import (
    hessian_from_lincomb,
    _compress_X,
    _get_X,
    _pdf_normalization,
    mc2hessian_xgrid,
)
from validphys.checks import check_pdf_is_montecarlo

from colibri.constants import (
    LHAPDF_XGRID,
    evolution_to_flavour_matrix,
    EXPORT_LABELS,
    FLAVOUR_TO_ID_MAPPING,
)
from colibri.export_results import write_exportgrid


log = logging.getLogger(__name__)


def sum_rules_dict(pdf, Q=1.65):
    """
    Calculates the momentum, u-valence and d-valence sum rules
    (as well as s - and c - valence)for a given PDF set.

    Parameters
    ----------
    pdf: validphys.core.PDF
        The PDF set to calculate the sum rules for.

    Q: float
        The scale at which to calculate the sum rules.

    Returns
    -------
    dict
        A dictionary containing the sum rules for the given PDF set.
    """
    return {str(pdf): sum_rules(pdf, Q)}


"""
Collects the sum rules for all PDF sets.
"""
pdfs_sum_rules = collect("sum_rules_dict", ("pdfs",))





def wmin_pdfbasis_normalization(pdf_grid, pdf_basis="intrinsic_charm"):
    """
    Imposes certain conditions on the 14 PDF flavours in the evolution basis.
    
    Intrinsic charm basis: 
    V = V15 = V24 = V35 and Sigma = T24 = T35, this means that 
    in this basis we have 8 independent PDF flavours (photon is zero). 

    Perturbative charm basis:
    V = V15 = V24 = V35 and Sigma = T15 = T24 = T35, this means that
    in this basis we have 7 independent PDF flavours (photon is zero).

    Parameters
    ----------
    pdf_grid: np.array, shape (Nfl x Ngrid)

    pdf_basis: str, default is "intrinsic_charm"
        The PDF basis to normalize to, can be either "intrinsic_charm" or "perturbative_charm".


    Returns
    -------
    np.array, an array of shape (Nreplicas x Nfl x Ngrid)
    TODO: decide whether we want to return something of shape (Nreplicas x Nfl x Ngrid) or (Nfl x Ngrid)
    """
    # check if the pdf_basis is valid
    if pdf_basis not in ["intrinsic_charm", "perturbative_charm"]:
        raise ValueError(
            f"pdf_basis must be either 'intrinsic_charm' or 'perturbative_charm', got {pdf_basis}"
        )
    
    sigma = pdf_grid[FLAVOUR_TO_ID_MAPPING["\Sigma"],:]
    valence = pdf_grid[FLAVOUR_TO_ID_MAPPING["V"],:]

    if pdf_basis == "intrinsic_charm":
        # impose the intrinsic charm basis: V = V15 = V24 = V35 and Sigma = T24 = T35
        pdf_grid[FLAVOUR_TO_ID_MAPPING["V15"],:] = valence
        pdf_grid[FLAVOUR_TO_ID_MAPPING["V24"],:] = valence
        pdf_grid[FLAVOUR_TO_ID_MAPPING["V35"],:] = valence
        pdf_grid[FLAVOUR_TO_ID_MAPPING["T24"],:] = sigma
        pdf_grid[FLAVOUR_TO_ID_MAPPING["T35"],:] = sigma
        
    elif pdf_basis == "perturbative_charm":
        # impose the perturbative charm basis: V = V15 = V24 = V35 and Sigma = T15 = T24 = T35
        pdf_grid[FLAVOUR_TO_ID_MAPPING["V15"],:] = valence
        pdf_grid[FLAVOUR_TO_ID_MAPPING["V24"],:] = valence
        pdf_grid[FLAVOUR_TO_ID_MAPPING["V35"],:] = valence
        pdf_grid[FLAVOUR_TO_ID_MAPPING["T15"],:] = sigma
        pdf_grid[FLAVOUR_TO_ID_MAPPING["T24"],:] = sigma
        pdf_grid[FLAVOUR_TO_ID_MAPPING["T35"],:] = sigma
        
    return pdf_grid


def wmin_basis_pdf_grid(
    pdfs_sum_rules, sum_rule_atol=1e-3, Q=1.65, xgrid=LHAPDF_XGRID
):
    """
    For each pdf set select replicas that simultaneously pass the
    momentum, u-valence, d-valence, s-valence, and c-valence sum rules to the required accuracy.
    Returns a pdf grid of dimension (Nreplicas x Nfl x Ngrid) that combines all the replicas from
    different PDF sets at the given scale Q

    Parameters
    ----------
    pdfs_sum_rules: list
        A list of dictionaries containing the sum rules for each PDF set.

    sum_rule_atol: float
        The absolute tolerance for the sum rules.

    Q: float
        The scale at which to calculate the sum rules.

    xgrid: array
        The xgrid to use.

    Returns
    -------
    array of shape (Nreplicas x Nfl x Ngrid)
        A pdf grid that combines all the replicas from different PDF sets at the given scale Q.
    """

    wmin_basis = []
    for sr_dict in pdfs_sum_rules:

        for pdf, sr in sr_dict.items():

            momentum_sr = sr["momentum"]
            uvalence_sr = sr["uvalence"]
            dvalence_sr = sr["dvalence"]
            svalence_sr = sr["svalence"]
            cvalence_sr = sr["cvalence"]

            momentum_sr_idx = np.where(
                np.isclose(
                    momentum_sr,
                    KNOWN_SUM_RULES_EXPECTED["momentum"],
                    atol=sum_rule_atol,
                )
            )[0]
            uvalence_sr_idx = np.where(
                np.isclose(
                    uvalence_sr,
                    KNOWN_SUM_RULES_EXPECTED["uvalence"],
                    atol=sum_rule_atol,
                )
            )[0]
            dvalence_sr_idx = np.where(
                np.isclose(
                    dvalence_sr,
                    KNOWN_SUM_RULES_EXPECTED["dvalence"],
                    atol=sum_rule_atol,
                )
            )[0]
            svalence_sr_idx = np.where(
                np.isclose(
                    svalence_sr,
                    KNOWN_SUM_RULES_EXPECTED["svalence"],
                    atol=sum_rule_atol,
                )
            )[0]
            cvalence_sr_idx = np.where(
                np.isclose(
                    cvalence_sr,
                    KNOWN_SUM_RULES_EXPECTED["cvalence"],
                    atol=sum_rule_atol,
                )
            )[0]

            # Select replicas that pass all sum rules simultaneously
            selected_replicas_idxs = np.intersect1d(
                np.intersect1d(
                    np.intersect1d(
                        np.intersect1d(momentum_sr_idx, uvalence_sr_idx),
                        dvalence_sr_idx,
                    ),
                    svalence_sr_idx,
                ),
                cvalence_sr_idx,
            )

            log.info(
                f"Selected {len(selected_replicas_idxs)} replicas for {pdf} that pass all sum rules simultaneously"
            )

            # Calculate the pdf grid for the selected replicas at the given scale Q and xgrid
            pdf_grid = convolution.evolution.grid_values(
                PDF(pdf), convolution.FK_FLAVOURS, xgrid, [Q]
            ).squeeze(-1)[selected_replicas_idxs]

            # normalize the pdf grid so that sum rules are exact
            Amomentum = momentum_sr[selected_replicas_idxs]
            Avalence = (
                uvalence_sr[selected_replicas_idxs]
                + dvalence_sr[selected_replicas_idxs]
                + svalence_sr[selected_replicas_idxs]
            )
            Avalence3 = (
                uvalence_sr[selected_replicas_idxs]
                - dvalence_sr[selected_replicas_idxs]
            )
            Avalence8 = (
                uvalence_sr[selected_replicas_idxs]
                + dvalence_sr[selected_replicas_idxs]
                - 2 * svalence_sr[selected_replicas_idxs]
            )

            pdf_grid[
                :, [FLAVOUR_TO_ID_MAPPING["\Sigma"], FLAVOUR_TO_ID_MAPPING["g"]], :
            ] /= Amomentum[:, None, None]
            pdf_grid[:, [FLAVOUR_TO_ID_MAPPING["V"]], :] *= 3 / Avalence[:, None, None]
            pdf_grid[:, [FLAVOUR_TO_ID_MAPPING["V3"]], :] *= (
                1 / Avalence3[:, None, None]
            )
            pdf_grid[:, [FLAVOUR_TO_ID_MAPPING["V8"]], :] *= (
                3 / Avalence8[:, None, None]
            )

            wmin_basis.append(pdf_grid)

    return np.concatenate(wmin_basis, axis=0)


def write_wmin_basis(
    wmin_basis_pdf_grid,
    output_path,
    Q=1.65,
    xgrid=LHAPDF_XGRID,
    export_labels=EXPORT_LABELS,
):
    """
    Writes the wmin basis at the parametrisation scale Q to the output_path.
    """
    wmin_basis_pdf_grid = wmin_basis_pdf_grid

    replicas_path = str(output_path) + "/replicas"
    if not os.path.exists(replicas_path):
        os.mkdir(replicas_path)

    for replica_index in range(1, wmin_basis_pdf_grid.shape[0]):

        rep_path = replicas_path + f"/replica_{replica_index}"
        if not os.path.exists(rep_path):
            os.mkdir(rep_path)
        fit_name = str(output_path).split("/")[-1]

        grid_name = rep_path + "/" + fit_name

        write_exportgrid(
            grid_for_writing=wmin_basis_pdf_grid[replica_index],
            grid_name=grid_name,
            replica_index=replica_index,
            Q=Q,
            xgrid=xgrid,
            export_labels=export_labels,
        )

    log.info(f"Replicas written to {replicas_path}")
    log.info(
        "Now you can evolve them with evolve_fit and then compress them with mc2_hessian"
    )


def _create_mc2hessian(
    pdf, Q, xgrid, Neig, output_path, name=None, hessian_normalization=False
):
    """
    Same as validphys.mc2hessian._create_mc2hessian but with the option not to normalize the eigenvectors.
    Note: setting hessian_normalization to False has the advantage of getting a PDF set with a more
    natural variance (since we treat replicas as samples from a posterior distribution), this, in turn,
    yields wmin coefficients that are smaller by a factor of Hessian_norm. The advantage of having this
    is that we get a smaller error in the sum rules due to miscancellation effects of the type (SR_i - SR_0).
    """
    X = _get_X(pdf, Q, xgrid, reshape=True)
    vec = _compress_X(X, Neig)
    if hessian_normalization:
        norm = _pdf_normalization(pdf)
    else:
        norm = 1.0
    return hessian_from_lincomb(pdf, vec / norm, folder=output_path, set_name=name)


@check_pdf_is_montecarlo
def mc2_pca(
    pdf,
    Q,
    Neig: int,
    output_path,
    gridname,
    installgrid: bool = False,
    hessian_normalization=False,
):
    """
    Same as validphys.mc2hessian.mc2hessian but with the option not to normalize the eigenvectors.

    Note: mc2hessian_xgrid is taken as the default xgrid that is returned by validphys.mc2hessian.mc2hessian_xgrid
    """
    log.warning("Using default xgrid from mc2hessian_xgrid for PCA.")
    result_path = _create_mc2hessian(
        pdf,
        Q=Q,
        xgrid=mc2hessian_xgrid(),
        Neig=Neig,
        output_path=output_path,
        name=gridname,
        hessian_normalization=hessian_normalization,
    )
    if installgrid:
        lhafolder = pathlib.Path(lhaindex.get_lha_datapath())
        dest = lhafolder / gridname
        if lhaindex.isinstalled(gridname):
            log.warning(
                "Target directory for new PDF, %s, already exists. "
                "Removing contents.",
                dest,
            )
            if dest.is_dir():
                shutil.rmtree(str(dest))
            else:
                dest.unlink()
        shutil.copytree(result_path, dest)
        log.info("Wmin PDF set installed at %s", dest)
