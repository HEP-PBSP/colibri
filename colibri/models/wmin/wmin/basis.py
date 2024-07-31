"""
TODO
"""

import numpy as np
import logging

from reportengine import collect

from validphys.sumrules import sum_rules, KNOWN_SUM_RULES_EXPECTED
from validphys import convolution
from validphys.core import PDF

from colibri.constants import LHAPDF_XGRID

log = logging.getLogger(__name__)

# PID_TO_FLAVOUR = {
#     1: 'd',
#     2: 'u',
#     3: 's',
#     4: 'c',
#     5: 'b',
#     6: 't',
#     21: 'g',
# }


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


def basis_replica_selector(
    pdfs_sum_rules, sum_rule_atol=1e-3, Q=1.65, xgrid=LHAPDF_XGRID
):
    """
    For each pdf set select replicas that simultaneously pass the
    momentum, u-valence and d-valence sum rules to the required accuracy.
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
    array of
        A pdf grid that combines all the replicas from different PDF sets at the given scale Q.
    """

    wmin_basis = []
    for sr_dict in pdfs_sum_rules:

        for pdf, sr in sr_dict.items():

            momentum_sr = sr["momentum"]
            uvalence_sr = sr["uvalence"]
            dvalence_sr = sr["dvalence"]
            svalence_sr = sr["svalence"]
            # cvalence_sr = sr['cvalence']

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
            # cvalence_sr_idx = np.where(np.isclose(cvalence_sr, KNOWN_SUM_RULES_EXPECTED['cvalence'], atol=sum_rule_atol))[0]

            # Select replicas that pass all sum rules simultaneously
            selected_replicas_idxs = np.intersect1d(
                np.intersect1d(
                    np.intersect1d(momentum_sr_idx, uvalence_sr_idx), dvalence_sr_idx
                ),
                svalence_sr_idx,
            )

            log.info(
                f"Selected {len(selected_replicas_idxs)} replicas for {pdf} that pass all sum rules simultaneously"
            )

            pdf_obj = PDF(pdf)
            lpdf = pdf_obj.load()

            # Calculate the pdf grid for the selected replicas at the given scale Q and xgrid
            pdf_grid = convolution.evolution.grid_values(
                pdf_obj, convolution.FK_FLAVOURS, xgrid, [Q]
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
            
            pdf_grid[:, [1,2], :] /= Amomentum[:, None, None]
            pdf_grid[:, [3], :] /= Avalence[:, None, None]
            pdf_grid[:, [4], :] /= Avalence3[:, None, None]
            pdf_grid[:, [5], :] /= Avalence8[:, None, None]

            wmin_basis.append(pdf_grid)            

    return np.concatenate(wmin_basis, axis=0)
