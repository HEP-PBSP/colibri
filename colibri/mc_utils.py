"""
colibri.mc_utils.py

Module containing utils functions for the Monte Carlo fit.

"""

import jax
import jax.numpy as jnp

import os
import numpy as np

from colibri.training_validation import training_validation_split
from colibri.constants import LHAPDF_XGRID, EXPORT_LABELS
from colibri.export_results import write_exportgrid
from colibri.core import MCPseudodata

from validphys.pseudodata import make_replica
from validphys.n3fit_data import replica_mcseed

import logging

log = logging.getLogger(__name__)


def mc_pseudodata(
    central_covmat_index,
    replica_index,
    trval_seed,
    mcseed,
    shuffle_indices=True,
    positive_pseudodata=False,
    mc_validation_fraction=0.2,
):
    """Produces Monte Carlo pseudodata for the replica with index replica_index.
    The pseudodata is returned with a set of training indices, which account for
    a fraction mc_validation_fraction of the data.

    If positive_pseudodata is True, the pseudodata will be resampled until all values
    are positive"""

    central_values = central_covmat_index.central_values
    covmat = central_covmat_index.covmat
    all_indices = central_covmat_index.central_values_idx
    # Produce the same seed as in NNPDF for the pseudodata generation
    seed = replica_mcseed(replica_index, mcseed, genrep=True)

    if positive_pseudodata:
        log.warning(
            f"Sampling only positive pseudodata for all datasets - This does not provide the correct treatment of asymmetry observables"
        )
        group_positivity_mask = np.ones_like(central_values, dtype=bool)
    else:
        group_positivity_mask = None

    pseudodata = jnp.array(
        make_replica(
            central_values,
            seed,
            covmat,
            group_positivity_mask=group_positivity_mask,
        ).squeeze()
    )

    # Now select a subset of 1 - mc_validation_fraction indices to be the
    # training indices.
    if not mc_validation_fraction:
        return MCPseudodata(
            pseudodata=pseudodata,
            training_indices=all_indices,
            validation_indices=jnp.array([]),
            trval_split=False,
        )

    trval_obj = training_validation_split(
        all_indices,
        mc_validation_fraction,
        trval_seed,
        shuffle_indices,
    )

    training_indices = trval_obj.training
    validation_indices = trval_obj.validation

    return MCPseudodata(
        pseudodata=pseudodata,
        training_indices=training_indices,
        validation_indices=validation_indices,
        trval_split=True,
    )


def len_trval_data(mc_pseudodata):
    """Returns the number of training data points."""
    return len(mc_pseudodata.training_indices), len(mc_pseudodata.validation_indices)


def training_indices(mc_pseudodata):
    """Returns the training indices."""
    return mc_pseudodata.training_indices


def write_exportgrid_mc(
    parameters,
    pdf_model,
    replica_index,
    output_path,
    Q0=1.65,
    xgrid=LHAPDF_XGRID,
    export_labels=EXPORT_LABELS,
):
    """
    Similar to colibri.export_results.write_replicas but for a Monte Carlo fit.
    The main difference is that the replicas are written to a fit_replicas folder
    which is then used by the postfit script to select valid replicas.

    """
    replicas_path = str(output_path) + "/fit_replicas"

    rep_path = replicas_path + f"/replica_{replica_index}"
    if not os.path.exists(rep_path):
        os.mkdir(rep_path)

    fit_name = str(output_path).split("/")[-1]

    # Create the exportgrid
    lhapdf_interpolator = pdf_model.grid_values_func(LHAPDF_XGRID)
    n_pdf_params = len(pdf_model.param_names)

    # Rotate the grid from the evolution basis into the export grid basis
    grid_for_writing = np.array(lhapdf_interpolator(parameters[:n_pdf_params]))

    write_exportgrid(
        grid_for_writing=grid_for_writing,
        grid_name=rep_path + "/" + fit_name,
        replica_index=replica_index,
        Q0=Q0,
        xgrid=xgrid,
        export_labels=export_labels,
    )
