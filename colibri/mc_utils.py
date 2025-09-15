"""
colibri.mc_utils.py

Module containing utils functions for the Monte Carlo fit.

"""

import jax
import jax.numpy as jnp

import os
import numpy as np
from dataclasses import dataclass, asdict

from colibri.training_validation import training_validation_split
from colibri.constants import LHAPDF_XGRID, EXPORT_LABELS
from colibri.export_results import write_exportgrid

import logging

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPseudodata:
    pseudodata: jnp.array
    training_indices: jnp.array
    validation_indices: jnp.array
    trval_split: bool = False

    def to_dict(self):
        return asdict(self)


def mc_pseudodata(
    pseudodata_central_covmat_index,
    replica_index,
    trval_seed,
    shuffle_indices=True,
    mc_validation_fraction=0.2,
):
    """Produces Monte Carlo pseudodata for the replica with index replica_index.
    The pseudodata is returned with a set of training indices, which account for
    a fraction mc_validation_fraction of the data.
    """

    central_values = pseudodata_central_covmat_index.central_values
    covmat = pseudodata_central_covmat_index.covmat
    all_indices = pseudodata_central_covmat_index.central_values_idx

    # Generate pseudodata according to a multivariate Gaussian centred on
    # central_values and with covariance matrix covmat.
    key = jax.random.PRNGKey(replica_index)
    pseudodata = jax.random.multivariate_normal(
        key,
        central_values,
        covmat,
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


def write_exportgrid_mc(
    parameters,
    pdf_model,
    replica_index,
    output_path,
    Q=1.65,
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

    # Rotate the grid from the evolution basis into the export grid basis
    grid_for_writing = np.array(lhapdf_interpolator(parameters))

    write_exportgrid(
        grid_for_writing=grid_for_writing,
        grid_name=rep_path + "/" + fit_name,
        replica_index=replica_index,
        Q=Q,
        xgrid=xgrid,
        export_labels=export_labels,
    )
