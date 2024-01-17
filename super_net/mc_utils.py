"""
super_net.mc_utils.py

Module containing functions for the Monte Carlo fit.

Date: 17.1.2023
"""

import jax
import jax.numpy as jnp

import dataclass

from super_net.utils import training_validation_split

@dataclass(frozen=True)
class MCPseudodata:
    pseudodata: jnp.array
    training_indices: jnp.array
    validation_indices: jnp.array

    def to_dict(self):
        return asdict(self)

def mc_pseudodata(
    central_covmat_index,
    replica_index,
    trval_seed,
    shuffle_indices=True,
    mc_validation_fraction=0.2,
):
    """Produces Monte Carlo pseudodata for the replica with index replica_index.
    The pseudodata is returned with a set of training indices, which account for
    a fraction mc_validation_fraction of the data.
    """

    central_values = central_covmat_index.central_values
    covmat = central_covmat_index.covmat
    all_indices = central_covmat_index.central_values_idx

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
    )
