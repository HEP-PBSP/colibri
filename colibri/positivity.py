"""
colibri.positivity.py

Module containing functions for positivity datasets.

Date: 07.02.2024
"""

import jax.numpy as jnp
from dataclasses import dataclass

from colibri.utils import training_validation_split, TrainValidationSplit


@dataclass(frozen=True)
class PosdataTrainValidationSplit(TrainValidationSplit):
    n_training: int
    n_validation: int


def mc_posdata_split(
    posdatasets,
    trval_seed,
    mc_validation_fraction=0.2,
    shuffle_indices=True,
):
    """
    Function for positivity training validation split.

    Note: the random split is done using the same seed as
          for data tr/val split is used.

    Parameters
    ----------
    posdatasets: list
        list of positivity datasets, see also validphys.config.parse_posdataset.

    trval_seed: jax.random.PRNGKey
        utils.trval_seed, colibri provider.

    mc_validation_fraction: float, default is 0.2

    shuffle_indices: bool, default is True

    Returns
    -------
    PosdataTrainValidationSplit
        dataclass

    """

    ndata_pos = jnp.sum(
        jnp.array(
            [
                pos_ds.load_commondata().with_cuts(pos_ds.cuts).ndata
                for pos_ds in posdatasets
            ]
        )
    )
    indices = jnp.arange(ndata_pos)

    trval_split = training_validation_split(
        indices, mc_validation_fraction, trval_seed, shuffle_indices
    )
    n_training = len(trval_split.training)
    n_validation = len(trval_split.validation)

    return PosdataTrainValidationSplit(
        **trval_split.to_dict(), n_training=n_training, n_validation=n_validation
    )


def posdata_split(posdatasets):
    """
    Function for positivity split.

    Parameters
    ----------
    posdatasets: list
        list of positivity datasets, see also validphys.config.parse_posdataset.

    Returns
    -------
    PosdataTrainValidationSplit
        dataclass

    """

    ndata_pos = jnp.sum(
        jnp.array(
            [
                pos_ds.load_commondata().with_cuts(pos_ds.cuts).ndata
                for pos_ds in posdatasets
            ]
        )
    )
    indices = jnp.arange(ndata_pos)

    return PosdataTrainValidationSplit(
        training=indices,
        validation=None,
        n_training=len(indices),
        n_validation=None,
    )
