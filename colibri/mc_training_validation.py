"""
colibri.mc_training_validation.py

Module containing training validation dataclasses for MC fits.

Date: 11.11.2023
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass, asdict

from colibri.commondata_utils import CentralCovmatIndex


@dataclass(frozen=True)
class TrainValidationSplit:
    training: jnp.array
    validation: jnp.array

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PosdataTrainValidationSplit(TrainValidationSplit):
    n_training: int
    n_validation: int


def trval_seed(trval_index):
    """
    Returns a PRNGKey key given `trval_index` seed.
    """
    key = jax.random.PRNGKey(trval_index)
    return key


def training_validation_split(
    indices, mc_validation_fraction, random_seed, shuffle_indices=True
):
    """
    Performs training validation split on an array.

    Parameters
    ----------
    indices: jaxlib.xla_extension.Array

    mc_validation_fraction: float

    random_seed: jaxlib.xla_extension.Array
        PRNGKey, obtained as jax.random.PRNGKey(random_number)

    shuffle_indices: bool

    Returns
    -------
    dataclass
    """

    if shuffle_indices:
        # shuffle indices
        permuted_indices = jax.random.permutation(random_seed, indices)
    else:
        permuted_indices = indices

    # determine split point
    split_point = int(indices.shape[0] * (1 - mc_validation_fraction))

    # split indices
    indices_train = permuted_indices[:split_point]
    indices_validation = permuted_indices[split_point:]

    return TrainValidationSplit(training=indices_train, validation=indices_validation)


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
