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


def trval_seed(trval_index):
    """
    Returns a PRNGKey key given `trval_index` seed.
    """
    key = jax.random.PRNGKey(trval_index)
    return key


@dataclass(frozen=True)
class TrainCentralCovmatIndex(CentralCovmatIndex):
    n_training_points: int


@dataclass(frozen=True)
class ValidationCentralCovmatIndex(CentralCovmatIndex):
    n_validation_points: int


@dataclass(frozen=True)
class PosdataTrainValidationSplit(TrainValidationSplit):
    n_training: int
    n_validation: int
