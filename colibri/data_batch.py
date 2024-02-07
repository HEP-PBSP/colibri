"""
colibri.data_batch.py

Module containing data batches provider.

Author: Mark N. Costantini
Date: 11.11.2023
"""

from typing import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class DataBatches:
    data_batch_stream_index: Callable
    num_batches: int
    batch_size: int


def data_batches(n_training_points, batch_size, batch_seed=1):
    """
    Parameters
    ----------
    n_training_points: int

    batch_size: int

    batch_seed: int, default is 1

    Returns
    -------
    DataBatches dataclass
    """

    if batch_size > n_training_points:
        raise ValueError(
            f"Size of batch = {batch_size} should be smaller or equal to the number of data {n_training_points}"
        )

    num_complete_batches, leftover = divmod(n_training_points, batch_size)
    # discard leftover to avoid the a slow down due to having to recompile make_chi2 functionm
    num_batches = num_complete_batches  # + bool(leftover)

    def data_batch_stream_index():
        """
        Mutable objects like generators should not be used as valiphys
        actions, hence the need to add the closure.
        """

        key = jax.random.PRNGKey(batch_seed)

        while True:
            perm = jax.random.permutation(key, jnp.arange(n_training_points))

            for i in range(num_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                yield batch_idx

    return DataBatches(
        data_batch_stream_index=data_batch_stream_index,
        num_batches=num_batches,
        batch_size=batch_size,
    )
