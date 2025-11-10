"""
colibri.data_batch.py

Module containing data batches provider.
"""

from typing import Callable, Optional, Iterator, List, NamedTuple
from dataclasses import dataclass
import logging

import jax
import jax.numpy as jnp

log = logging.getLogger(__name__)


class BatchSpec(NamedTuple):
    idx: jnp.ndarray
    inv_cov: Optional[jnp.ndarray] = None


@dataclass(frozen=True)
class DataBatches:
    data_batch_stream: Callable[[], Iterator[BatchSpec]]
    num_batches: int
    batch_size: int
    batch_seed: int
    # Optional cache for visibility / reuse
    fixed_batches: Optional[List[BatchSpec]] = None


def data_batches(
    training_indices,
    batch_size=None,
    batch_seed=1,
    fit_covariance_matrix=None,
    shuffle_each_epoch=False,
) -> DataBatches:
    """
    Parameters
    ----------
    training_indices: jax.Array
        Indices of training data points.

    batch_size: int, default is None which sets it to n_training_points

    batch_seed: int, default is 1

    fit_covariance_matrix: jax.Array, optional
        If provided together with shuffle_each_epoch=False, fixed batches are
        precomputed once and the corresponding inverse covariance submatrices
        are cached for reuse. This avoids inverting within the likelihood at
        every step.

    shuffle_each_epoch: bool, default False
        If True, a new random permutation is generated each epoch.
        If False, batches are fixed once per seed (enables caching inverses).

    Returns
    -------
    DataBatches dataclass
    """

    n_training_points = len(training_indices)

    if not batch_size:
        log.warning(
            f"Batch size not specified, setting it to the full number of data points {n_training_points}"
        )
        batch_size = n_training_points

    if batch_size > n_training_points:
        raise ValueError(
            f"Size of batch = {batch_size} should be smaller or equal to the number of data {n_training_points}"
        )

    num_complete_batches, _leftover = divmod(n_training_points, batch_size)
    # Discard leftover to avoid shape changes / recompiles
    num_batches = num_complete_batches

    def _make_perm(rng_key: jax.Array) -> jax.Array:
        return jax.random.permutation(rng_key, jnp.arange(n_training_points))

    def _slice_batches_from_perm(perm: jax.Array) -> List[jax.Array]:
        # Slice contiguous chunks without Python loops over jnp (keeps it simple/readability)
        return [perm[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)]

    key = jax.random.PRNGKey(batch_seed)

    fixed_batches_specs = None

    if not shuffle_each_epoch:
        # Single permutation → fixed batches
        perm0 = _make_perm(key)
        fixed_batches = _slice_batches_from_perm(perm0)

        if fit_covariance_matrix is not None:
            train_covmat = fit_covariance_matrix[training_indices][:, training_indices]
            fixed_batches_specs = [
                BatchSpec(
                    idx=b,
                    inv_cov=jnp.linalg.inv(train_covmat[b][:, b]),
                )
                for b in fixed_batches
            ]
        else:
            fixed_batches_specs = [BatchSpec(idx=b) for b in fixed_batches]

    def data_batch_stream() -> Iterator[BatchSpec]:
        nonlocal key
        if shuffle_each_epoch:
            while True:
                key, subkey = jax.random.split(key)
                perm = _make_perm(subkey)
                for b in _slice_batches_from_perm(perm):
                    # no cached inverse when shuffling each epoch
                    yield BatchSpec(idx=b)
        else:
            while True:
                # cycle through precomputed specs (with inv_cov if available)
                for spec in fixed_batches_specs:
                    yield spec

    return DataBatches(
        data_batch_stream=data_batch_stream,
        num_batches=num_batches,
        batch_size=batch_size,
        batch_seed=batch_seed,
        fixed_batches=fixed_batches_specs,
    )
