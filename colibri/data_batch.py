"""
colibri.data_batch.py

Module containing data batches provider.
"""

from typing import Callable, Optional, Iterator, Tuple, List
from dataclasses import dataclass
import logging

import jax
import jax.numpy as jnp

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataBatches:
    data_batch_stream_index: Callable[[], Iterator[jax.Array]]
    num_batches: int
    batch_size: int
    batch_seed: int
    # Optional advanced batching: fixed batches with precomputed inverses
    data_batch_stream_index_and_inv: Optional[
        Callable[[], Iterator[Tuple[jax.Array, jax.Array]]]
    ] = None
    fixed_batches: Optional[List[jax.Array]] = None
    fixed_inv_covmats: Optional[List[jax.Array]] = None


def data_batches(
    n_training_points: int,
    batch_size: Optional[int],
    batch_seed: int = 1,
    fit_covariance_matrix: Optional[jax.Array] = None,
    shuffle_each_epoch: bool = False,
) -> DataBatches:
    """
    Parameters
    ----------
    n_training_points: int

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

    def _precompute_inv_for_batches(
        batches: List[jax.Array], cov: jax.Array
    ) -> List[jax.Array]:
        invs = []
        for b in batches:
            cov_b = cov[b][:, b]
            invs.append(jnp.linalg.inv(cov_b))
        return invs

    key = jax.random.PRNGKey(batch_seed)

    fixed_batches = None
    fixed_inv_covmats = None

    if not shuffle_each_epoch:
        # Single permutation → fixed batches
        perm0 = _make_perm(key)
        fixed_batches = _slice_batches_from_perm(perm0)

        if fit_covariance_matrix is not None:
            fixed_inv_covmats = _precompute_inv_for_batches(
                fixed_batches, fit_covariance_matrix
            )

    # --- generators ---
    def data_batch_stream_index() -> Iterator[jax.Array]:
        """
        Yields indices of each batch, epoch after epoch.
        - shuffle_each_epoch=True: new permutation every epoch (key is advanced).
        - shuffle_each_epoch=False: cycles over fixed batches.
        """
        nonlocal key  # important: advance the outer key safely inside the closure

        if shuffle_each_epoch:
            while True:
                key, subkey = jax.random.split(key)
                perm = _make_perm(subkey)
                for b in _slice_batches_from_perm(perm):
                    yield b
        else:
            # Cycle forever over fixed batches
            while True:
                for b in fixed_batches:
                    yield b

    data_batch_stream_index_and_inv = None

    if (fixed_batches is not None) and (fixed_inv_covmats is not None):

        def _gen_idx_and_inv() -> Iterator[Tuple[jax.Array, jax.Array]]:
            while True:
                for b, inv in zip(fixed_batches, fixed_inv_covmats):
                    yield b, inv

        data_batch_stream_index_and_inv = _gen_idx_and_inv

    return DataBatches(
        data_batch_stream_index=data_batch_stream_index,
        num_batches=num_batches,
        batch_size=batch_size,
        batch_seed=batch_seed,
        data_batch_stream_index_and_inv=data_batch_stream_index_and_inv,
        fixed_batches=fixed_batches,
        fixed_inv_covmats=fixed_inv_covmats,
    )
