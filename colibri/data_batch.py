"""
colibri.data_batch.py

Module containing data batches provider.
"""

from typing import Iterator, List
import logging

import jax
import jax.lax.linalg as jlinalg
import jax.numpy as jnp
from colibri.core import DataBatches, BatchSpec

log = logging.getLogger(__name__)


def data_batches(
    training_indices,
    batch_size=None,
    batch_seed=1,
    general_sqrt_covariance_matrix=None,
    shuffle_each_epoch=False,
) -> DataBatches:
    """
    Parameters
    ----------
    training_indices: jax.Array
        Indices of training data points.

    batch_size: int, default is None which sets it to n_training_points

    batch_seed: int, default is 1

    general_sqrt_covariance_matrix: jax.Array, optional
        Lower triangular Cholesky factor L of the full covariance matrix. If
        provided together with shuffle_each_epoch=False, fixed batches are
        precomputed once and the rows of L^{-1} corresponding to each batch
        are cached in ``BatchSpec.inv_cov`` for reuse. These rows have shape
        ``(batch_size, n_data)`` and are passed directly to chi2 to whiten
        predictions without reforming the covariance.

    shuffle_each_epoch: bool, default False
        If True, a new random permutation is generated each epoch.
        If False, batches are fixed once per seed (enables caching L_inv rows).

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

        if general_sqrt_covariance_matrix is not None:
            n = general_sqrt_covariance_matrix.shape[0]
            L_inv = jlinalg.triangular_solve(
                general_sqrt_covariance_matrix,
                jnp.eye(n),
                left_side=True,
                lower=True,
            )
            train_L_inv = L_inv[training_indices]  # (n_train, n_data) rows
            fixed_batches_specs = [
                BatchSpec(idx=b, inv_cov=train_L_inv[b]) for b in fixed_batches
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
