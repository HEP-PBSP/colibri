"""
colibri.tests.test_data_batch

Module for testing the data_batch module.
"""

from typing import Callable, Generator

import jax
import jax.numpy as jnp

from colibri.data_batch import DataBatches, data_batches


def test_data_batches():
    """
    Tests the function in colibri.data_batch.data_batches works as expected.
    """
    n_training_points = 100
    batch_size = 10
    batch_seed = 1

    training_indices = jnp.arange(n_training_points)
    data_batch = data_batches(training_indices, batch_size, batch_seed)

    assert isinstance(data_batch, DataBatches)
    assert isinstance(data_batch.data_batch_stream, Callable)
    assert isinstance(data_batch.data_batch_stream(), Generator)
    assert isinstance(data_batch.num_batches, int)
    assert isinstance(data_batch.batch_size, int)

    assert data_batch.num_batches == divmod(n_training_points, batch_size)[0]
    assert data_batch.batch_size == batch_size

    # The stream yields BatchSpec objects with `idx` and optional `inv_cov`
    batches = data_batch.data_batch_stream()
    next_batch = next(batches)

    assert hasattr(next_batch, "idx")
    assert isinstance(next_batch.idx, jax.Array)
    assert len(next_batch.idx) == batch_size
    # inv_cov is optional and for this call (no fit_covariance_matrix) should be None
    assert getattr(next_batch, "inv_cov", None) is None

    # When shuffle_each_epoch=False (default) fixed_batches should be available
    assert isinstance(data_batch.fixed_batches, list)
    assert len(data_batch.fixed_batches) == data_batch.num_batches
