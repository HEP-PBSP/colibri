"""
colibri.tests.test_data_batch

Module for testing the data_batch module.
"""

from typing import Callable, Generator

import pytest
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
    # inv_cov is optional and for this call (no general_covariance_matrix) should be None
    assert getattr(next_batch, "inv_cov", None) is None

    # When shuffle_each_epoch=False (default) fixed_batches should be available
    assert isinstance(data_batch.fixed_batches, list)
    assert len(data_batch.fixed_batches) == data_batch.num_batches


def test_data_batches_value_error():
    """Batch size larger than number of training points should raise."""
    n_training_points = 10
    training_indices = jnp.arange(n_training_points)
    # batch size too large
    with pytest.raises(ValueError):
        data_batches(training_indices, batch_size=n_training_points + 1)


def test_data_batches_with_covmat():
    """When a sqrt covariance matrix is provided, BatchSpec.inv_cov should be set
    to the corresponding L_inv rows (shape: batch_size x n_data)."""
    n_training_points = 20
    batch_size = 5
    training_indices = jnp.arange(n_training_points)

    # sqrt of scaled identity: L = sqrt(2) * I
    sqrt_cov = jnp.eye(n_training_points) * jnp.sqrt(2.0)

    db = data_batches(
        training_indices,
        batch_size=batch_size,
        general_sqrt_covariance_matrix=sqrt_cov,
        batch_seed=42,
    )

    # fixed_batches_specs should be populated and include inv_cov (L_inv rows)
    assert isinstance(db.fixed_batches, list)
    assert len(db.fixed_batches) == db.num_batches

    spec = next(db.data_batch_stream())
    assert hasattr(spec, "inv_cov")
    assert isinstance(spec.inv_cov, jax.Array)
    # inv_cov is now L_inv rows: shape (batch_size, n_data)
    assert spec.inv_cov.shape == (batch_size, n_training_points)


def test_data_batches_shuffle_each_epoch():
    """When shuffle_each_epoch=True, fixed_batches is None and inv_cov is not cached."""
    n_training_points = 30
    batch_size = 6
    training_indices = jnp.arange(n_training_points)

    db = data_batches(
        training_indices, batch_size=batch_size, shuffle_each_epoch=True, batch_seed=7
    )

    assert db.fixed_batches is None

    spec = next(db.data_batch_stream())
    assert hasattr(spec, "idx")
    assert isinstance(spec.idx, jax.Array)
    assert len(spec.idx) == batch_size
    # no cached inverse when shuffling each epoch
    assert getattr(spec, "inv_cov", None) is None
