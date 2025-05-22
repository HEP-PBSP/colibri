"""
Module for testing the data_batch module.
"""

from typing import Callable, Generator
import jax

from colibri.data_batch import data_batches, DataBatches


def test_data_batches():
    """
    Tests the function in colibri.data_batch.data_batches works as expected.
    """
    n_training_points = 100
    batch_size = 10
    batch_seed = 1
    data_batch = data_batches(n_training_points, batch_size, batch_seed)

    assert isinstance(data_batch, DataBatches)
    assert isinstance(data_batch.data_batch_stream_index, Callable)
    assert isinstance(data_batch.data_batch_stream_index(), Generator)
    assert isinstance(data_batch.num_batches, int)
    assert isinstance(data_batch.batch_size, int)

    assert data_batch.num_batches == divmod(n_training_points, batch_size)[0]
    assert data_batch.batch_size == batch_size

    batches = data_batch.data_batch_stream_index()
    next_batch = next(batches)

    assert isinstance(next_batch, jax.Array)
    assert len(next_batch) == batch_size
