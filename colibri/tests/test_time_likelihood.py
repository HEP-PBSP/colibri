"""
colibri.tests.test_time_likelihood.py

Tests for the time_likelihood module.
"""

import logging
from unittest.mock import patch, call
import csv

import jax
import jax.numpy as jnp
import pytest

from colibri.time_likelihood import time_log_likelihood
from colibri.tests.conftest import TEST_FORWARD_MAP_DIS, TEST_FORWARD_MAP_DIS

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def mock_log_likelihood():
    """Mock log likelihood function that returns a scalar per parameter set."""

    def likelihood(params):
        # Simple mock: return sum of parameters
        return jnp.sum(params, axis=-1)

    return likelihood


@pytest.fixture
def mock_param_initialiser_settings():
    """Mock parameter initializer settings."""
    return {"type": "uniform", "random_seed": 42, "min_val": -1.0, "max_val": 1.0}


@pytest.fixture
def tmp_output_path(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


def test_time_log_likelihood_default_sizes(
    mock_log_likelihood, mock_param_initialiser_settings, tmp_output_path
):
    """Test time_log_likelihood with default batch sizes."""

    with patch("colibri.time_likelihood.pdf_initial_parameters") as mock_init:
        # Mock pdf_initial_parameters to return simple arrays
        mock_init.side_effect = lambda _, __, idx: jnp.array([0.1 * idx, 0.2 * idx])

        sizes, times = time_log_likelihood(
            mock_log_likelihood,
            mock_param_initialiser_settings,
            TEST_FORWARD_MAP_DIS,
            tmp_output_path,
            batch_sample_sizes=[1, 10, 100],  # Use small sizes for testing
        )

    # Check that sizes are returned correctly
    assert sizes == [1, 10, 100]

    # Check that times list has correct length
    assert len(times) == 3

    # Check that all times are positive
    assert all(t > 0 for t in times)

    # Check that CSV file was created
    csv_path = tmp_output_path / "log_likelihood_times.csv"
    assert csv_path.exists()

    # Read and verify CSV contents
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        assert headers == ["batch_size", "avg_time_seconds", "relative_time"]

        rows = list(reader)
        assert len(rows) == 3

        # Check first row
        assert int(rows[0][0]) == 1
        assert float(rows[0][1]) > 0
        assert float(rows[0][2]) == 1.0  # First relative time should be 1.0


def test_time_log_likelihood_custom_sizes(
    mock_log_likelihood, mock_param_initialiser_settings, tmp_output_path
):
    """Test time_log_likelihood with custom batch sizes."""

    custom_sizes = [5, 20, 50]

    with patch("colibri.time_likelihood.pdf_initial_parameters") as mock_init:
        mock_init.side_effect = lambda _, __, idx: jnp.array([0.1 * idx, 0.2 * idx])

        sizes, times = time_log_likelihood(
            mock_log_likelihood,
            mock_param_initialiser_settings,
            TEST_FORWARD_MAP_DIS,
            tmp_output_path,
            batch_sample_sizes=custom_sizes,
        )

    # Verify custom sizes were used
    assert sizes == custom_sizes
    assert len(times) == len(custom_sizes)


def test_time_log_likelihood_generates_correct_number_of_samples(
    mock_log_likelihood, mock_param_initialiser_settings, tmp_output_path
):
    """Test that the correct number of samples are generated."""

    custom_sizes = [2, 5]
    max_size = max(custom_sizes)

    with patch("colibri.time_likelihood.pdf_initial_parameters") as mock_init:
        mock_init.side_effect = lambda _, __, idx: jnp.array([float(idx), float(idx)])

        time_log_likelihood(
            mock_log_likelihood,
            mock_param_initialiser_settings,
            TEST_FORWARD_MAP_DIS,
            tmp_output_path,
            batch_sample_sizes=custom_sizes,
        )

        # Check that pdf_initial_parameters was called max_size times
        assert mock_init.call_count == max_size

        # Check that it was called with correct replica indices
        expected_calls = [
            call(TEST_FORWARD_MAP_DIS, mock_param_initialiser_settings, i)
            for i in range(max_size)
        ]
        mock_init.assert_has_calls(expected_calls)


def test_time_log_likelihood_vectorization(
    mock_param_initialiser_settings, tmp_output_path
):
    """Test that the likelihood is properly vectorized."""

    call_count = 0

    def counting_likelihood(params):
        nonlocal call_count
        call_count += 1

        return jnp.sum(params, axis=-1) if params.ndim > 1 else jnp.sum(params)

    with patch("colibri.time_likelihood.pdf_initial_parameters") as mock_init:
        mock_init.side_effect = lambda _, __, ___: jnp.array([0.1, 0.2])

        time_log_likelihood(
            counting_likelihood,
            mock_param_initialiser_settings,
            TEST_FORWARD_MAP_DIS,
            tmp_output_path,
            batch_sample_sizes=[2, 5],
        )

    assert call_count > 0


def test_time_log_likelihood_csv_format(
    mock_log_likelihood, mock_param_initialiser_settings, tmp_output_path
):
    """Test that CSV is written with correct format."""

    with patch("colibri.time_likelihood.pdf_initial_parameters") as mock_init:
        mock_init.side_effect = lambda _, __, ___: jnp.array([0.1, 0.2])

        _, __ = time_log_likelihood(
            mock_log_likelihood,
            mock_param_initialiser_settings,
            TEST_FORWARD_MAP_DIS,
            tmp_output_path,
            batch_sample_sizes=[3, 7],
        )

    csv_path = tmp_output_path / "log_likelihood_times.csv"

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

        # Check headers
        assert headers == ["batch_size", "avg_time_seconds", "relative_time"]

        # Check that we have the right number of rows
        assert len(rows) == 2

        # Check that batch sizes match
        assert int(rows[0][0]) == 3
        assert int(rows[1][0]) == 7

        # Check relative times
        relative_time_1 = float(rows[0][2])
        relative_time_2 = float(rows[1][2])

        assert relative_time_1 == 1.0  # First should always be 1.0
        assert relative_time_2 > 0  # Second should be positive


def test_time_log_likelihood_none_uses_defaults(
    mock_log_likelihood, mock_param_initialiser_settings, tmp_output_path, caplog
):
    """Test that passing None for batch_sample_sizes uses default sizes."""
    with patch("colibri.time_likelihood.pdf_initial_parameters") as mock_init:
        mock_init.side_effect = lambda _, __, ___: jnp.array([0.1, 0.2])
        with caplog.at_level(logging.INFO):
            sizes, _ = time_log_likelihood(
                mock_log_likelihood,
                mock_param_initialiser_settings,
                TEST_FORWARD_MAP_DIS,
                tmp_output_path,
                batch_sample_sizes=None,
            )
            # Check that default sizes were used
            default_sizes = [1, 10, 100, 1000, 5000, 10000, 20000, 50000, 100000]
            assert sizes == default_sizes
            assert "Using default batch sample sizes" in caplog.text


def test_time_log_likelihood_handles_exception_during_warmup(
    mock_param_initialiser_settings, tmp_output_path, caplog
):
    """Test that exceptions during warm-up are properly logged and raised."""

    def failing_likelihood(_):
        raise RuntimeError("Simulated warm-up failure")

    with patch("colibri.time_likelihood.pdf_initial_parameters") as mock_init:
        mock_init.side_effect = lambda _, __, ___: jnp.array([0.1, 0.2])

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="Simulated warm-up failure"):
                time_log_likelihood(
                    failing_likelihood,
                    mock_param_initialiser_settings,
                    TEST_FORWARD_MAP_DIS,
                    tmp_output_path,
                    batch_sample_sizes=[2, 5],
                )

        assert "Warm-up failed" in caplog.text


def test_time_log_likelihood_handles_exception_during_timing(
    mock_param_initialiser_settings, tmp_output_path, caplog
):
    """
    Test that exceptions during timing are caught and logged properly.
    """

    with patch("colibri.time_likelihood.pdf_initial_parameters") as mock_init:
        mock_init.side_effect = lambda _, __, ___: jnp.array([0.1, 0.2])

        original_block = jax.block_until_ready
        block_calls = {"n": 0}

        def failing_block(x):
            block_calls["n"] += 1

            if block_calls["n"] == 2:
                raise RuntimeError("Simulated timing failure")
            return original_block(x)

        with patch("jax.block_until_ready", side_effect=failing_block):
            with caplog.at_level(logging.ERROR):
                sizes, times = time_log_likelihood(
                    lambda params: jnp.sum(params, axis=-1),
                    mock_param_initialiser_settings,
                    TEST_FORWARD_MAP_DIS,
                    tmp_output_path,
                    batch_sample_sizes=[2, 5, 10],
                )

        assert "Error at batch size" in caplog.text
        assert "No batch sizes were successfully timed" in caplog.text


def test_time_log_likelihood_successful_partial_run(
    mock_param_initialiser_settings, tmp_output_path, caplog
):
    """
    Test when some batch sizes succeed before an error occurs.

    We let the first batch size complete fully, then fail on the FIRST timing
    iteration of the second batch size, by raising from jax.block_until_ready.
    """

    n_repeats = 100  # must match implementation

    with patch("colibri.time_likelihood.pdf_initial_parameters") as mock_init:
        mock_init.side_effect = lambda _, __, ___: jnp.array([0.1, 0.2])

        original_block = jax.block_until_ready
        block_calls = {"n": 0}

        fail_on = 1 + n_repeats + 1 + 1  # 103

        def failing_block(x):
            block_calls["n"] += 1
            if block_calls["n"] == fail_on:
                raise RuntimeError("Fail on second batch")
            return original_block(x)

        with patch("jax.block_until_ready", side_effect=failing_block):
            with caplog.at_level(logging.WARNING):
                sizes, times = time_log_likelihood(
                    lambda params: jnp.sum(params, axis=-1),
                    mock_param_initialiser_settings,
                    TEST_FORWARD_MAP_DIS,
                    tmp_output_path,
                    batch_sample_sizes=[2, 5, 10],
                )

        assert sizes == [2]
        assert len(times) == 1
        assert "Stopping timing" in caplog.text

        assert "Results for batch sizes up to" in caplog.text


def test_time_log_likelihood_all_sizes_successful(
    mock_log_likelihood, mock_param_initialiser_settings, tmp_output_path, caplog
):
    """Test the success path where all sizes complete successfully."""

    with patch("colibri.time_likelihood.pdf_initial_parameters") as mock_init:
        mock_init.side_effect = lambda _, __, ___: jnp.array([0.1, 0.2])

        with caplog.at_level(logging.INFO):
            sizes, times = time_log_likelihood(
                mock_log_likelihood,
                mock_param_initialiser_settings,
                TEST_FORWARD_MAP_DIS,
                tmp_output_path,
                batch_sample_sizes=[2, 5, 10],
            )

    # All sizes should succeed
    assert len(sizes) == 3
    assert sizes == [2, 5, 10]
    assert len(times) == 3

    # Check success logging
    assert "Timing completed for 3 batch sizes" in caplog.text
    assert "Final results saved to" in caplog.text


def test_time_log_likelihood_relative_time_calculation(
    mock_log_likelihood, mock_param_initialiser_settings, tmp_output_path
):
    """Test that relative times are calculated correctly."""

    with patch("colibri.time_likelihood.pdf_initial_parameters") as mock_init:
        mock_init.side_effect = lambda _, __, ___: jnp.array([0.1, 0.2])

        _, __ = time_log_likelihood(
            mock_log_likelihood,
            mock_param_initialiser_settings,
            TEST_FORWARD_MAP_DIS,
            tmp_output_path,
            batch_sample_sizes=[2, 5, 10],
        )

    csv_path = tmp_output_path / "log_likelihood_times.csv"

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        rows = list(reader)

        relative_times = [float(row[2]) for row in rows]

        assert relative_times[0] == 1.0  # First relative time should always be 1.0
        assert all(
            rt > 0 for rt in relative_times
        )  # All relative times should be positive

        # Relative times should be calculated as time/first_time
        first_time = float(rows[0][1])
        for i, row in enumerate(rows):
            expected_relative = float(row[1]) / first_time
            actual_relative = float(row[2])
            # Allow for small floating point differences
            assert abs(expected_relative - actual_relative) < 1e-6


def test_time_log_likelihood_jax_block_until_ready(
    mock_param_initialiser_settings, tmp_output_path
):
    """Test that jax.block_until_ready is called to ensure proper timing."""

    ready_calls = []
    original_block = jax.block_until_ready

    def tracking_block(x):
        ready_calls.append(x)
        return original_block(x)

    with patch("colibri.time_likelihood.pdf_initial_parameters") as mock_init:
        mock_init.side_effect = lambda _, __, ___: jnp.array([0.1, 0.2])

        with patch("jax.block_until_ready", side_effect=tracking_block):

            def simple_likelihood(params):
                return jnp.sum(params, axis=-1)

            time_log_likelihood(
                simple_likelihood,
                mock_param_initialiser_settings,
                TEST_FORWARD_MAP_DIS,
                tmp_output_path,
                batch_sample_sizes=[2, 5],
            )

    assert len(ready_calls) >= 201


def test_time_log_likelihood_with_single_batch_size(
    mock_log_likelihood, mock_param_initialiser_settings, tmp_output_path
):
    """Test with only a single batch size."""

    with patch("colibri.time_likelihood.pdf_initial_parameters") as mock_init:
        mock_init.side_effect = lambda _, __, ___: jnp.array([0.1, 0.2])

        sizes, times = time_log_likelihood(
            mock_log_likelihood,
            mock_param_initialiser_settings,
            TEST_FORWARD_MAP_DIS,
            tmp_output_path,
            batch_sample_sizes=[10],
        )

    assert sizes == [10]
    assert len(times) == 1
    assert times[0] > 0

    # Check CSV
    csv_path = tmp_output_path / "log_likelihood_times.csv"
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        rows = list(reader)
        assert len(rows) == 1
        assert float(rows[0][2]) == 1.0  # Relative time should be 1.0
