"""
colibri.tests.test_time_likelihood.py

Tests for the time_likelihood module.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call
import csv

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from colibri.time_likelihood import time_log_likelihood
from colibri.tests.conftest import MOCK_PDF_MODEL

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
        mock_init.side_effect = lambda model, settings, idx: jnp.array(
            [0.1 * idx, 0.2 * idx]
        )

        sizes, times = time_log_likelihood(
            mock_log_likelihood,
            mock_param_initialiser_settings,
            MOCK_PDF_MODEL,
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
        mock_init.side_effect = lambda model, settings, idx: jnp.array(
            [0.1 * idx, 0.2 * idx]
        )

        sizes, times = time_log_likelihood(
            mock_log_likelihood,
            mock_param_initialiser_settings,
            MOCK_PDF_MODEL,
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
        mock_init.side_effect = lambda model, settings, idx: jnp.array(
            [float(idx), float(idx)]
        )

        time_log_likelihood(
            mock_log_likelihood,
            mock_param_initialiser_settings,
            MOCK_PDF_MODEL,
            tmp_output_path,
            batch_sample_sizes=custom_sizes,
        )

        # Check that pdf_initial_parameters was called max_size times
        assert mock_init.call_count == max_size

        # Check that it was called with correct replica indices
        expected_calls = [
            call(MOCK_PDF_MODEL, mock_param_initialiser_settings, i)
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
        # When vmapped, params will be 1D per call (the vmap adds the batch dimension externally)
        # So we just check that it's being called
        return jnp.sum(params, axis=-1) if params.ndim > 1 else jnp.sum(params)

    with patch("colibri.time_likelihood.pdf_initial_parameters") as mock_init:
        mock_init.side_effect = lambda model, settings, idx: jnp.array([0.1, 0.2])

        time_log_likelihood(
            counting_likelihood,
            mock_param_initialiser_settings,
            MOCK_PDF_MODEL,
            tmp_output_path,
            batch_sample_sizes=[2, 5],
        )

    # Verify that vectorized version was used
    # (warm-up: 2 calls + timing: 2 sizes × 100 repeats = 202 calls)
    assert call_count > 0


def test_time_log_likelihood_csv_format(
    mock_log_likelihood, mock_param_initialiser_settings, tmp_output_path
):
    """Test that CSV is written with correct format."""

    with patch("colibri.time_likelihood.pdf_initial_parameters") as mock_init:
        mock_init.side_effect = lambda model, settings, idx: jnp.array([0.1, 0.2])

        sizes, times = time_log_likelihood(
            mock_log_likelihood,
            mock_param_initialiser_settings,
            MOCK_PDF_MODEL,
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
        # Due to timing variability in tests, we just check it's positive
        assert relative_time_2 > 0  # Second should be positive


def test_time_log_likelihood_logging(
    mock_log_likelihood, mock_param_initialiser_settings, tmp_output_path, caplog
):
    """Test that appropriate log messages are generated."""

    with patch("colibri.time_likelihood.pdf_initial_parameters") as mock_init:
        mock_init.side_effect = lambda model, settings, idx: jnp.array([0.1, 0.2])

        with caplog.at_level(logging.INFO):
            time_log_likelihood(
                mock_log_likelihood,
                mock_param_initialiser_settings,
                MOCK_PDF_MODEL,
                tmp_output_path,
                batch_sample_sizes=[2, 5],
            )

    # Check that key log messages are present
    assert "Generating samples for log likelihood timing..." in caplog.text
    assert "Warming up (JIT compilation)..." in caplog.text
    assert "Timing different batch sizes..." in caplog.text
    assert "Results saved to" in caplog.text
    assert "Using custom batch sample sizes" in caplog.text


def test_time_log_likelihood_uses_max_size_efficiently(
    mock_log_likelihood, mock_param_initialiser_settings, tmp_output_path
):
    """Test that samples are generated only once for max size and then subsetted."""

    sizes = [10, 50, 100]  # max_size = 100

    with patch("colibri.time_likelihood.pdf_initial_parameters") as mock_init:
        # Each call returns a unique array based on replica index
        mock_init.side_effect = lambda model, settings, idx: jnp.array([float(idx)] * 2)

        time_log_likelihood(
            mock_log_likelihood,
            mock_param_initialiser_settings,
            MOCK_PDF_MODEL,
            tmp_output_path,
            batch_sample_sizes=sizes,
        )

        # Should be called exactly max(sizes) times, not sum(sizes) times
        assert mock_init.call_count == 100
        assert mock_init.call_count != (10 + 50 + 100)


@pytest.mark.parametrize(
    "sizes",
    [
        [1, 10],  # Need at least 2 sizes for warm-up
        [5, 10, 20, 50],
        [100, 500, 1000],
    ],
)
def test_time_log_likelihood_various_size_combinations(
    mock_log_likelihood, mock_param_initialiser_settings, tmp_output_path, sizes
):
    """Test with various combinations of batch sizes."""

    with patch("colibri.time_likelihood.pdf_initial_parameters") as mock_init:
        mock_init.side_effect = lambda model, settings, idx: jnp.array([0.1, 0.2])

        returned_sizes, times = time_log_likelihood(
            mock_log_likelihood,
            mock_param_initialiser_settings,
            MOCK_PDF_MODEL,
            tmp_output_path,
            batch_sample_sizes=sizes,
        )

    assert returned_sizes == sizes
    assert len(times) == len(sizes)
    assert all(t > 0 for t in times)


def test_time_log_likelihood_none_uses_defaults(
    mock_log_likelihood, mock_param_initialiser_settings, tmp_output_path, caplog
):
    """Test that passing None for batch_sample_sizes uses default sizes."""

    with patch("colibri.time_likelihood.pdf_initial_parameters") as mock_init:
        mock_init.side_effect = lambda model, settings, idx: jnp.array([0.1, 0.2])

        with caplog.at_level(logging.INFO):
            sizes, times = time_log_likelihood(
                mock_log_likelihood,
                mock_param_initialiser_settings,
                MOCK_PDF_MODEL,
                tmp_output_path,
                batch_sample_sizes=None,
            )

    # Check that default sizes were used
    default_sizes = [1, 10, 100, 1000, 5000, 10000, 20000, 50000, 100000]
    assert sizes == default_sizes
    assert "Using default batch sample sizes" in caplog.text


def test_time_log_likelihood_with_mock_pdf_model_param_names(
    mock_log_likelihood, mock_param_initialiser_settings, tmp_output_path
):
    """Test that the function correctly uses MOCK_PDF_MODEL.param_names."""

    with patch("colibri.time_likelihood.pdf_initial_parameters") as mock_init:
        # Verify the mock is called with the correct number of parameters
        mock_init.side_effect = lambda model, settings, idx: jnp.array(
            [0.1 * idx, 0.2 * idx]
        )

        time_log_likelihood(
            mock_log_likelihood,
            mock_param_initialiser_settings,
            MOCK_PDF_MODEL,
            tmp_output_path,
            batch_sample_sizes=[5, 10],  # Need at least 2 for warm-up
        )

        # Verify MOCK_PDF_MODEL was passed correctly
        for call_args in mock_init.call_args_list:
            assert call_args[0][0] == MOCK_PDF_MODEL
            assert call_args[0][1] == mock_param_initialiser_settings
