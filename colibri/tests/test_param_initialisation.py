"""
colibri.tests.test_param_initialisation

Tests for the Monte Carlo initialisation functions in the colibri package.
"""

import logging
import unittest
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np

import pytest
import logging
from colibri.param_initialisation import pdf_initial_parameters

logging.basicConfig(level=logging.DEBUG)

# Mock PDF model setup
pdf_model = MagicMock()
pdf_model.full_param_names = ["param1", "param2", "param3"]


def test_zeros_initializer():
    settings = {"type": "zeros"}
    replica_index = 0
    result = pdf_initial_parameters(pdf_model, settings, replica_index)
    expected_result = jnp.array([0.0] * len(pdf_model.full_param_names))
    np.testing.assert_array_equal(result, expected_result)


@patch("jax.random.PRNGKey")
@patch("jax.random.normal")
def test_normal_initializer(mock_normal, mock_PRNGKey, caplog):
    # ---- Test no means or stds provided (full defaults) ----
    settings = {"type": "normal", "random_seed": 42}
    replica_index = 0

    mock_normal.return_value = jnp.array([0.1, -0.1, 0.2])

    with caplog.at_level("WARNING"):
        result = pdf_initial_parameters(pdf_model, settings, replica_index)

    mock_PRNGKey.assert_called_once_with(42)
    mock_normal.assert_called_once_with(
        key=jax.random.PRNGKey(42), shape=(len(pdf_model.full_param_names),)
    )

    assert "param_initialiser_settings: No 'means' or 'stds' provided." in caplog.text
    np.testing.assert_array_equal(result, jnp.array([0.1, -0.1, 0.2]))

    # ---- Test case where random_seed is not provided ----
    settings = {"type": "normal"}
    replica_index = 1
    mock_normal.return_value = jnp.array([0.5, -0.5, 0.0])

    result = pdf_initial_parameters(pdf_model, settings, replica_index)

    mock_PRNGKey.assert_called_with(1)

    # ---- Test specified mean and standard deviation (both as dicts) ----
    means = {
        "param1": 1,
        "param2": 2,
        "param3": 3,
    }

    stds = {
        "param1": 0.5,
        "param2": 1.0,
        "param3": 2.0,
    }

    settings_both = {"type": "normal", "means": means, "stds": stds, "random_seed": 99}

    mock_normal.reset_mock()
    mock_PRNGKey.reset_mock()

    mock_normal.return_value = jnp.array([0.2, -0.5, 1.0])
    mock_PRNGKey.return_value = "mocked_key_99"

    with caplog.at_level("WARNING"):
        caplog.clear()
        result = pdf_initial_parameters(pdf_model, settings_both, replica_index=0)

    # No warning should be issued when both are provided
    assert len(caplog.records) == 0

    expected = jnp.array(
        [
            1.0 + 0.5 * 0.2,  # 1.1
            2.0 + 1.0 * -0.5,  # 1.5
            3.0 + 2.0 * 1.0,  # 5.0
        ]
    )

    np.testing.assert_allclose(result, expected)

    # ---- Test means dict provided without stds ----
    settings_means_dict_only = {
        "type": "normal",
        "means": {"param1": 1.0, "param2": 2.0, "param3": 3.0},
        "random_seed": 42,
    }

    with caplog.at_level("WARNING"):
        caplog.clear()
        pdf_initial_parameters(pdf_model, settings_means_dict_only, replica_index=0)

    assert "'means' provided without 'stds'" in caplog.text
    assert "Using default std=1.0 for all parameters." in caplog.text

    # ---- Test means scalar provided without stds ----
    settings_means_scalar_only = {
        "type": "normal",
        "means": 2.5,  # scalar mean
        "random_seed": 42,
    }

    with caplog.at_level("WARNING"):
        caplog.clear()
        pdf_initial_parameters(pdf_model, settings_means_scalar_only, replica_index=0)

    assert "'means' provided without 'stds'" in caplog.text
    assert "Using default std=1.0 for all parameters." in caplog.text

    # ---- Test stds dict provided without means ----
    settings_stds_dict_only = {
        "type": "normal",
        "stds": {"param1": 0.1, "param2": 0.2, "param3": 0.3},
        "random_seed": 42,
    }

    with caplog.at_level("WARNING"):
        caplog.clear()
        pdf_initial_parameters(pdf_model, settings_stds_dict_only, replica_index=0)

    assert "'stds' provided without 'means'" in caplog.text
    assert "Using default mean=0.0 for all parameters." in caplog.text

    # ---- Test stds scalar provided without means ----
    settings_stds_scalar_only = {
        "type": "normal",
        "stds": 0.5,  # scalar std
        "random_seed": 42,
    }

    with caplog.at_level("WARNING"):
        caplog.clear()
        pdf_initial_parameters(pdf_model, settings_stds_scalar_only, replica_index=0)

    assert "'stds' provided without 'means'" in caplog.text
    assert "Using default mean=0.0 for all parameters." in caplog.text

    # ---- Test scalar means and stds together ----
    settings_scalars = {
        "type": "normal",
        "means": 1.5,  # scalar mean
        "stds": 0.8,  # scalar std
        "random_seed": 123,
    }

    mock_normal.reset_mock()
    mock_PRNGKey.reset_mock()

    mock_normal.return_value = jnp.array([0.5, -1.0, 2.0])
    mock_PRNGKey.return_value = "mocked_key_123"

    with caplog.at_level("WARNING"):
        caplog.clear()
        result = pdf_initial_parameters(pdf_model, settings_scalars, replica_index=0)

    # No warning should be issued when both are provided
    assert len(caplog.records) == 0

    expected = jnp.array(
        [
            1.5 + 0.8 * 0.5,  # mean + std * sample
            1.5 + 0.8 * -1.0,
            1.5 + 0.8 * 2.0,
        ]
    )

    np.testing.assert_allclose(result, expected)

    # ---- Test mixed dict/scalar (means dict, stds scalar) ----
    settings_mixed = {
        "type": "normal",
        "means": {"param1": 1.0, "param2": 2.0, "param3": 3.0},
        "stds": 0.5,  # scalar std applied to all
        "random_seed": 456,
    }

    mock_normal.reset_mock()
    mock_normal.return_value = jnp.array([1.0, 0.0, -1.0])

    with caplog.at_level("WARNING"):
        caplog.clear()
        result = pdf_initial_parameters(pdf_model, settings_mixed, replica_index=0)

    # No warning should be issued
    assert len(caplog.records) == 0

    expected = jnp.array(
        [
            1.0 + 0.5 * 1.0,  # 1.5
            2.0 + 0.5 * 0.0,  # 2.0
            3.0 + 0.5 * -1.0,  # 2.5
        ]
    )

    np.testing.assert_allclose(result, expected)

    # ---- Test validation errors ----
    # Too few means in dict
    settings_few_means = {
        "type": "normal",
        "means": {"param1": 0.0, "param2": 0.0},  # Only 2 means
        "stds": {"param1": 1.0, "param2": 1.0, "param3": 1.0},
        "random_seed": 42,
    }

    with pytest.raises(
        ValueError, match="'means' dict must have one entry per parameter"
    ):
        pdf_initial_parameters(pdf_model, settings_few_means, replica_index=0)

    # Too few stds in dict
    settings_few_stds = {
        "type": "normal",
        "means": {"param1": 0.0, "param2": 0.0, "param3": 0.0},
        "stds": {"param1": 1.0},  # Only 1 std
        "random_seed": 42,
    }

    with pytest.raises(
        ValueError, match="'stds' dict must have one entry per parameter"
    ):
        pdf_initial_parameters(pdf_model, settings_few_stds, replica_index=0)

    # Invalid type for means
    settings_invalid_means = {
        "type": "normal",
        "means": ["invalid", "list"],  # Invalid type
        "random_seed": 42,
    }

    with pytest.raises(TypeError, match="'means' must be dict or scalar"):
        pdf_initial_parameters(pdf_model, settings_invalid_means, replica_index=0)

    # Invalid type for stds
    settings_invalid_stds = {
        "type": "normal",
        "stds": ["invalid", "list"],  # Invalid type
        "random_seed": 42,
    }

    with pytest.raises(TypeError, match="'stds' must be dict or scalar"):
        pdf_initial_parameters(pdf_model, settings_invalid_stds, replica_index=0)


@patch("jax.random.PRNGKey")
@patch("jax.random.uniform")
def test_uniform_initializer(mock_uniform, mock_PRNGKey):
    settings = {"type": "uniform", "random_seed": 42, "min_val": -1.0, "max_val": 1.0}
    replica_index = 1
    mock_uniform.return_value = jnp.array([0.5, -0.5, 0.0])

    result = pdf_initial_parameters(pdf_model, settings, replica_index)

    mock_PRNGKey.assert_called_once_with(43)
    mock_uniform.assert_called_once_with(
        key=jax.random.PRNGKey(43),
        shape=(len(pdf_model.full_param_names),),
        minval=-1.0,
        maxval=1.0,
    )
    np.testing.assert_array_equal(result, jnp.array([0.5, -0.5, 0.0]))

    # Reset mock between calls

    mock_PRNGKey.reset_mock()
    mock_uniform.reset_mock()

    # ---- Test per-parameter bounds case ----

    bounds = {
        "param1": (-1.0, 1.0),
        "param2": (0.0, 2.0),
        "param3": (-0.5, 0.5),
    }

    settings_bounds = {"type": "uniform", "random_seed": 42, "bounds": bounds}

    # Mock return value to match param count
    mock_uniform.return_value = jnp.array([0.1, 1.5, 0.0])

    result_bounds = pdf_initial_parameters(pdf_model, settings_bounds, replica_index)

    np.testing.assert_array_equal(result_bounds, jnp.array([0.1, 1.5, 0.0]))

    # Get the actual call arguments
    _, called_kwargs = mock_uniform.call_args

    # Check the 'key' argument matches
    assert called_kwargs["key"] == jax.random.PRNGKey(43)

    # Check the 'shape' argument matches
    assert called_kwargs["shape"] == (len(pdf_model.full_param_names),)

    # Use numpy/jax testing utilities for arrays
    np.testing.assert_array_equal(called_kwargs["minval"], jnp.array([-1.0, 0.0, -0.5]))
    np.testing.assert_array_equal(called_kwargs["maxval"], jnp.array([1.0, 2.0, 0.5]))

    # ---- Test missing parameter in bounds ----
    incomplete_bounds = {
        "param0": (-1.0, 1.0),
        # "param1" is missing on purpose
        "param3": (-0.5, 0.5),
    }

    settings_missing_bounds = {
        "type": "uniform",
        "random_seed": 42,
        "bounds": incomplete_bounds,
    }

    with pytest.raises(ValueError, match="Missing bounds for parameters"):
        pdf_initial_parameters(pdf_model, settings_missing_bounds, 1)

    # ---- Test missing min_val/max_val and bounds ----
    settings_invalid = {
        "type": "uniform",
        "random_seed": 42,
        # neither "bounds" nor min/max
    }

    with pytest.raises(
        ValueError, match="param_initialiser_settings must define either"
    ):
        pdf_initial_parameters(pdf_model, settings_invalid, 1)


def test_invalid_initializer_type():
    settings = {"type": "invalid_type"}
    replica_index = 0
    with unittest.TestCase().assertLogs(level="WARNING") as log:
        result = pdf_initial_parameters(pdf_model, settings, replica_index)
        # Asserting that at least one warning was logged
        assert log.output
    expected_result = jnp.array([0.0] * len(pdf_model.full_param_names))
    np.testing.assert_array_equal(result, expected_result)
