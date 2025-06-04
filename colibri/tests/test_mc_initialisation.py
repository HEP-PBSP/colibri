import unittest
from unittest.mock import MagicMock, patch
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import logging
from colibri.mc_initialisation import mc_initial_parameters

logging.basicConfig(level=logging.DEBUG)

# Mock PDF model setup
pdf_model = MagicMock()
pdf_model.param_names = ["param1", "param2", "param3"]


def test_zeros_initializer():
    settings = {"type": "zeros"}
    replica_index = 0
    result = mc_initial_parameters(pdf_model, settings, replica_index)
    expected_result = jnp.array([0.0] * len(pdf_model.param_names))
    np.testing.assert_array_equal(result, expected_result)


@patch("jax.random.PRNGKey")
@patch("jax.random.normal")
def test_normal_initializer(mock_normal, mock_PRNGKey):
    settings = {"type": "normal", "random_seed": 42}
    replica_index = 0

    mock_normal.return_value = jnp.array([0.1, -0.1, 0.2])

    result = mc_initial_parameters(pdf_model, settings, replica_index)

    mock_PRNGKey.assert_called_once_with(42)
    mock_normal.assert_called_once_with(
        key=jax.random.PRNGKey(42), shape=(len(pdf_model.param_names),)
    )
    np.testing.assert_array_equal(result, jnp.array([0.1, -0.1, 0.2]))

    # Now test the case where the random_seed is not provided
    settings = {"type": "normal"}
    replica_index = 1
    mock_normal.return_value = jnp.array([0.5, -0.5, 0.0])

    result = mc_initial_parameters(pdf_model, settings, replica_index)

    mock_PRNGKey.assert_called_with(1)


@patch("jax.random.PRNGKey")
@patch("jax.random.uniform")
def test_uniform_initializer(mock_uniform, mock_PRNGKey):
    settings = {"type": "uniform", "random_seed": 42, "min_val": -1.0, "max_val": 1.0}
    replica_index = 1
    mock_uniform.return_value = jnp.array([0.5, -0.5, 0.0])

    result = mc_initial_parameters(pdf_model, settings, replica_index)

    mock_PRNGKey.assert_called_once_with(43)
    mock_uniform.assert_called_once_with(
        key=jax.random.PRNGKey(43),
        shape=(len(pdf_model.param_names),),
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

    result_bounds = mc_initial_parameters(pdf_model, settings_bounds, replica_index)

    np.testing.assert_array_equal(result_bounds, jnp.array([0.1, 1.5, 0.0]))

    # Get the actual call arguments
    _, called_kwargs = mock_uniform.call_args

    # Check the 'key' argument matches
    assert called_kwargs["key"] == jax.random.PRNGKey(43)

    # Check the 'shape' argument matches
    assert called_kwargs["shape"] == (len(pdf_model.param_names),)

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
        mc_initial_parameters(pdf_model, settings_missing_bounds, 1)

    # ---- Test missing min_val/max_val and bounds ----
    settings_invalid = {
        "type": "uniform",
        "random_seed": 42,
        # neither "bounds" nor min/max
    }

    with pytest.raises(ValueError, match="mc_initialiser_settings must define either"):
        mc_initial_parameters(pdf_model, settings_invalid, 1)


def test_invalid_initializer_type():
    settings = {"type": "invalid_type"}
    replica_index = 0
    with unittest.TestCase().assertLogs(level="WARNING") as log:
        result = mc_initial_parameters(pdf_model, settings, replica_index)
        # Asserting that at least one warning was logged
        assert log.output
    expected_result = jnp.array([0.0] * len(pdf_model.param_names))
    np.testing.assert_array_equal(result, expected_result)
