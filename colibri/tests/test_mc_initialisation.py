import unittest
from unittest.mock import MagicMock, patch
import jax
import jax.numpy as jnp
import numpy as np
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


def test_invalid_initializer_type():
    settings = {"type": "invalid_type"}
    replica_index = 0
    with unittest.TestCase().assertLogs(level="WARNING") as log:
        result = mc_initial_parameters(pdf_model, settings, replica_index)
        # Asserting that at least one warning was logged
        assert log.output
    expected_result = jnp.array([0.0] * len(pdf_model.param_names))
    np.testing.assert_array_equal(result, expected_result)
