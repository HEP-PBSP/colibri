from unittest.mock import MagicMock, patch
from reportengine.configparser import ConfigError
from colibri.config import colibriConfig, Environment
import unittest.mock as mock


def test_float32_precision_enabled():
    with mock.patch("colibri.config.jax") as mock_jax:
        env = Environment(float32=True)
        assert env.float32
        mock_jax.config.update.assert_called_once_with("jax_enable_x64", False)


def test_float64_precision_enabled():
    with mock.patch("colibri.config.jax") as mock_jax:
        env = Environment(float32=False)
        assert not env.float32
        mock_jax.config.update.assert_called_once_with("jax_enable_x64", True)


def test_ns_dump_description():
    description = Environment.ns_dump_description()
    assert "replica_index" in description
    assert "trval_index" in description


@patch("colibri.config.log.warning")
def test_parse_analytic_settings(mock_warning):
    # Create input_params required for colibriConfig initialization
    input_params = {}
    # Create an instance of the class
    config = colibriConfig(input_params)

    # Define the settings input
    settings = {
        "sampling_seed": 42,
        "n_posterior_samples": 500,
        "full_sample_size": 2000,
        "optimal_prior": True,
        "unknown_key": "should_warn",
    }

    # Call the method
    result = config.parse_analytic_settings(settings)

    # Assert the result is as expected
    expected = {
        "sampling_seed": 42,
        "n_posterior_samples": 500,
        "full_sample_size": 2000,
        "optimal_prior": True,
    }
    assert result == expected

    # Check that the warning was called for the unknown key
    mock_warning.assert_called_once()
    args, _ = mock_warning.call_args
    assert isinstance(args[0], ConfigError)


def test_parse_analytic_settings_defaults():
    # Create input_params required for colibriConfig initialization
    input_params = {}
    # Create an instance of the class
    config = colibriConfig(input_params)

    # Define the settings input with no values to test defaults
    settings = {}

    # Call the method
    result = config.parse_analytic_settings(settings)

    # Assert the result is as expected with default values
    expected = {
        "sampling_seed": 123456,
        "n_posterior_samples": 100,
        "full_sample_size": 1000,
        "optimal_prior": False,
    }
    assert result == expected
