"""
colibri.tests.test_config.py

Test module for config.py
"""

import unittest
import unittest.mock as mock
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import reportengine
from reportengine.configparser import ConfigError

from colibri.config import Environment, colibriConfig
from colibri.tests.conftest import TEST_PRIOR_SETTINGS_UNIFORM

BASE_CONFIG = colibriConfig({})


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


@patch("hashlib.md5")
@patch("builtins.open", new_callable=mock_open)
@patch("colibri.config.shutil.copy2")
def test_init_output(mock_copy2, mock_open, mock_md5, tmp_path):
    env = Environment()
    env.output_path = tmp_path
    env.config_yml = Path(tmp_path / "config.yml")

    # Mock the md5 hash calculation
    mock_md5.return_value.hexdigest.return_value = "fake_md5_hash"

    env.init_output()

    # Check that the output directory was created
    assert env.output_path.exists()
    assert env.output_path.is_dir()

    # Check that input_folder was created
    assert (env.output_path / "input").exists()

    # Check that the config files were copied
    mock_copy2.assert_any_call(env.config_yml, env.output_path / "filter.yml")
    mock_copy2.assert_any_call(env.config_yml, env.output_path / "input/runcard.yaml")

    # Check if md5 hash is generated and stored
    mock_open().write.assert_called_once_with("fake_md5_hash")


@patch("colibri.config.log.warning")
def test_parse_prior_settings(mock_warning):
    # Define the settings input
    settings1 = {
        "prior_distribution": "uniform_parameter_prior",
        "unknown_key": "should_warn",
    }

    # Call the method
    result1 = BASE_CONFIG.parse_prior_settings(settings1)

    # Assert the result is as expected
    expected1 = TEST_PRIOR_SETTINGS_UNIFORM
    assert result1 == expected1

    # Check that the warning was called for the unknown key
    assert len(mock_warning.mock_calls) == 2

    # check that error is raised
    settings2 = {
        "prior_distribution": "prior_from_gauss_posterior",
    }
    with unittest.TestCase.assertRaises(unittest.TestCase(), ConfigError):
        BASE_CONFIG.parse_prior_settings(settings2)


@patch("colibri.config.log.warning")
def test_parse_analytic_settings(mock_warning):
    # Define the settings input
    settings = {
        "sampling_seed": 42,
        "n_posterior_samples": 500,
        "full_sample_size": 2000,
        "unknown_key": "should_warn",
    }

    # Call the method
    result = BASE_CONFIG.parse_analytic_settings(settings)

    # Assert the result is as expected
    expected = {
        "sampling_seed": 42,
        "n_posterior_samples": 500,
        "full_sample_size": 2000,
    }
    assert result == expected

    # Check that the warning was called for the unknown key
    mock_warning.assert_called_once()
    args, _ = mock_warning.call_args
    assert isinstance(args[0], ConfigError)


def test_parse_analytic_settings_defaults():
    # Define the settings input with no values to test defaults
    settings = {}

    # Call the method
    result = BASE_CONFIG.parse_analytic_settings(settings)

    # Assert the result is as expected with default values
    expected = {
        "sampling_seed": 123456,
        "n_posterior_samples": 100,
        "full_sample_size": 1000,
    }
    assert result == expected


@patch("colibri.config.os.path.exists")
@patch("colibri.config.log.warning")
@patch("colibri.config.log.info")
def test_parse_ns_settings(mock_info, mock_warning, mock_exists, tmp_path):
    # Test known key settings
    settings = {
        "n_posterior_samples": 500,
        "posterior_resampling_seed": 78910,
        "ReactiveNS_settings": {
            "log_dir": str(tmp_path / "mock_log_dir"),
            "resume": True,
            "vectorized": True,
        },
        "ultranest_seed": 654321,
        "sampler_plot": False,
        "popstepsampler": True,
    }

    # Mock the existence of the log directory
    mock_exists.return_value = True

    # Call the function
    ns_settings = BASE_CONFIG.parse_ns_settings(settings, tmp_path)

    # Check that the settings were parsed correctly
    expected_settings = {
        "n_posterior_samples": 500,
        "posterior_resampling_seed": 78910,
        "ReactiveNS_settings": {
            "log_dir": str(tmp_path / "mock_log_dir"),
            "resume": True,
            "vectorized": True,
        },
        "Run_settings": {},
        "SliceSampler_settings": {},
        "ultranest_seed": 654321,
        "sampler_plot": False,
        "popstepsampler": True,
    }

    assert ns_settings == expected_settings
    assert mock_info.called


@patch("colibri.config.os.path.exists")
@patch("colibri.config.log.warning")
def test_parse_ns_settings_with_unknown_keys(mock_warning, mock_exists, tmp_path):
    # Test with unknown keys in settings
    settings = {
        "unknown_key": "value",
        "n_posterior_samples": 500,
        "posterior_resampling_seed": 78910,
    }

    # Mock the existence of the log directory
    mock_exists.return_value = False

    # Call the function
    ns_settings = BASE_CONFIG.parse_ns_settings(settings, tmp_path)

    # Check that the settings were parsed correctly
    expected_settings = {
        "n_posterior_samples": 500,
        "posterior_resampling_seed": 78910,
        "ReactiveNS_settings": {
            "log_dir": str(tmp_path / "ultranest_logs"),
            "resume": "overwrite",
            "vectorized": False,
        },
        "Run_settings": {},
        "SliceSampler_settings": {},
        "ultranest_seed": 123456,
        "sampler_plot": True,
        "popstepsampler": False,
    }

    assert ns_settings == expected_settings
    assert mock_warning.called


@patch("colibri.config.os.path.exists")
@patch("colibri.config.log.info")
def test_parse_ns_settings_with_missing_log_dir(mock_info, mock_exists, tmp_path):
    # Test missing log directory
    settings = {
        "ReactiveNS_settings": {
            "log_dir": str(tmp_path / "mock_log_dir"),
            "resume": True,
        },
    }

    # Mock the existence of the log directory to False
    mock_exists.return_value = False

    with unittest.TestCase.assertRaises(unittest.TestCase(), FileNotFoundError):
        BASE_CONFIG.parse_ns_settings(settings, tmp_path)


def test_parse_ns_settings_with_defaults(tmp_path):
    # Test default settings
    settings = {}

    # Call the function
    ns_settings = BASE_CONFIG.parse_ns_settings(settings, tmp_path)

    # Check that the settings were parsed correctly
    expected_settings = {
        "n_posterior_samples": 1000,
        "posterior_resampling_seed": 123456,
        "ReactiveNS_settings": {
            "log_dir": str(tmp_path / "ultranest_logs"),
            "resume": "overwrite",
            "vectorized": False,
        },
        "Run_settings": {},
        "SliceSampler_settings": {},
        "ultranest_seed": 123456,
        "sampler_plot": True,
        "popstepsampler": False,
    }

    assert ns_settings == expected_settings


def test_parse_positivity_penalty_settings_defaults():
    """
    Test that the correct defaults are returned by positivity penalty
    parser.
    """
    # Test default settings
    settings = {}

    # Call the function
    pos_settings = BASE_CONFIG.parse_positivity_penalty_settings(settings)

    # Check that the settings were parsed correctly
    expected_settings = {
        "positivity_penalty": False,
        "alpha": 1e-7,
        "lambda_positivity": 3000,
    }

    assert pos_settings == expected_settings


@patch("colibri.config.log.warning")
def test_parse_positivity_penalty_settings(mock_warning):
    """
    Test that the inputs are parsed as expected by positivity penalty
    settings parser.
    """
    # Test default settings
    settings = {
        "unknown_key": 1,
        "positivity_penalty": True,
        "alpha": 1e-7,
        "lambda_positivity": 10000,
    }

    # Call the function
    pos_settings = BASE_CONFIG.parse_positivity_penalty_settings(settings)

    # Check that the settings were parsed correctly
    expected_settings = {
        "positivity_penalty": True,
        "alpha": 1e-7,
        "lambda_positivity": 10000,
    }

    assert pos_settings == expected_settings
    assert mock_warning.called


def test_parse_integrability_settings_valid():
    """Test with valid settings."""
    settings = {
        "integrability": True,
        "integrability_specs": {
            "lambda_integrability": 50,
            "evolution_flavours": ["V", "T8"],
        },
    }

    result = BASE_CONFIG.parse_integrability_settings(settings)

    assert result.integrability == True
    assert result.integrability_specs["lambda_integrability"] == 50
    assert result.integrability_specs["evolution_flavours"] == [3, 10]  # Translated IDs


def test_parse_integrability_settings_default_values():
    """Test with missing optional settings to check defaults."""
    settings = {
        "integrability": True,
    }

    result = BASE_CONFIG.parse_integrability_settings(settings)

    assert result.integrability == True
    assert result.integrability_specs["lambda_integrability"] == 100
    assert result.integrability_specs["evolution_flavours"] == [9, 10]  # Defaults


def test_parse_integrability_settings_unknown_key():
    """Test with an unknown key in the settings."""
    settings = {
        "integrability": True,
        "unknown_key": "some_value",
    }

    with pytest.raises(
        ConfigError, match="Key 'unknown_key' in integrability_settings not known."
    ):
        BASE_CONFIG.parse_integrability_settings(settings)


def test_parse_integrability_settings_invalid_evolution_flavours():
    """Test with an invalid evolution_flavours value."""
    settings = {
        "integrability": True,
        "integrability_specs": {
            "lambda_integrability": 50,
            "evolution_flavours": ["Invalid"],
        },
    }

    with pytest.raises(
        ConfigError, match="evolution_flavours ids can only be taken from"
    ):
        BASE_CONFIG.parse_integrability_settings(settings)


def test_parse_integrability_settings_empty():
    """Test with empty settings."""
    settings = {}

    result = BASE_CONFIG.parse_integrability_settings(settings)

    assert result.integrability == False
    assert result.integrability_specs == {
        "lambda_integrability": 100,
        "evolution_flavours": [9, 10],
        "integrability_xgrid": [2.00000000e-07],
    }  # Default


def test_parse_closure_test_pdf_colibri_model():
    # test for colibri_model input
    closure_test_pdf = "colibri_model"
    assert BASE_CONFIG.parse_closure_test_pdf(closure_test_pdf) == "colibri_model"


def test_produce_commondata_tuple():
    """
    Tests that the production rule (explicit node) produce_commondata_tuple
    behaves as expected by always returning a reportengine explicit node.

    NOTE: specific tests for the functions returned by this explicit node
    can be found in the test_commondata_utils.py file.
    """
    # test default behaviour
    closure_test_level = False
    explicit_node_exp = BASE_CONFIG.produce_commondata_tuple(closure_test_level)
    assert type(explicit_node_exp) == reportengine.configparser.ExplicitNode

    # test closure test levels
    closure_test_level = 0
    explicit_node_0 = BASE_CONFIG.produce_commondata_tuple(closure_test_level)
    assert type(explicit_node_0) == reportengine.configparser.ExplicitNode

    closure_test_level = 1
    explicit_node_1 = BASE_CONFIG.produce_commondata_tuple(closure_test_level)
    assert type(explicit_node_1) == reportengine.configparser.ExplicitNode

    # test that ConfigError is raised when an invalid value for closure_test_level is passed
    closure_test_level = 2
    with pytest.raises(ConfigError):
        BASE_CONFIG.produce_commondata_tuple(closure_test_level)
