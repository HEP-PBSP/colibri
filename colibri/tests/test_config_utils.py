"""
colibri.tests.test_config_utils.py

This module contains the tests for the config_utils module of colibri.
"""

import unittest
from unittest.mock import patch

from reportengine.configparser import ConfigError

from colibri.config_utils import ns_settings_parser, analytic_settings_parser
from colibri.core import (
    DEFAULT_N_POSTERIOR_SAMPLES,
    DEFAULT_SAMPLING_SEED,
    DEFAULT_FULL_SAMPLE_SIZE,
    DEFAULT_OPTIMAL_PRIOR,
)


@patch("colibri.config_utils.log.warning")
def test_parse_analytic_settings(mock_warning):

    # Define the settings input
    settings = {
        "sampling_seed": 42,
        "n_posterior_samples": 500,
        "full_sample_size": 2000,
        "optimal_prior": True,
        "unknown_key": "should_warn",
    }

    # Call the method
    result = analytic_settings_parser(settings)

    # Assert the result is as expected
    expected = {
        "sampling_seed": 42,
        "n_posterior_samples": 500,
        "full_sample_size": 2000,
        "optimal_prior": True,
    }

    assert result.to_dict() == expected

    # Check that the warning was called for the unknown key
    mock_warning.assert_called_once()
    args, _ = mock_warning.call_args
    assert isinstance(args[0], ConfigError)


def test_parse_analytic_settings_defaults():

    # Define the settings input with no values to test defaults
    settings = {}

    # Call the method
    result = analytic_settings_parser(settings)

    # Assert the result is as expected with default values
    expected = {
        "sampling_seed": DEFAULT_SAMPLING_SEED,
        "n_posterior_samples": DEFAULT_N_POSTERIOR_SAMPLES,
        "full_sample_size": DEFAULT_FULL_SAMPLE_SIZE,
        "optimal_prior": DEFAULT_OPTIMAL_PRIOR,
    }
    assert result.to_dict() == expected


@patch("colibri.config_utils.os.path.exists")
@patch("colibri.config_utils.log.warning")
@patch("colibri.config_utils.log.info")
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
    ns_settings = ns_settings_parser(settings, tmp_path)

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

    assert ns_settings.to_dict() == expected_settings
    assert mock_info.called


@patch("colibri.config_utils.os.path.exists")
@patch("colibri.config_utils.log.warning")
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
    ns_settings = ns_settings_parser(settings, tmp_path)

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
        "ultranest_seed": DEFAULT_SAMPLING_SEED,
        "sampler_plot": True,
        "popstepsampler": False,
    }

    assert ns_settings.to_dict() == expected_settings
    assert mock_warning.called


@patch("colibri.config_utils.os.path.exists")
@patch("colibri.config_utils.log.info")
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
        ns_settings_parser(settings, tmp_path)


def test_parse_ns_settings_with_defaults(tmp_path):

    # Test default settings
    settings = {}

    # Call the function
    ns_settings = ns_settings_parser(settings, tmp_path)

    # Check that the settings were parsed correctly
    expected_settings = {
        "n_posterior_samples": DEFAULT_N_POSTERIOR_SAMPLES,
        "posterior_resampling_seed": DEFAULT_SAMPLING_SEED,
        "ReactiveNS_settings": {
            "log_dir": str(tmp_path / "ultranest_logs"),
            "resume": "overwrite",
            "vectorized": False,
        },
        "Run_settings": {},
        "SliceSampler_settings": {},
        "ultranest_seed": DEFAULT_SAMPLING_SEED,
        "sampler_plot": True,
        "popstepsampler": False,
    }

    assert ns_settings.to_dict() == expected_settings
