"""
colibri/tests/test_config.py

This module contains the tests for the config module of colibri.
"""

from unittest.mock import patch, mock_open
from colibri.config import colibriConfig, Environment
import unittest.mock as mock
from pathlib import Path
from colibri.core import (
    ColibriPriorSettings,
    NestedSamplingSettings,
    DEFAULT_N_POSTERIOR_SAMPLES,
    DEFAULT_SAMPLING_SEED,
    DEFAULT_FULL_SAMPLE_SIZE,
    DEFAULT_OPTIMAL_PRIOR,
)


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


def test_parse_positivity_penalty_settings_defaults():
    """
    Test that the correct defaults are returned by positivity penalty
    parser.
    """
    # Create input_params required for colibriConfig initialization
    input_params = {}
    # Create an instance of the class
    config = colibriConfig(input_params)
    # Test default settings
    settings = {}

    # Call the function
    pos_settings = config.parse_positivity_penalty_settings(settings)

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
    # Create input_params required for colibriConfig initialization
    input_params = {}
    # Create an instance of the class
    config = colibriConfig(input_params)
    # Test default settings
    settings = {
        "unknown_key": 1,
        "positivity_penalty": True,
        "alpha": 1e-7,
        "lambda_positivity": 10000,
    }

    # Call the function
    pos_settings = config.parse_positivity_penalty_settings(settings)

    # Check that the settings were parsed correctly
    expected_settings = {
        "positivity_penalty": True,
        "alpha": 1e-7,
        "lambda_positivity": 10000,
    }

    assert pos_settings == expected_settings
    assert mock_warning.called


@patch("colibri.config.log.warning")
def test_parse_colibri_specs_log_warning(mock_warning, tmp_path):
    """
    Test that warning is logged when unknown keys are present in the
    colibri_specs.
    """
    # Create input_params required for colibriConfig initialization
    input_params = {}
    # Create an instance of the class
    config = colibriConfig(input_params)
    # Test default settings
    settings = {
        "unknown_key": 1,
    }

    # Call the function
    config.parse_colibri_specs(settings, tmp_path)

    assert mock_warning.called


def test_parse_colibri_specs_defaults(tmp_path):
    """
    Test that the correct defaults are returned by colibri specs parser.
    """
    # Create input_params required for colibriConfig initialization
    input_params = {}
    # Create an instance of the class
    config = colibriConfig(input_params)
    # Test default settings
    settings = {}

    # Call the function
    colibri_specs = config.parse_colibri_specs(settings, tmp_path)

    # check loss function specs
    loss_specs = colibri_specs.loss_function_specs
    assert loss_specs.t0pdfset == None
    assert loss_specs.use_fit_t0 == False

    # check prior settings
    prior_settings = colibri_specs.prior_settings
    assert type(prior_settings) == ColibriPriorSettings
    assert prior_settings.prior_distribution == None
    assert prior_settings.max_val == None
    assert prior_settings.min_val == None
    assert prior_settings.prior_fit == None

    # check nested sampling settings
    ns_settings = colibri_specs.ns_settings
    assert type(ns_settings) == NestedSamplingSettings

    assert ns_settings.ReactiveNS_settings == {
        "log_dir": str(tmp_path / "ultranest_logs"),
        "resume": "overwrite",
        "vectorized": False,
    }
    assert ns_settings.Run_settings == {}
    assert ns_settings.SliceSampler_settings == {}
    assert ns_settings.ultranest_seed == DEFAULT_SAMPLING_SEED
    assert ns_settings.sampler_plot == True
    assert ns_settings.popstepsampler == False
    assert ns_settings.n_posterior_samples == DEFAULT_N_POSTERIOR_SAMPLES

    # test analytic fit settings
    analytic_settings = colibri_specs.analytic_settings
    assert analytic_settings.n_posterior_samples == DEFAULT_N_POSTERIOR_SAMPLES
    assert analytic_settings.sampling_seed == DEFAULT_SAMPLING_SEED
    assert analytic_settings.optimal_prior == DEFAULT_OPTIMAL_PRIOR
    assert analytic_settings.full_sample_size == DEFAULT_FULL_SAMPLE_SIZE


def test_produce_prior_settings(tmp_path):
    # Create input_params required for colibriConfig initialization
    input_params = {}
    # Create an instance of the class
    config = colibriConfig(input_params)
    # Test default settings
    settings = {}

    colibri_specs = config.parse_colibri_specs(settings, tmp_path)
    prior_settings = config.produce_prior_settings(colibri_specs)

    assert type(prior_settings) == ColibriPriorSettings
    assert prior_settings.prior_distribution == None
    assert prior_settings.max_val == None
    assert prior_settings.min_val == None
    assert prior_settings.prior_fit == None


def test_produce_ns_settings(tmp_path):
    # Create input_params required for colibriConfig initialization
    input_params = {}
    # Create an instance of the class
    config = colibriConfig(input_params)
    # Test default settings
    settings = {}

    colibri_specs = config.parse_colibri_specs(settings, tmp_path)
    ns_settings = config.produce_ns_settings(colibri_specs)

    assert type(ns_settings) == NestedSamplingSettings

    assert ns_settings.ReactiveNS_settings == {
        "log_dir": str(tmp_path / "ultranest_logs"),
        "resume": "overwrite",
        "vectorized": False,
    }
    assert ns_settings.Run_settings == {}
    assert ns_settings.SliceSampler_settings == {}
    assert ns_settings.ultranest_seed == DEFAULT_SAMPLING_SEED
    assert ns_settings.sampler_plot == True
    assert ns_settings.popstepsampler == False
    assert ns_settings.n_posterior_samples == DEFAULT_N_POSTERIOR_SAMPLES


def test_produce_analytic_settings(tmp_path):
    # Create input_params required for colibriConfig initialization
    input_params = {}
    # Create an instance of the class
    config = colibriConfig(input_params)
    # Test default settings
    settings = {}

    colibri_specs = config.parse_colibri_specs(settings, tmp_path)
    analytic_settings = config.produce_analytic_settings(colibri_specs)

    assert analytic_settings.n_posterior_samples == DEFAULT_N_POSTERIOR_SAMPLES
    assert analytic_settings.sampling_seed == DEFAULT_SAMPLING_SEED
    assert analytic_settings.optimal_prior == DEFAULT_OPTIMAL_PRIOR
    assert analytic_settings.full_sample_size == DEFAULT_FULL_SAMPLE_SIZE


def test_produce_vectorized(tmp_path):
    # Create input_params required for colibriConfig initialization
    input_params = {}
    # Create an instance of the class
    config = colibriConfig(input_params)
    # Test default settings
    settings = {}

    colibri_specs = config.parse_colibri_specs(settings, tmp_path)
    ns_settings = config.produce_ns_settings(colibri_specs)
    vectorized = config.produce_vectorized(ns_settings)

    assert vectorized == False
