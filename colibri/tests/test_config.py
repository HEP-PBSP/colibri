from unittest.mock import patch, mock_open
from reportengine.configparser import ConfigError
from colibri.config import colibriConfig, Environment
import unittest.mock as mock
from pathlib import Path
import unittest


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


@patch("colibri.config.os.path.exists")
@patch("colibri.config.log.warning")
@patch("colibri.config.log.info")
def test_parse_ns_settings(mock_info, mock_warning, mock_exists, tmp_path):
    # Create input_params required for colibriConfig initialization
    input_params = {}
    # Create an instance of the class
    config = colibriConfig(input_params)
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
    ns_settings = config.parse_ns_settings(settings, tmp_path)

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
    # Create input_params required for colibriConfig initialization
    input_params = {}
    # Create an instance of the class
    config = colibriConfig(input_params)
    # Test with unknown keys in settings
    settings = {
        "unknown_key": "value",
        "n_posterior_samples": 500,
        "posterior_resampling_seed": 78910,
    }

    # Mock the existence of the log directory
    mock_exists.return_value = False

    # Call the function
    ns_settings = config.parse_ns_settings(settings, tmp_path)

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
    # Create input_params required for colibriConfig initialization
    input_params = {}
    # Create an instance of the class
    config = colibriConfig(input_params)
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
        config.parse_ns_settings(settings, tmp_path)


def test_parse_ns_settings_with_defaults(tmp_path):
    # Create input_params required for colibriConfig initialization
    input_params = {}
    # Create an instance of the class
    config = colibriConfig(input_params)
    # Test default settings
    settings = {}

    # Call the function
    ns_settings = config.parse_ns_settings(settings, tmp_path)

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
