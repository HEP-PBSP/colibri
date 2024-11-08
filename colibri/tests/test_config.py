from unittest.mock import patch, mock_open
from colibri.config import colibriConfig, Environment
import unittest.mock as mock
from pathlib import Path


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
