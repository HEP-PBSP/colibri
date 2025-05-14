"""
colibri.tests.test_mc_utils

Tests for the Monte Carlo utilities in the colibri package.
"""

from unittest.mock import Mock, mock_open, patch

import jax.numpy as jnp
import pandas as pd
from colibri.tests.conftest import (
    CLOSURE_TEST_PDFSET,
    PSEUDODATA_SEED,
    REPLICA_INDEX,
    TEST_DATASETS,
    TRVAL_INDEX,
    TEST_COMMONDATA_FOLDER,
    MOCK_PDF_MODEL,
)
from numpy.testing import assert_allclose

from colibri.api import API as colibriAPI
from colibri.constants import EXPORT_LABELS, LHAPDF_XGRID
from colibri.mc_utils import write_exportgrid_mc

MC_PSEUDODATA = {
    "level_1_seed": PSEUDODATA_SEED,
    **CLOSURE_TEST_PDFSET,
    **TRVAL_INDEX,
    **REPLICA_INDEX,
    **TEST_DATASETS,
}


def test_mc_pseudodata():
    """
    Regression test, testing that currently generated pseudodata is consistent
    with the reference one.
    """
    reference_pseudodata = pd.read_csv(
        TEST_COMMONDATA_FOLDER / "NMC_level2_central_values.csv"
    )

    current_pseudodata = colibriAPI.mc_pseudodata(**MC_PSEUDODATA)

    assert_allclose(reference_pseudodata["cv"].values, current_pseudodata.pseudodata)


# Define the test parameters
parameters = [0.1, 0.2, 0.3]  # Example parameters
replica_index = REPLICA_INDEX["replica_index"]


@patch("os.path.exists")
@patch("os.mkdir")
@patch("builtins.open", new_callable=mock_open)
def test_write_exportgrid_creates_directories(
    mock_open, mock_mkdir, mock_exists, tmp_path
):
    mock_exists.side_effect = lambda path: False

    write_exportgrid_mc(parameters, MOCK_PDF_MODEL, replica_index, tmp_path)

    expected_dir_path = f"{tmp_path}/fit_replicas/replica_1"
    mock_mkdir.assert_called_once_with(expected_dir_path)


@patch("os.path.exists")
@patch("os.mkdir")
@patch("builtins.open", new_callable=mock_open)
def test_write_exportgrid_writes_file(mock_open, mock_mkdir, mock_exists, tmp_path):
    mock_exists.side_effect = lambda path: False

    with patch("yaml.dump") as mock_yaml_dump:
        write_exportgrid_mc(parameters, MOCK_PDF_MODEL, replica_index, tmp_path)

        fit_name = str(tmp_path).split("/")[-1]

        expected_file_path = f"{tmp_path}/fit_replicas/replica_1/{fit_name}.exportgrid"
        mock_open.assert_called_once_with(expected_file_path, "w")

        written_data = mock_yaml_dump.call_args[0][0]

        # Check the contents of the written data
        assert written_data["q20"] == 1.65**2
        assert written_data["xgrid"] == LHAPDF_XGRID
        assert written_data["replica"] == replica_index
        assert written_data["labels"] == EXPORT_LABELS
        assert isinstance(written_data["pdfgrid"], list)


@patch("os.path.exists")
@patch("os.mkdir")
@patch("builtins.open", new_callable=mock_open)
def test_write_exportgrid_correct_paths_for_monte_carlo(
    mock_open, mock_mkdir, mock_exists, tmp_path
):
    mock_exists.side_effect = lambda path: False

    write_exportgrid_mc(parameters, MOCK_PDF_MODEL, replica_index, tmp_path)

    expected_dir_path = f"{tmp_path}/fit_replicas/replica_1"
    mock_mkdir.assert_called_once_with(expected_dir_path)


@patch("os.path.exists")
@patch("os.mkdir")
@patch("builtins.open", new_callable=mock_open)
def test_write_exportgrid_no_directory_creation_if_exists(
    mock_open, mock_mkdir, mock_exists, tmp_path
):
    mock_exists.side_effect = lambda path: True

    write_exportgrid_mc(parameters, MOCK_PDF_MODEL, replica_index, tmp_path)

    mock_mkdir.assert_not_called()
