"""
colibri.tests.test_export_results.py

This module contains the tests for the export_results module of colibri.
"""

from unittest.mock import Mock, mock_open, patch

import jax
import numpy as np
import pandas as pd
import yaml
import pathlib

from colibri.export_results import (
    export_bayes_results,
    write_exportgrid,
    write_replicas,
    read_exportgrid,
    get_pdfgrid_from_exportgrids,
)

# Mock the objects and functions used in tests
# Mock the BayesianFit object
bayes_fit = Mock()
bayes_fit.resampled_posterior = jax.random.uniform(jax.random.PRNGKey(0), shape=(10, 2))
bayes_fit.full_posterior_samples = jax.random.uniform(
    jax.random.PRNGKey(0), shape=(100, 2)
)
bayes_fit.bayesian_metrics = {"logz": 1}
bayes_fit.param_names = ["param1", "param2"]

# Mock the pdf_model object
pdf_model = Mock()

rank = 0
size = 1
results_name = "results"


def test_export_bayes_results(tmp_path):

    output_path = str(tmp_path)
    export_bayes_results(bayes_fit, output_path, results_name)

    # Load full_posterior_sample.csv with pandas and check that the values are the correct ones
    full_posterior_samples = pd.read_csv(
        str(output_path) + "/full_posterior_sample.csv", sep=",", index_col=0
    )
    assert full_posterior_samples.shape == (100, 2)
    assert full_posterior_samples.columns.tolist() == bayes_fit.param_names
    assert (
        np.round(full_posterior_samples.values).tolist()
        == np.round(bayes_fit.full_posterior_samples).tolist()
    )

    # Load resampled_posterior.csv with pandas and check that the values are the correct ones
    resampled_posterior = pd.read_csv(
        str(output_path) + f"/{results_name}.csv", sep=",", index_col=0
    )
    assert resampled_posterior.shape == (10, 2)
    assert resampled_posterior.columns.tolist() == bayes_fit.param_names
    assert (
        np.round(resampled_posterior.values).tolist()
        == np.round(bayes_fit.resampled_posterior).tolist()
    )

    # Check if the bayes_metrics.csv file was created and contains the correct data
    with open(str(output_path) + "/bayes_metrics.csv", "r") as f:
        content = f.read()

    assert content.strip() == f'logz\n{bayes_fit.bayesian_metrics["logz"]}\n'.strip()


def test_write_exportgrid():
    # Mock data
    grid_for_writing = np.random.rand(14, 2)  # Mock a random 14x2 array
    grid_name = "test_grid"
    replica_index = 1
    Q = 1.65
    xgrid = [0.001, 0.01, 0.1, 1.0]  # Mock xgrid values
    export_labels = ["a", "b", "c"]  # Mock labels

    # Mock open and yaml.dump
    with patch("builtins.open", mock_open()) as mocked_file:
        with patch("yaml.dump") as mocked_yaml_dump:

            write_exportgrid(
                grid_for_writing=grid_for_writing,
                grid_name=grid_name,
                replica_index=replica_index,
                Q=Q,
                xgrid=xgrid,
                export_labels=export_labels,
            )

            # Ensure that yaml.dump was called with a dictionary that contains
            # the expected keys and some key properties.
            called_args, _ = mocked_yaml_dump.call_args
            written_data = called_args[0]

            # Check that the main structure is correct
            assert "q20" in written_data
            assert "xgrid" in written_data
            assert "replica" in written_data
            assert "labels" in written_data
            assert "pdfgrid" in written_data

            # Check specific values
            assert written_data["q20"] == Q**2
            assert written_data["xgrid"] == xgrid
            assert written_data["replica"] == replica_index
            assert written_data["labels"] == export_labels

            # Check that the pdfgrid was transformed as expected (shape, not values)
            assert np.array(written_data["pdfgrid"]).shape == (2, 14)

    # Verify that the file was opened with the correct name
    mocked_file.assert_called_once_with(f"{grid_name}.exportgrid", "w")


@patch("colibri.export_results.write_exportgrid")
@patch("colibri.export_results.log.info")
def test_write_replicas(mock_log_info, mock_write_exportgrid, tmp_path):
    output_path = str(tmp_path)
    write_replicas(bayes_fit, output_path, pdf_model)

    # Check if the write_exportgrid function was called for each sample
    assert mock_write_exportgrid.call_count == bayes_fit.resampled_posterior.shape[0]

    # Check if the log info was called for each sample
    assert mock_log_info.call_count == bayes_fit.resampled_posterior.shape[0]


def test_read_exportgrid():
    """
    Test the read_exportgrid function to ensure it correctly reads and processes
    an exportgrid file.
    """
    # Mock data for the test
    mock_yaml_data = {"pdfgrid": [[1, 2], [3, 4]]}
    mock_yaml_str = yaml.dump(mock_yaml_data)

    # Mock the flavour_to_evolution_matrix
    mock_flavour_to_evolution_matrix = np.array([[1, 0], [0, 1]])

    # Expected output after applying the matrix multiplication
    expected_pdfgrid = (
        mock_flavour_to_evolution_matrix @ np.array(mock_yaml_data["pdfgrid"]).T
    )
    expected_result = {
        "pdfgrid": expected_pdfgrid,
    }

    # Mock the open function and the flavour_to_evolution_matrix
    with patch("builtins.open", mock_open(read_data=mock_yaml_str)) as mock_file, patch(
        "colibri.export_results.flavour_to_evolution_matrix",
        mock_flavour_to_evolution_matrix,
    ):

        # Call the function under test
        exportgrid_path = pathlib.Path("mock_path/exportgrid.yaml")
        result = read_exportgrid(exportgrid_path)

        # Assertions
        mock_file.assert_called_once_with(exportgrid_path, "r")
        assert "pdfgrid" in result, "The key 'pdfgrid' is missing in the result."
        np.testing.assert_array_equal(
            result["pdfgrid"],
            expected_result["pdfgrid"],
            "The pdfgrid transformation result is incorrect.",
        )


def test_get_pdfgrid_from_exportgrids():
    """
    Test the get_pdfgrid_from_exportgrids function to ensure it correctly reads and aggregates exportgrids.
    """
    # Mock data for the test
    mock_exportgrid_1 = {
        "pdfgrid": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "labels": ["label1", "label2"],
        "xgrid": [0.1, 0.2],
    }
    mock_exportgrid_2 = {
        "pdfgrid": np.array([[5.0, 6.0], [7.0, 8.0]]),
        "labels": ["label1", "label2"],
        "xgrid": [0.1, 0.2],
    }

    mock_flavour_to_evolution_matrix = np.array([[1, 0], [0, 1]])

    expected_pdfgrid = np.array(
        [
            mock_flavour_to_evolution_matrix @ np.array(mock_exportgrid_1["pdfgrid"]),
            mock_flavour_to_evolution_matrix @ np.array(mock_exportgrid_2["pdfgrid"]),
        ]
    )

    # Mock the read_exportgrid function
    def mock_read_exportgrid(path):
        if "replica_1" in str(path):
            return mock_exportgrid_1
        elif "replica_2" in str(path):
            return mock_exportgrid_2

    # Mock the file system and read_exportgrid function
    with patch(
        "pathlib.Path.glob",
        return_value=[
            pathlib.Path("replica_1/mock.exportgrid"),
            pathlib.Path("replica_2/mock.exportgrid"),
        ],
    ), patch(
        "colibri.export_results.read_exportgrid", side_effect=mock_read_exportgrid
    ), patch(
        "colibri.export_results.flavour_to_evolution_matrix",
        mock_flavour_to_evolution_matrix,
    ):

        # Call the function under test
        fit_path = pathlib.Path("mock_path")
        result = get_pdfgrid_from_exportgrids(fit_path)

        # Assertions
        np.testing.assert_array_equal(
            result, expected_pdfgrid, "The aggregated pdf grid result is incorrect."
        )
