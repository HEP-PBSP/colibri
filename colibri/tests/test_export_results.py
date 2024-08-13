"""
colibri.tests.test_export_results.py

This module contains the tests for the export_results module of colibri.
"""

from unittest.mock import Mock, mock_open, patch

import jax
import numpy as np
import pandas as pd

from colibri.export_results import (
    export_bayes_results,
    write_exportgrid,
    write_replicas,
)

# Mock the objects and functions used in tests
# Mock the BayesianFit object
bayes_fit = Mock()
bayes_fit.resampled_posterior = jax.random.uniform(jax.random.PRNGKey(0), shape=(10, 2))
bayes_fit.full_posterior_samples = jax.random.uniform(
    jax.random.PRNGKey(0), shape=(100, 2)
)
bayes_fit.bayes_complexity = 1.0
bayes_fit.avg_chi2 = 1.0
bayes_fit.min_chi2 = 1.0
bayes_fit.logz = 1.0
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
    assert (
        content
        == f"logz,min_chi2,avg_chi2,Cb\n{bayes_fit.logz},{bayes_fit.min_chi2},{bayes_fit.avg_chi2},{bayes_fit.bayes_complexity}\n"
    )


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
