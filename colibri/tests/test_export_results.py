from unittest.mock import Mock, patch
from colibri.export_results import write_replicas, export_bayes_results
import jax
import os
import pandas as pd
import numpy as np

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


@patch("colibri.export_results.os.path.exists", return_value=False)
@patch("colibri.export_results.os.mkdir")
@patch("colibri.export_results.write_exportgrid")
@patch("colibri.export_results.log.info")
def test_write_replicas(
    mock_log_info, mock_write_exportgrid, mock_os_mkdir, mock_os_path_exists, tmp_path
):
    output_path = str(tmp_path)
    write_replicas(bayes_fit, output_path, pdf_model)

    # Check if the replicas directory was created
    mock_os_path_exists.assert_called_once_with(output_path + "/replicas")
    mock_os_mkdir.assert_called_once_with(output_path + "/replicas")

    # Check if the write_exportgrid function was called for each sample
    assert mock_write_exportgrid.call_count == bayes_fit.resampled_posterior.shape[0]

    # Check if the log info was called for each sample
    assert mock_log_info.call_count == bayes_fit.resampled_posterior.shape[0]
