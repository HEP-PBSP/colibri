from unittest.mock import Mock, patch
from colibri.export_results import write_replicas, export_bayes_results
import jax
import os

# Mock the objects and functions used in tests
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
output_path = "./"
pdf_model = Mock()
rank = 0
size = 1
results_name = "results"


@patch("colibri.export_results.pd.DataFrame.to_csv")
def test_export_bayes_results(mock_to_csv):
    export_bayes_results(bayes_fit, output_path, results_name)

    # Check if the to_csv method was called with the correct arguments
    assert mock_to_csv.call_count == 2
    mock_to_csv.assert_any_call(
        str(output_path) + "/full_posterior_sample.csv", float_format="%.5e"
    )
    mock_to_csv.assert_any_call(
        str(output_path) + f"/{results_name}.csv", float_format="%.5e"
    )

    # Check if the bayes_metrics.csv file was created and contains the correct data
    with open(str(output_path) + "/bayes_metrics.csv", "r") as f:
        content = f.read()
    assert (
        content
        == f"logz,min_chi2,avg_chi2,Cb\n{bayes_fit.logz},{bayes_fit.min_chi2},{bayes_fit.avg_chi2},{bayes_fit.bayes_complexity}\n"
    )

    # clean bayes_metrics.csv file
    os.remove(str(output_path) + "/bayes_metrics.csv")


@patch("colibri.export_results.os.path.exists", return_value=False)
@patch("colibri.export_results.os.mkdir")
@patch("colibri.export_results.write_exportgrid")
@patch("colibri.export_results.log.info")
def test_write_replicas(
    mock_log_info,
    mock_write_exportgrid,
    mock_os_mkdir,
    mock_os_path_exists,
):

    write_replicas(bayes_fit, output_path, pdf_model)

    # Check if the replicas directory was created
    mock_os_path_exists.assert_called_once_with(output_path + "/replicas")
    mock_os_mkdir.assert_called_once_with(output_path + "/replicas")

    # Check if the write_exportgrid function was called for each sample
    assert mock_write_exportgrid.call_count == bayes_fit.resampled_posterior.shape[0]

    # Check if the log info was called for each sample
    assert mock_log_info.call_count == bayes_fit.resampled_posterior.shape[0]
