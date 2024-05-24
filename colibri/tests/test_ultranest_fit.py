from unittest.mock import Mock, patch
import numpy as np
import jax.numpy as jnp
import jax

from colibri.ultranest_fit import (
    UltranestFit,
    ultranest_fit,
    run_ultranest_fit,
)

# Define mock input parameters
bayesian_prior = lambda x: x
FIT_XGRID = jnp.logspace(-7, 0, 50)
_chi2_with_positivity_mock = lambda pred, pdf: 1.0

ns_settings = {
    "ultranest_seed": 42,
    "ReactiveNS_settings": {"vectorized": False},
    "SliceSampler_settings": None,
    "Run_settings": {"frac_remain": 0.5, "min_num_live_points": 5},
    "n_posterior_samples": 10,
    "posterior_resampling_seed": 123,
    "sampler_plot": False,
}


def test_ultranest_fit():
    # Create mock pdf model
    mock_pdf_model = Mock()
    mock_pdf_model.param_names = ["param1", "param2"]
    mock_pdf_model.grid_values_func = lambda xgrid: lambda params: jnp.ones(
        (14, len(xgrid))
    )
    mock_pdf_model.pred_and_pdf_func = lambda xgrid, forward_map: (
        lambda params: (params, jnp.ones((14, len(xgrid))))
    )
    _pred_data = None

    fit_result = ultranest_fit(
        _chi2_with_positivity_mock,
        _pred_data,
        mock_pdf_model,
        bayesian_prior,
        ns_settings,
        FIT_XGRID,
    )

    assert isinstance(fit_result, UltranestFit)
    assert fit_result.resampled_posterior.shape == (
        ns_settings["n_posterior_samples"],
        len(mock_pdf_model.param_names),
    )
    assert fit_result.param_names == ["param1", "param2"]
    assert fit_result.ultranest_specs == ns_settings
    assert isinstance(fit_result.ultranest_result, dict)


@patch("colibri.export_results.write_exportgrid")
def test_run_ultranest_fit(mock_write_exportgrid, tmp_path):

    # Define mock ultranest fit
    mock_ultranest_fit = Mock()
    mock_ultranest_fit.resampled_posterior = jax.random.normal(
        jax.random.PRNGKey(0), (10, 2)
    )
    mock_ultranest_fit.param_names = ["param1", "param2"]
    mock_ultranest_fit.full_posterior_samples = jax.random.normal(
        jax.random.PRNGKey(0), (100, 2)
    )
    mock_ultranest_fit.bayes_complexity = 1.0
    mock_ultranest_fit.avg_chi2 = 1.0
    mock_ultranest_fit.min_chi2 = 1.0
    mock_ultranest_fit.logz = 1.0

    # Create mock pdf model
    mock_pdf_model = Mock()
    mock_pdf_model.param_names = ["param1", "param2"]

    # Run the run_analytic_fit function
    output_path = str(tmp_path)
    run_ultranest_fit(mock_ultranest_fit, output_path, mock_pdf_model)

    # Check if the write_exportgrid function was called for each sample
    assert (
        mock_write_exportgrid.call_count
        == mock_ultranest_fit.resampled_posterior.shape[0]
    )

    # Assertions - check if files are created in the output path
    assert (tmp_path / "ns_result.csv").exists()
    assert (tmp_path / "bayes_metrics.csv").exists()
    assert (tmp_path / "full_posterior_sample.csv").exists()
