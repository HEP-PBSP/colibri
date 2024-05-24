import numpy as np
import jax.numpy as jnp
import jax.random
from unittest.mock import Mock, patch
from colibri.analytic_fit import analytic_fit, run_analytic_fit

mock_pdf_model = Mock()
mock_pdf_model.param_names = ["param1", "param2"]
mock_pdf_model.grid_values_func = lambda xgrid: lambda params: jnp.ones(
    (14, len(xgrid))
)
mock_pdf_model.pred_and_pdf_func = lambda xgrid, forward_map: (
    lambda params: (params, jnp.ones((14, len(xgrid))))
)

mock_central_covmat_index = Mock()
mock_central_covmat_index.central_values = jnp.ones(2)
mock_central_covmat_index.covmat = jnp.eye(2)

analytic_settings = {
    "sampling_seed": 123,
    "full_sample_size": 100,
    "n_posterior_samples": 10,
    "optimal_prior": True,
}

_pred_data = None

# Define mock analytic fit
mock_analytic_fit = Mock()
mock_analytic_fit.analytic_specs = analytic_settings
mock_analytic_fit.resampled_posterior = jax.random.normal(
    jax.random.PRNGKey(0), (10, 2)
)
mock_analytic_fit.param_names = ["param1", "param2"]
mock_analytic_fit.full_posterior_samples = jax.random.normal(
    jax.random.PRNGKey(0), (100, 2)
)
mock_analytic_fit.bayes_complexity = 1.0
mock_analytic_fit.avg_chi2 = 1.0
mock_analytic_fit.min_chi2 = 1.0
mock_analytic_fit.logz = 1.0


def test_analytic_fit():
    # Define mock input parameters
    bayesian_prior = lambda x: jnp.ones_like(x)  # Mock flat prior
    FIT_XGRID = np.linspace(0, 1, 50)  # Example FIT_XGRID

    # Run the analytic fit
    result = analytic_fit(
        mock_central_covmat_index,
        _pred_data,
        mock_pdf_model,
        analytic_settings,
        bayesian_prior,
        FIT_XGRID,
    )

    # Assertions
    assert result.analytic_specs == analytic_settings
    assert (
        result.resampled_posterior.shape[0] == analytic_settings["n_posterior_samples"]
    )
    assert len(result.param_names) == len(mock_pdf_model.param_names)


@patch("colibri.export_results.write_exportgrid")
def test_run_analytic_fit(mock_write_exportgrid, tmp_path):

    # Run the run_analytic_fit function
    output_path = str(tmp_path)
    run_analytic_fit(mock_analytic_fit, output_path, mock_pdf_model)

    # Check if the write_exportgrid function was called for each sample
    assert (
        mock_write_exportgrid.call_count
        == mock_analytic_fit.resampled_posterior.shape[0]
    )

    # Assertions - check if files are created in the output path
    assert (tmp_path / "analytic_result.csv").exists()
    assert (tmp_path / "bayes_metrics.csv").exists()
    assert (tmp_path / "full_posterior_sample.csv").exists()
