import jax.numpy as jnp
import jax.random
from unittest.mock import Mock, patch
from colibri.analytic_fit import AnalyticFit, analytic_fit, run_analytic_fit
from colibri.tests.conftest import TEST_FK_ARRAYS
import logging
import pytest

mock_central_inv_covmat_index = Mock()
mock_central_inv_covmat_index.central_values = jnp.ones(2)
mock_central_inv_covmat_index.inv_covmat = jnp.eye(2)

analytic_settings = {
    "sampling_seed": 123,
    "full_sample_size": 100,
    "n_posterior_samples": 10,
    "min_max_prior": True,
    "n_sigma_prior": False,
    "n_sigma_value": 5,
}

# Define mock input parameters
bayesian_prior = lambda x: x
FIT_XGRID = jnp.logspace(-7, 0, 50)


def test_analytic_fit_flat_direction():
    # Create mock pdf model
    mock_pdf_model = Mock()
    mock_pdf_model.param_names = ["param1", "param2"]
    mock_pdf_model.grid_values_func = lambda xgrid: lambda params: jnp.ones(
        (14, len(xgrid))
    )
    mock_pdf_model.pred_and_pdf_func = lambda xgrid, forward_map: (
        lambda params, fkarrs: (jnp.ones_like(params), jnp.ones((14, len(xgrid))))
    )

    _pred_data = None

    with pytest.raises(ValueError):
        # Run the analytic fit and make sure that the Value Error is raised
        analytic_fit(
            mock_central_inv_covmat_index,
            _pred_data,
            mock_pdf_model,
            analytic_settings,
            bayesian_prior,
            FIT_XGRID,
            TEST_FK_ARRAYS,
        )


def test_analytic_fit(caplog):
    # Create mock pdf model
    mock_pdf_model = Mock()
    mock_pdf_model.param_names = ["param1", "param2"]
    mock_pdf_model.grid_values_func = lambda xgrid: lambda params: jnp.ones(
        (14, len(xgrid))
    )
    mock_pdf_model.pred_and_pdf_func = lambda xgrid, forward_map: (
        lambda params, fkarrs: (params, jnp.ones((14, len(xgrid))))
    )

    _pred_data = None

    # Run the analytic fit
    result = analytic_fit(
        mock_central_inv_covmat_index,
        _pred_data,
        mock_pdf_model,
        analytic_settings,
        bayesian_prior,
        FIT_XGRID,
        TEST_FK_ARRAYS,
    )

    assert isinstance(result, AnalyticFit)

    assert result.analytic_specs == analytic_settings
    assert (
        result.resampled_posterior.shape[0] == analytic_settings["n_posterior_samples"]
    )
    assert len(result.param_names) == len(mock_pdf_model.param_names)

    # Check that it works if min_max_prior is False
    analytic_settings["min_max_prior"] = False
    # Run the analytic fit
    with caplog.at_level(logging.ERROR):  # Set the log level to ERROR
        result_2 = analytic_fit(
            mock_central_inv_covmat_index,
            _pred_data,
            mock_pdf_model,
            analytic_settings,
            bayesian_prior,
            FIT_XGRID,
            TEST_FK_ARRAYS,
        )

    # Check that an error message was logged, because the prior was not wide enough
    error_logged = any(record.levelno == logging.ERROR for record in caplog.records)
    assert error_logged, "No error message was logged"

    assert result_2.analytic_specs == analytic_settings
    assert (
        result_2.resampled_posterior.shape[0]
        == analytic_settings["n_posterior_samples"]
    )
    assert len(result_2.param_names) == len(mock_pdf_model.param_names)


def test_analytic_fit_nsigma_prior(caplog):

    analytic_settings["min_max_prior"] = False
    analytic_settings["n_sigma_prior"] = True

    # Create mock pdf model
    mock_pdf_model = Mock()
    mock_pdf_model.param_names = ["param1", "param2"]
    mock_pdf_model.grid_values_func = lambda xgrid: lambda params: jnp.ones(
        (14, len(xgrid))
    )
    mock_pdf_model.pred_and_pdf_func = lambda xgrid, forward_map: (
        lambda params, fkarrs: (params, jnp.ones((14, len(xgrid))))
    )

    _pred_data = None

    # Run the analytic fit
    result = analytic_fit(
        mock_central_inv_covmat_index,
        _pred_data,
        mock_pdf_model,
        analytic_settings,
        bayesian_prior,
        FIT_XGRID,
        TEST_FK_ARRAYS,
    )

    assert isinstance(result, AnalyticFit)

    assert result.analytic_specs == analytic_settings
    assert (
        result.resampled_posterior.shape[0] == analytic_settings["n_posterior_samples"]
    )
    assert len(result.param_names) == len(mock_pdf_model.param_names)


@patch("colibri.export_results.write_exportgrid")
def test_run_analytic_fit(mock_write_exportgrid, tmp_path):

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
    mock_analytic_fit.bayes_complexity = 2.0
    mock_analytic_fit.avg_chi2 = 0.3
    mock_analytic_fit.min_chi2 = 0.1
    mock_analytic_fit.logz = 7.0

    # Create mock pdf model
    mock_pdf_model = Mock()
    mock_pdf_model.param_names = ["param1", "param2"]

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
