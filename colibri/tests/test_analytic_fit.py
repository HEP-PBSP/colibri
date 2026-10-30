"""
colibri.tests.test_analytic_fit

Module to test the analytic fit functionality of the colibri package.
"""

import logging
from unittest.mock import Mock, patch

import jax.numpy as jnp
import jax.random
import numpy as np
import pytest

from colibri.analytic_fit import AnalyticFit, analytic_fit, run_analytic_fit
from colibri.core import PriorSettings
from colibri.forward_map import FKTableForwardMap
from colibri.tests.conftest import (
    MOCK_CENTRAL_COVMAT_INDEX,
    MOCK_PDF_MODEL,
    TEST_FK_ARRAYS,
    TEST_PRIOR_SETTINGS_UNIFORM,
    TEST_XGRID,
)

analytic_settings = {
    "sampling_seed": 123,
    "full_sample_size": 100,
    "n_posterior_samples": 10,
    "sampler_plot": False,
}


def test_analytic_fit_flat_direction():
    """
    Tests that the analytic fit raises a ValueError when the
    forward_map returns a flat direction in the parameter space.
    """
    n_params = len(MOCK_PDF_MODEL.param_names)

    forward_map = FKTableForwardMap(
        lambda pdf, fkarrs: jnp.ones(n_params),
        pdf_model=MOCK_PDF_MODEL,
        pdf_grid_func=MOCK_PDF_MODEL.grid_values_func(TEST_XGRID),
    )

    with pytest.raises(ValueError):
        # Run the analytic fit and make sure that the Value Error is raised
        analytic_fit(
            MOCK_CENTRAL_COVMAT_INDEX,
            forward_map,
            analytic_settings,
            TEST_FK_ARRAYS,
            TEST_PRIOR_SETTINGS_UNIFORM,
        )


def test_analytic_fit(caplog, monkeypatch):
    """
    Tests basic functionality of the analytic fit function.
    """

    # Mock the grid_values_func of the PDF model within the test to return the input parameters as the PDF grid values
    monkeypatch.setattr(
        MOCK_PDF_MODEL, "grid_values_func", lambda xgrid: lambda params: params
    )

    forward_map = FKTableForwardMap(
        lambda pdf, fkarrs: pdf,
        pdf_model=MOCK_PDF_MODEL,
        pdf_grid_func=MOCK_PDF_MODEL.grid_values_func(TEST_XGRID),
    )

    # Run the analytic fit
    result = analytic_fit(
        MOCK_CENTRAL_COVMAT_INDEX,
        forward_map,
        analytic_settings,
        TEST_FK_ARRAYS,
        TEST_PRIOR_SETTINGS_UNIFORM,
    )

    assert isinstance(result, AnalyticFit)

    assert result.analytic_specs == analytic_settings
    assert (
        result.resampled_posterior.shape[0] == analytic_settings["n_posterior_samples"]
    )
    assert len(result.param_names) == len(MOCK_PDF_MODEL.param_names)

    # Check that it works if min_max_prior is False
    analytic_settings["min_max_prior"] = False
    # Run the analytic fit
    with caplog.at_level(logging.ERROR):  # Set the log level to ERROR
        result_2 = analytic_fit(
            MOCK_CENTRAL_COVMAT_INDEX,
            forward_map,
            analytic_settings,
            TEST_FK_ARRAYS,
            TEST_PRIOR_SETTINGS_UNIFORM,
        )

    # Check that an error message was logged, because the prior was not wide enough
    error_logged = any(record.levelno == logging.ERROR for record in caplog.records)
    assert error_logged, "No error message was logged"

    assert result_2.analytic_specs == analytic_settings
    assert (
        result_2.resampled_posterior.shape[0]
        == analytic_settings["n_posterior_samples"]
    )
    assert len(result_2.param_names) == len(MOCK_PDF_MODEL.param_names)


def test_analytic_fit_different_priors(caplog, monkeypatch):

    PRIOR_SETTINGS1 = PriorSettings(
        **{
            "prior_distribution": "n_sigma_prior",
            "prior_distribution_specs": {"n_sigma_value": 2.0},
        }
    )

    # Mock the grid_values_func of the PDF model within the test to return the input parameters as the PDF grid values
    monkeypatch.setattr(
        MOCK_PDF_MODEL, "grid_values_func", lambda xgrid: lambda params: params
    )

    forward_map = FKTableForwardMap(
        lambda pdf, fkarrs: pdf,
        pdf_model=MOCK_PDF_MODEL,
        pdf_grid_func=MOCK_PDF_MODEL.grid_values_func(TEST_XGRID),
    )

    # Run the analytic fit
    result = analytic_fit(
        MOCK_CENTRAL_COVMAT_INDEX,
        forward_map,
        analytic_settings,
        TEST_FK_ARRAYS,
        PRIOR_SETTINGS1,
    )

    assert isinstance(result, AnalyticFit)

    assert result.analytic_specs == analytic_settings
    assert (
        result.resampled_posterior.shape[0] == analytic_settings["n_posterior_samples"]
    )
    assert len(result.param_names) == len(MOCK_PDF_MODEL.param_names)

    PRIOR_SETTINGS2 = PriorSettings(
        **{
            "prior_distribution": "custom_uniform_parameter_prior",
            "prior_distribution_specs": {"upper_bounds": [2.0], "lower_bounds": [-2.0]},
        }
    )

    # Run the analytic fit with custom uniform prior
    result = analytic_fit(
        MOCK_CENTRAL_COVMAT_INDEX,
        forward_map,
        analytic_settings,
        TEST_FK_ARRAYS,
        PRIOR_SETTINGS2,
    )
    assert isinstance(result, AnalyticFit)

    PRIOR_SETTINGS3 = PriorSettings(
        **{
            "prior_distribution": "min_max_prior",
            "prior_distribution_specs": {},
        }
    )

    # Run the analytic fit with min-max prior
    with caplog.at_level(logging.INFO):
        result = analytic_fit(
            MOCK_CENTRAL_COVMAT_INDEX,
            forward_map,
            analytic_settings,
            TEST_FK_ARRAYS,
            PRIOR_SETTINGS3,
        )
    assert isinstance(result, AnalyticFit)
    assert "Using min-max prior" in caplog.text


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
    mock_analytic_fit.bayesian_metrics = {
        "bayes_complexity": 2.0,
        "avg_chi2": 0.3,
        "avg_chi2_reduced": 0.01,
        "min_chi2": 0.1,
        "min_chi2_reduced": 0.0033,
        "logz": 7.0,
    }

    # Run the run_analytic_fit function
    output_path = str(tmp_path)
    run_analytic_fit(
        mock_analytic_fit, analytic_settings, output_path, MOCK_PDF_MODEL, Q0=1.65
    )

    # Check if the write_exportgrid function was called for each sample
    assert (
        mock_write_exportgrid.call_count
        == mock_analytic_fit.resampled_posterior.shape[0]
    )

    # Assertions - check if files are created in the output path
    assert (tmp_path / "analytic_result.csv").exists()
    assert (tmp_path / "bayes_metrics.csv").exists()
    assert (tmp_path / "full_posterior_sample.csv").exists()


@patch("colibri.analytic_fit.plot_bayes_plots")
def test_run_analytic_fit_with_sampler_plot(mock_plot_bayes_plots, tmp_path):
    """Test the run_analytic_fit function with sampler_plot=True to cover the plotting lines."""

    # Create settings with sampler_plot enabled
    analytic_settings_with_plot = {
        "sampling_seed": 123,
        "full_sample_size": 100,
        "n_posterior_samples": 10,
        "sampler_plot": True,  # Enable plotting
    }

    # Define mock analytic fit
    mock_analytic_fit = Mock()
    mock_analytic_fit.analytic_specs = analytic_settings_with_plot
    mock_analytic_fit.resampled_posterior = jax.random.normal(
        jax.random.PRNGKey(0), (10, 2)
    )
    mock_analytic_fit.param_names = ["param1", "param2"]
    mock_analytic_fit.full_posterior_samples = jax.random.normal(
        jax.random.PRNGKey(0), (100, 2)
    )
    mock_analytic_fit.bayesian_metrics = {
        "bayes_complexity": 2.0,
        "avg_chi2": 0.3,
        "avg_chi2_reduced": 0.01,
        "min_chi2": 0.1,
        "min_chi2_reduced": 0.0033,
        "logz": 7.0,
    }

    # Run the run_analytic_fit function
    run_analytic_fit(
        mock_analytic_fit,
        analytic_settings_with_plot,
        tmp_path,
        MOCK_PDF_MODEL,
        Q0=1.65,
    )

    mock_plot_bayes_plots.assert_called_once_with(
        mock_analytic_fit, tmp_path, "analytic_logs"
    )


# ---------------------------------------------------------------------------
# Forward map with extra parameters
# ---------------------------------------------------------------------------

from colibri.forward_map import ForwardMap


class _LinearForwardMapWithExtra(ForwardMap):
    """Predictions (a, norm): one PDF parameter and one extra parameter."""

    def __init__(self):
        pdf_model = Mock()
        pdf_model.param_names = ["a"]
        super().__init__(pdf_model, extra_param_names=["norm"])

    def __call__(self, fk_tables, params):
        return jnp.array([params[0], params[1]]), jnp.zeros((1, 1))


def _norm_prior(low, high):
    return PriorSettings(
        prior_distribution="uniform_parameter_prior",
        prior_distribution_specs={"bounds": {"norm": [low, high]}},
    )


def _analytic_fit_with_extra(extra_prior_settings):
    return analytic_fit(
        MOCK_CENTRAL_COVMAT_INDEX,
        _LinearForwardMapWithExtra(),
        {**analytic_settings, "full_sample_size": 1000},
        TEST_FK_ARRAYS,
        PriorSettings(
            prior_distribution="uniform_parameter_prior",
            prior_distribution_specs={"min_val": -10.0, "max_val": 10.0},
        ),
        extra_prior_settings=extra_prior_settings,
    )


def test_analytic_fit_extra_parameters_prior():
    """The prior of the extra parameters comes from extra_prior_settings."""
    result = _analytic_fit_with_extra(_norm_prior(-10.0, 10.0))
    assert result.param_names == ["a", "norm"]

    # Doubling the width of the prior of norm costs log(2) of evidence
    wider = _analytic_fit_with_extra(_norm_prior(-20.0, 20.0))
    np.testing.assert_allclose(
        result.bayesian_metrics["logz"] - wider.bayesian_metrics["logz"],
        np.log(2),
        rtol=1e-6,
    )


def test_analytic_fit_extra_parameters_need_prior():
    with pytest.raises(ValueError, match="extra_prior_settings"):
        _analytic_fit_with_extra(None)


class _LinearForwardMapExtraOnly(ForwardMap):
    """No PDF parameters (as with fixed_pdf), two extra parameters."""

    def __init__(self):
        super().__init__(None, extra_param_names=["c1", "c2"])

    def __call__(self, fk_tables, params):
        return jnp.array([params[0], params[1]]), jnp.zeros((1, 1))


def test_analytic_fit_without_pdf_parameters_needs_no_prior_settings():
    result = analytic_fit(
        MOCK_CENTRAL_COVMAT_INDEX,
        _LinearForwardMapExtraOnly(),
        {**analytic_settings, "full_sample_size": 1000},
        TEST_FK_ARRAYS,
        extra_prior_settings=PriorSettings(
            prior_distribution="uniform_parameter_prior",
            prior_distribution_specs={"min_val": -10.0, "max_val": 10.0},
        ),
    )
    assert result.param_names == ["c1", "c2"]


def test_analytic_fit_pdf_parameters_need_prior_settings():
    with pytest.raises(ValueError, match="prior_settings"):
        analytic_fit(
            MOCK_CENTRAL_COVMAT_INDEX,
            _LinearForwardMapWithExtra(),
            analytic_settings,
            TEST_FK_ARRAYS,
            extra_prior_settings=_norm_prior(-10.0, 10.0),
        )
