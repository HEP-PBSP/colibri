"""
colibri.tests.test_analytic_fit

Module to test the analytic fit functionality of the colibri package.
"""

import logging
from unittest.mock import Mock, patch

import jax.numpy as jnp
import jax.random
import pytest

from colibri.api import API as colibriAPI
from colibri.analytic_fit import AnalyticFit, analytic_fit, run_analytic_fit
from colibri.core import PriorSettings
from colibri.forward_map import FKTableForwardMap
from colibri.tests.conftest import (
    MOCK_CENTRAL_COVMAT_INDEX,
    MOCK_PDF_MODEL,
    TEST_FK_ARRAYS,
    TEST_PRIOR_SETTINGS_UNIFORM,
    TEST_XGRID,
    TEST_DATASETS,
)

analytic_settings = {
    "sampling_seed": 123,
    "full_sample_size": 100,
    "n_posterior_samples": 10,
}

TEST_DATA = colibriAPI.data(**TEST_DATASETS)


def test_analytic_fit_flat_direction():
    """
    Tests that the analytic fit raises a ValueError when the
    forward_map returns a flat direction in the parameter space.
    """
    n_params = len(MOCK_PDF_MODEL.param_names)

    forward_map = FKTableForwardMap(
        lambda pdf, fkarrs: jnp.ones(n_params),
        pdf_model=MOCK_PDF_MODEL,
    )

    with pytest.raises(ValueError):
        # Run the analytic fit and make sure that the Value Error is raised
        analytic_fit(
            MOCK_CENTRAL_COVMAT_INDEX,
            forward_map,
            MOCK_PDF_MODEL,
            analytic_settings,
            TEST_PRIOR_SETTINGS_UNIFORM,
            TEST_XGRID,
            TEST_FK_ARRAYS,
            TEST_DATA,
        )


def test_analytic_fit(caplog, monkeypatch):
    """
    Tests basic functionality of the analytic fit function.
    """

    # Mock the grid_values_func of the PDF model within the test to return the input parameters as the PDF grid values
    monkeypatch.setattr(
        MOCK_PDF_MODEL, "grid_values_func", lambda xgrid: lambda params: params
    )

    forward_map = FKTableForwardMap(lambda pdf, fkarrs: pdf, pdf_model=MOCK_PDF_MODEL)

    # Run the analytic fit
    result = analytic_fit(
        MOCK_CENTRAL_COVMAT_INDEX,
        forward_map,
        MOCK_PDF_MODEL,
        analytic_settings,
        TEST_PRIOR_SETTINGS_UNIFORM,
        TEST_XGRID,
        TEST_FK_ARRAYS,
        TEST_DATA,
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
            MOCK_PDF_MODEL,
            analytic_settings,
            TEST_PRIOR_SETTINGS_UNIFORM,
            TEST_XGRID,
            TEST_FK_ARRAYS,
            TEST_DATA,
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

    forward_map = FKTableForwardMap(lambda pdf, fkarrs: pdf, pdf_model=MOCK_PDF_MODEL)

    # Run the analytic fit
    result = analytic_fit(
        MOCK_CENTRAL_COVMAT_INDEX,
        forward_map,
        MOCK_PDF_MODEL,
        analytic_settings,
        PRIOR_SETTINGS1,
        TEST_XGRID,
        TEST_FK_ARRAYS,
        TEST_DATA,
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
        MOCK_PDF_MODEL,
        analytic_settings,
        PRIOR_SETTINGS2,
        TEST_XGRID,
        TEST_FK_ARRAYS,
        TEST_DATA,
    )


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

    # Run the run_analytic_fit function
    output_path = str(tmp_path)
    run_analytic_fit(mock_analytic_fit, output_path, MOCK_PDF_MODEL, Q0=1.65)

    # Check if the write_exportgrid function was called for each sample
    assert (
        mock_write_exportgrid.call_count
        == mock_analytic_fit.resampled_posterior.shape[0]
    )

    # Assertions - check if files are created in the output path
    assert (tmp_path / "analytic_result.csv").exists()
    assert (tmp_path / "bayes_metrics.csv").exists()
    assert (tmp_path / "full_posterior_sample.csv").exists()
