"""
colibri.tests.test_ultranest_fit.py

Tests for the UltraNest fitting module.
"""

import copy
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import pytest


from colibri.loss_functions import chi2
from colibri.tests.conftest import (
    MOCK_CENTRAL_INV_COVMAT_INDEX,
    MOCK_PDF_MODEL,
    TEST_FK_ARRAYS,
    TEST_POS_FK_ARRAYS,
    TEST_XGRID,
    UltraNestLogLikelihoodMock,
)
from colibri.ultranest_fit import UltranestFit, run_ultranest_fit, ultranest_fit

jax.config.update("jax_enable_x64", True)

# Define mock input parameters
bayesian_prior = lambda x: x
mock_chi2 = lambda central_values, predictions, inv_covmat: 0.0

_penalty_posdata = (
    lambda pdf, alpha, lambda_positivity, positivity_fast_kernel_arrays: jnp.array(
        [1.0, 1.0]
    )
)

integrability_penalty = lambda pdf: jnp.array([0.0])

ns_settings = {
    "ultranest_seed": 42,
    "ReactiveNS_settings": {"vectorized": False},
    "SliceSampler_settings": None,
    "Run_settings": {"frac_remain": 0.5, "min_num_live_points": 5},
    "n_posterior_samples": 10,
    "posterior_resampling_seed": 123,
    "sampler_plot": False,
}

vect_ns_settings = copy.deepcopy(ns_settings)
vect_ns_settings["ReactiveNS_settings"]["vectorized"] = True


@pytest.mark.parametrize("pos_penalty", [True, False])
def test_ultranest_fit(pos_penalty):

    _pred_data = None
    mock_log_likelihood = UltraNestLogLikelihoodMock(
        MOCK_CENTRAL_INV_COVMAT_INDEX,
        MOCK_PDF_MODEL,
        TEST_XGRID,
        _pred_data,
        TEST_FK_ARRAYS,
        TEST_POS_FK_ARRAYS,
        ns_settings,
        chi2,
        _penalty_posdata,
        positivity_penalty_settings={
            "positivity_penalty": pos_penalty,
            "alpha": 1e-7,
            "lambda_positivity": 1000,
        },
    )

    fit_result = ultranest_fit(
        MOCK_PDF_MODEL,
        bayesian_prior,
        ns_settings,
        mock_log_likelihood,
    )

    assert isinstance(fit_result, UltranestFit)
    assert fit_result.resampled_posterior.shape == (
        ns_settings["n_posterior_samples"],
        len(MOCK_PDF_MODEL.param_names),
    )
    assert fit_result.param_names == ["param1", "param2"]
    assert fit_result.ultranest_specs == ns_settings
    assert isinstance(fit_result.ultranest_result, dict)


@pytest.mark.parametrize("pos_penalty", [True, False])
def test_ultranest_fit_vectorized(pos_penalty):

    _pred_data = None
    ns_settings["ReactiveNS_settings"]["vectorized"] = True

    mock_log_likelihood = UltraNestLogLikelihoodMock(
        MOCK_CENTRAL_INV_COVMAT_INDEX,
        MOCK_PDF_MODEL,
        TEST_XGRID,
        _pred_data,
        TEST_FK_ARRAYS,
        TEST_POS_FK_ARRAYS,
        ns_settings,
        chi2,
        _penalty_posdata,
        positivity_penalty_settings={
            "positivity_penalty": pos_penalty,
            "alpha": 1e-7,
            "lambda_positivity": 1000,
        },
    )

    fit_result = ultranest_fit(
        MOCK_PDF_MODEL,
        bayesian_prior,
        ns_settings,
        mock_log_likelihood,
    )

    assert isinstance(fit_result, UltranestFit)
    assert fit_result.resampled_posterior.shape == (
        ns_settings["n_posterior_samples"],
        len(MOCK_PDF_MODEL.param_names),
    )
    assert fit_result.param_names == ["param1", "param2"]
    assert fit_result.ultranest_specs == ns_settings
    assert isinstance(fit_result.ultranest_result, dict)


@pytest.mark.parametrize("pos_penalty", [True, False])
def test_ultranest_fit_with_SliceSampler(pos_penalty):
    ns_settings = {
        "ultranest_seed": 42,
        "ReactiveNS_settings": {"vectorized": False},
        "SliceSampler_settings": {"nsteps": 10},
        "Run_settings": {"frac_remain": 0.5, "min_num_live_points": 5},
        "n_posterior_samples": 10,
        "posterior_resampling_seed": 123,
        "sampler_plot": False,
        "popstepsampler": False,
    }

    _pred_data = None

    mock_log_likelihood = UltraNestLogLikelihoodMock(
        MOCK_CENTRAL_INV_COVMAT_INDEX,
        MOCK_PDF_MODEL,
        TEST_XGRID,
        _pred_data,
        TEST_FK_ARRAYS,
        TEST_POS_FK_ARRAYS,
        ns_settings,
        chi2,
        _penalty_posdata,
        positivity_penalty_settings={
            "positivity_penalty": pos_penalty,
            "alpha": 1e-7,
            "lambda_positivity": 1000,
        },
    )

    fit_result = ultranest_fit(
        MOCK_PDF_MODEL,
        bayesian_prior,
        ns_settings,
        mock_log_likelihood,
    )

    assert isinstance(fit_result, UltranestFit)
    assert fit_result.resampled_posterior.shape == (
        ns_settings["n_posterior_samples"],
        len(MOCK_PDF_MODEL.param_names),
    )
    assert fit_result.param_names == ["param1", "param2"]
    assert fit_result.ultranest_specs == ns_settings
    assert isinstance(fit_result.ultranest_result, dict)


@pytest.mark.parametrize("pos_penalty", [True, False])
def test_ultranest_fit_with_popSliceSampler(pos_penalty):
    ns_settings = {
        "ultranest_seed": 42,
        "ReactiveNS_settings": {"vectorized": False},
        "SliceSampler_settings": {"nsteps": 10, "popsize": 10},
        "Run_settings": {"frac_remain": 0.5, "min_num_live_points": 5},
        "n_posterior_samples": 10,
        "posterior_resampling_seed": 123,
        "sampler_plot": False,
        "popstepsampler": True,
    }

    _pred_data = None

    mock_log_likelihood = UltraNestLogLikelihoodMock(
        MOCK_CENTRAL_INV_COVMAT_INDEX,
        MOCK_PDF_MODEL,
        TEST_XGRID,
        _pred_data,
        TEST_FK_ARRAYS,
        TEST_POS_FK_ARRAYS,
        ns_settings,
        chi2,
        _penalty_posdata,
        positivity_penalty_settings={
            "positivity_penalty": pos_penalty,
            "alpha": 1e-7,
            "lambda_positivity": 1000,
        },
    )

    fit_result = ultranest_fit(
        MOCK_PDF_MODEL,
        bayesian_prior,
        ns_settings,
        mock_log_likelihood,
    )

    assert isinstance(fit_result, UltranestFit)
    assert fit_result.resampled_posterior.shape == (
        ns_settings["n_posterior_samples"],
        len(MOCK_PDF_MODEL.param_names),
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
    mock_ultranest_fit.bayes_complexity = 2.0
    mock_ultranest_fit.avg_chi2 = 0.3
    mock_ultranest_fit.min_chi2 = 0.1
    mock_ultranest_fit.logz = 7.0

    # Run the run_ultranest_fit function
    output_path = str(tmp_path)
    run_ultranest_fit(mock_ultranest_fit, output_path, MOCK_PDF_MODEL)

    # Check if the write_exportgrid function was called for each sample
    assert (
        mock_write_exportgrid.call_count
        == mock_ultranest_fit.resampled_posterior.shape[0]
    )

    # Assertions - check if files are created in the output path
    assert (tmp_path / "ns_result.csv").exists()
    assert (tmp_path / "bayes_metrics.csv").exists()
    assert (tmp_path / "full_posterior_sample.csv").exists()
