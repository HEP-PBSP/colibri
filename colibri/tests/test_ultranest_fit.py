"""
colibri.tests.test_ultranest_fit.py

Tests for the UltraNest fitting module.
"""

import copy
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import pytest

from colibri.tests.conftest import (
    MOCK_CENTRAL_COVMAT_INDEX,
    MOCK_PDF_MODEL,
    MOCK_PENALTY_POSDATA,
    TEST_FK_ARRAYS,
    TEST_POS_FK_ARRAYS,
    TEST_XGRID,
)
from colibri.ultranest_fit import UltranestFit, run_ultranest_fit, ultranest_fit
from colibri.likelihood import LogLikelihood
from colibri.core import BayesianPrior
from colibri.forward_map import FKTableForwardMap

jax.config.update("jax_enable_x64", True)


def mock_prior_transform(x):
    return x


def mock_log_prob(x):
    return jnp.array(0.0)


def mock_sample(rng_key, n_samples):
    n_params = len(MOCK_PDF_MODEL.param_names)
    return jax.random.uniform(rng_key, shape=(n_samples, n_params))


bayesian_prior = BayesianPrior(
    prior_transform=lambda x: x,
    log_prob=lambda x: -jnp.sum(x**2, axis=-1),
    sample=lambda rng, n: jnp.zeros((n, MOCK_PDF_MODEL.n_parameters)),
)


integrability_penalty = lambda pdf: jnp.array([0.0])

ultranest_settings = {
    "ultranest_seed": 42,
    "ReactiveNS_settings": {"vectorized": False},
    "SliceSampler_settings": None,
    "Run_settings": {"frac_remain": 0.5, "min_num_live_points": 5},
    "n_posterior_samples": 10,
    "posterior_resampling_seed": 123,
    "sampler_plot": False,
}

vect_ultranest_settings = copy.deepcopy(ultranest_settings)
vect_ultranest_settings["ReactiveNS_settings"]["vectorized"] = True


@pytest.mark.parametrize("pos_penalty", [True, False])
def test_ultranest_fit(pos_penalty):

    _pred_data = lambda *args: jnp.array([0.0])
    forward_map = FKTableForwardMap(
        _pred_data,
        pdf_model=MOCK_PDF_MODEL,
        pdf_grid_func=MOCK_PDF_MODEL.grid_values_func(TEST_XGRID),
    )
    mock_log_likelihood = LogLikelihood(
        MOCK_CENTRAL_COVMAT_INDEX,
        MOCK_PDF_MODEL,
        forward_map,
        TEST_FK_ARRAYS,
        TEST_POS_FK_ARRAYS,
        MOCK_PENALTY_POSDATA,
        positivity_penalty_settings={
            "positivity_penalty": pos_penalty,
            "alpha": 1e-7,
            "lambda_positivity": 1000,
        },
        integrability_penalty=integrability_penalty,
    )

    fit_result = ultranest_fit(
        forward_map,
        bayesian_prior,
        ultranest_settings,
        mock_log_likelihood,
    )

    assert isinstance(fit_result, UltranestFit)
    assert fit_result.resampled_posterior.shape == (
        ultranest_settings["n_posterior_samples"],
        len(MOCK_PDF_MODEL.param_names),
    )
    assert fit_result.param_names == ["param1", "param2"]
    assert fit_result.ultranest_specs == ultranest_settings
    assert isinstance(fit_result.ultranest_result, dict)


@pytest.mark.parametrize("pos_penalty", [True, False])
def test_ultranest_fit_vectorized(pos_penalty):

    _pred_data = lambda *args: jnp.array([0.0])
    forward_map = FKTableForwardMap(
        _pred_data,
        pdf_model=MOCK_PDF_MODEL,
        pdf_grid_func=MOCK_PDF_MODEL.grid_values_func(TEST_XGRID),
    )
    ultranest_settings["ReactiveNS_settings"]["vectorized"] = True

    mock_log_likelihood = LogLikelihood(
        MOCK_CENTRAL_COVMAT_INDEX,
        MOCK_PDF_MODEL,
        forward_map,
        TEST_FK_ARRAYS,
        TEST_POS_FK_ARRAYS,
        MOCK_PENALTY_POSDATA,
        positivity_penalty_settings={
            "positivity_penalty": pos_penalty,
            "alpha": 1e-7,
            "lambda_positivity": 1000,
        },
        integrability_penalty=integrability_penalty,
    )

    fit_result = ultranest_fit(
        forward_map,
        bayesian_prior,
        ultranest_settings,
        mock_log_likelihood,
    )

    assert isinstance(fit_result, UltranestFit)
    assert fit_result.resampled_posterior.shape == (
        ultranest_settings["n_posterior_samples"],
        len(MOCK_PDF_MODEL.param_names),
    )
    assert fit_result.param_names == ["param1", "param2"]
    assert fit_result.ultranest_specs == ultranest_settings
    assert isinstance(fit_result.ultranest_result, dict)


@pytest.mark.parametrize("pos_penalty", [True, False])
def test_ultranest_fit_with_SliceSampler(pos_penalty):
    ultranest_settings = {
        "ultranest_seed": 42,
        "ReactiveNS_settings": {"vectorized": False},
        "SliceSampler_settings": {"nsteps": 10},
        "Run_settings": {"frac_remain": 0.5, "min_num_live_points": 5},
        "n_posterior_samples": 10,
        "posterior_resampling_seed": 123,
        "sampler_plot": False,
        "popstepsampler": False,
    }

    _pred_data = lambda *args: jnp.array([0.0])
    forward_map = FKTableForwardMap(
        _pred_data,
        pdf_model=MOCK_PDF_MODEL,
        pdf_grid_func=MOCK_PDF_MODEL.grid_values_func(TEST_XGRID),
    )

    mock_log_likelihood = LogLikelihood(
        MOCK_CENTRAL_COVMAT_INDEX,
        MOCK_PDF_MODEL,
        forward_map,
        TEST_FK_ARRAYS,
        TEST_POS_FK_ARRAYS,
        MOCK_PENALTY_POSDATA,
        positivity_penalty_settings={
            "positivity_penalty": pos_penalty,
            "alpha": 1e-7,
            "lambda_positivity": 1000,
        },
        integrability_penalty=integrability_penalty,
    )

    fit_result = ultranest_fit(
        forward_map,
        bayesian_prior,
        ultranest_settings,
        mock_log_likelihood,
    )

    assert isinstance(fit_result, UltranestFit)
    assert fit_result.resampled_posterior.shape == (
        ultranest_settings["n_posterior_samples"],
        len(MOCK_PDF_MODEL.param_names),
    )
    assert fit_result.param_names == ["param1", "param2"]
    assert fit_result.ultranest_specs == ultranest_settings
    assert isinstance(fit_result.ultranest_result, dict)


@pytest.mark.parametrize("pos_penalty", [True, False])
def test_ultranest_fit_with_popSliceSampler(pos_penalty):
    ultranest_settings = {
        "ultranest_seed": 42,
        "ReactiveNS_settings": {"vectorized": False},
        "SliceSampler_settings": {"nsteps": 10, "popsize": 10},
        "Run_settings": {"frac_remain": 0.5, "min_num_live_points": 5},
        "n_posterior_samples": 10,
        "posterior_resampling_seed": 123,
        "sampler_plot": False,
        "popstepsampler": True,
    }

    _pred_data = lambda *args: jnp.array([0.0])
    forward_map = FKTableForwardMap(
        _pred_data,
        pdf_model=MOCK_PDF_MODEL,
        pdf_grid_func=MOCK_PDF_MODEL.grid_values_func(TEST_XGRID),
    )

    mock_log_likelihood = LogLikelihood(
        MOCK_CENTRAL_COVMAT_INDEX,
        MOCK_PDF_MODEL,
        forward_map,
        TEST_FK_ARRAYS,
        TEST_POS_FK_ARRAYS,
        MOCK_PENALTY_POSDATA,
        positivity_penalty_settings={
            "positivity_penalty": pos_penalty,
            "alpha": 1e-7,
            "lambda_positivity": 1000,
        },
        integrability_penalty=integrability_penalty,
    )

    fit_result = ultranest_fit(
        forward_map,
        bayesian_prior,
        ultranest_settings,
        mock_log_likelihood,
    )

    assert isinstance(fit_result, UltranestFit)
    assert fit_result.resampled_posterior.shape == (
        ultranest_settings["n_posterior_samples"],
        len(MOCK_PDF_MODEL.param_names),
    )
    assert fit_result.param_names == ["param1", "param2"]
    assert fit_result.ultranest_specs == ultranest_settings
    assert isinstance(fit_result.ultranest_result, dict)


@patch("ultranest.ReactiveNestedSampler")
@pytest.mark.parametrize("pos_penalty", [True, False])
def test_ultranest_fit_with_sampler_plot(mock_sampler_class, pos_penalty):
    """Test the ultranest_fit function with sampler_plot=True to cover the plotting lines."""

    # Create settings with sampler_plot enabled
    ultranest_settings_with_plot = {
        "ultranest_seed": 42,
        "ReactiveNS_settings": {"vectorized": False},
        "SliceSampler_settings": None,
        "Run_settings": {"frac_remain": 0.5, "min_num_live_points": 5},
        "n_posterior_samples": 10,
        "posterior_resampling_seed": 123,
        "sampler_plot": True,  # Enable plotting
        "popstepsampler": False,
    }

    _pred_data = lambda *args: jnp.array([0.0])
    forward_map = FKTableForwardMap(
        _pred_data,
        pdf_model=MOCK_PDF_MODEL,
        pdf_grid_func=MOCK_PDF_MODEL.grid_values_func(TEST_XGRID),
    )

    mock_log_likelihood = LogLikelihood(
        MOCK_CENTRAL_COVMAT_INDEX,
        MOCK_PDF_MODEL,
        forward_map,
        TEST_FK_ARRAYS,
        TEST_POS_FK_ARRAYS,
        MOCK_PENALTY_POSDATA,
        positivity_penalty_settings={
            "positivity_penalty": pos_penalty,
            "alpha": 1e-7,
            "lambda_positivity": 1000,
        },
        integrability_penalty=integrability_penalty,
    )

    # Mock the sampler instance
    mock_sampler_instance = Mock()
    mock_sampler_class.return_value = mock_sampler_instance

    # Mock the run method to return the expected ultranest result
    mock_ultranest_result = {
        "samples": jnp.ones((20, 2)),  # Mock samples
        "maximum_likelihood": {"logl": -0.05},  # Mock maximum likelihood
        "logz": 7.0,  # Mock log evidence
    }
    mock_sampler_instance.run.return_value = mock_ultranest_result

    # Mock the plot method
    mock_sampler_instance.plot = Mock()

    fit_result = ultranest_fit(
        forward_map,
        bayesian_prior,
        ultranest_settings_with_plot,
        mock_log_likelihood,
    )

    # Verify that the sampler.plot() method was called
    mock_sampler_instance.plot.assert_called_once()

    # Verify the rest of the functionality
    assert isinstance(fit_result, UltranestFit)
    assert fit_result.resampled_posterior.shape == (
        ultranest_settings_with_plot["n_posterior_samples"],
        len(MOCK_PDF_MODEL.param_names),
    )
    assert fit_result.param_names == ["param1", "param2"]
    assert fit_result.ultranest_specs == ultranest_settings_with_plot
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
    mock_ultranest_fit.bayesian_metrics = {
        "bayes_complexity": 2.0,
        "avg_chi2": 0.3,
        "avg_chi2_reduced": 0.009,
        "min_chi2": 0.1,
        "logz": 7.0,
    }

    # Run the run_ultranest_fit function
    output_path = str(tmp_path)
    run_ultranest_fit(mock_ultranest_fit, output_path, MOCK_PDF_MODEL, Q0=1.65)

    # Check if the write_exportgrid function was called for each sample
    assert (
        mock_write_exportgrid.call_count
        == mock_ultranest_fit.resampled_posterior.shape[0]
    )

    # Assertions - check if files are created in the output path
    assert (tmp_path / "ns_result.csv").exists()
    assert (tmp_path / "bayes_metrics.csv").exists()
    assert (tmp_path / "full_posterior_sample.csv").exists()
