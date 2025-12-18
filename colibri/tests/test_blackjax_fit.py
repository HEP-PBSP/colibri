"""
colibri.tests.test_blackjax_fit.py

Tests for the BlackJAX fitting module.
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
    MOCK_PENALTY_POSDATA,
    TEST_FK_ARRAYS,
    TEST_POS_FK_ARRAYS,
    TEST_XGRID,
)

from colibri.core import BlackJAXFit
from colibri.blackjax_fit import blackjax_fit, run_blackjax_fit
from colibri.likelihood import LogLikelihood

jax.config.update("jax_enable_x64", True)

bayesian_prior = {
    "sample": lambda key, n: jax.random.normal(
        key, (n, len(MOCK_PDF_MODEL.param_names))
    ),
    "log_prob": lambda params: -0.5 * jnp.sum(params**2),
}

integrability_penalty = lambda pdf: jnp.array([0.0])

blackjax_settings = {
    "seed": 42,
    "n_live": 50,
    "delete_fraction": 0.5,
    "repeats": 2,
    "log_precision": -1.0,
    "n_posterior_samples": 10,
    "posterior_resampling_seed": 123,
    "log_dir": "test_logs",
}


@pytest.mark.parametrize("pos_penalty", [True, False])
@patch("colibri.blackjax_fit.jax.jit", side_effect=lambda f: f)
def test_blackjax_fit_basic(mock_jit, pos_penalty):

    MOCK_PDF_MODEL.n_parameters = len(MOCK_PDF_MODEL.param_names)

    mock_log_likelihood = Mock(spec=LogLikelihood)
    mock_log_likelihood.return_value = jnp.array(-1.0)

    fit_result = blackjax_fit(
        MOCK_PDF_MODEL,
        bayesian_prior,
        blackjax_settings,
        mock_log_likelihood,
    )

    assert isinstance(fit_result, BlackJAXFit)
    assert fit_result.resampled_posterior.shape == (
        blackjax_settings["n_posterior_samples"],
        len(MOCK_PDF_MODEL.param_names),
    )
    assert fit_result.param_names == MOCK_PDF_MODEL.param_names
    assert fit_result.blackjax_specs == blackjax_settings
    assert isinstance(fit_result.blackjax_result, dict)


@patch("colibri.blackjax_fit.jax.jit", side_effect=lambda f: f)
def test_blackjax_fit_posterior_sample_limit(mock_jit):

    MOCK_PDF_MODEL.n_parameters = len(MOCK_PDF_MODEL.param_names)

    limited_settings = copy.deepcopy(blackjax_settings)
    limited_settings["n_posterior_samples"] = 1000

    mock_log_likelihood = Mock(spec=LogLikelihood)
    mock_log_likelihood.return_value = jnp.array(-1.0)

    fit_result = blackjax_fit(
        MOCK_PDF_MODEL,
        bayesian_prior,
        limited_settings,
        mock_log_likelihood,
    )

    assert fit_result.resampled_posterior.shape[1] == len(MOCK_PDF_MODEL.param_names)
    assert isinstance(fit_result, BlackJAXFit)


@patch("colibri.blackjax_fit.write_replicas")
@patch("colibri.blackjax_fit.export_bayes_results")
def test_run_blackjax_fit(mock_export_bayes, mock_write_replicas, tmp_path):
    """Test run_blackjax_fit output behaviour."""

    mock_fit = Mock(spec=BlackJAXFit)
    mock_fit.resampled_posterior = jnp.ones((10, 2))
    mock_fit.param_names = ["param1", "param2"]
    mock_fit.full_posterior_samples = jnp.ones((100, 2))
    mock_fit.bayesian_metrics = {
        "bayes_complexity": 1.0,
        "avg_chi2": 0.1,
        "min_chi2": 0.05,
        "logz": 5.0,
    }

    run_blackjax_fit(mock_fit, tmp_path, MOCK_PDF_MODEL)

    mock_export_bayes.assert_called_once_with(
        mock_fit,
        tmp_path,
        "ns_result",
    )
    mock_write_replicas.assert_called_once_with(
        mock_fit,
        tmp_path,
        MOCK_PDF_MODEL,
    )
