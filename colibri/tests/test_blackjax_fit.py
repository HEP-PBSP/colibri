"""
colibri.tests.test_blackjax_fit.py
Tests for the BlackJAX fitting module.
"""

from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import pytest
import types

from colibri.tests.conftest import (
    MOCK_CENTRAL_COVMAT_INDEX,
    MOCK_PDF_MODEL,
    MOCK_PENALTY_POSDATA,
    TEST_FK_ARRAYS,
    TEST_POS_FK_ARRAYS,
    TEST_XGRID,
)

from colibri.core import BlackJAXFit, BayesianPrior
from colibri.blackjax_fit import blackjax_fit, run_blackjax_fit
from colibri.forward_map import FKTableForwardMap
from colibri.likelihood import LogLikelihood

jax.config.update("jax_enable_x64", True)


def mock_prior_transform(x):
    return x


def mock_log_prob(x):
    return -0.5 * jnp.sum(x**2, axis=-1)


def mock_sample(rng_key, n_samples):
    n_params = len(MOCK_PDF_MODEL.param_names)
    return jax.random.normal(rng_key, shape=(n_samples, n_params))


bayesian_prior = BayesianPrior(
    prior_transform=lambda x: x,
    log_prob=lambda x: -jnp.sum(x**2, axis=-1),
    sample=lambda rng, n: jnp.zeros((n, MOCK_PDF_MODEL.n_parameters)),
)

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
def test_blackjax_fit(pos_penalty):
    forward_map = FKTableForwardMap(
        lambda pdf, fk: jnp.zeros(len(MOCK_PDF_MODEL.param_names)),
        pdf_param_names=MOCK_PDF_MODEL.param_names,
    )
    mock_log_likelihood = LogLikelihood(
        MOCK_CENTRAL_COVMAT_INDEX,
        MOCK_PDF_MODEL,
        TEST_XGRID,
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

    MOCK_PDF_MODEL.n_parameters = len(MOCK_PDF_MODEL.param_names)

    with patch("colibri.blackjax_fit.anesthetic.NestedSamples"):

        fit_result = blackjax_fit(
            MOCK_PDF_MODEL,
            bayesian_prior,
            blackjax_settings,
            mock_log_likelihood,
        )

    assert isinstance(fit_result, BlackJAXFit)


def test_blackjax_fit_truncates_posterior_and_warns(caplog):
    # --- ensure pdf_model is consistent ---
    MOCK_PDF_MODEL.n_parameters = len(MOCK_PDF_MODEL.param_names)

    bayesian_prior = BayesianPrior(
        prior_transform=lambda x: x,
        log_prob=lambda x: -jnp.sum(x**2, axis=-1),
        sample=lambda rng, n: jnp.zeros((n, MOCK_PDF_MODEL.n_parameters)),
    )

    blackjax_settings = {
        "seed": 0,
        "n_live": 4,
        "delete_fraction": 0.5,
        "repeats": 1,
        "log_precision": 1.0,  # skip while-loop
        "n_posterior_samples": 10,  # deliberately too large
        "posterior_resampling_seed": 123,
        "log_dir": "test_logs",
    }

    log_likelihood = lambda x: -jnp.sum(x**2)

    # --- minimal blackjax.nss mock ---
    fake_algo = types.SimpleNamespace(
        init=lambda particles: types.SimpleNamespace(
            logZ=0.0,
            logZ_live=0.0,
        )
    )

    with (
        patch("colibri.blackjax_fit.blackjax.nss", return_value=fake_algo),
        patch("colibri.blackjax_fit.finalise") as mock_finalise,
        patch("colibri.blackjax_fit.ess", return_value=2),
        patch("colibri.blackjax_fit.log_weights", return_value=jnp.zeros(5)),
        patch(
            "colibri.blackjax_fit.sample", return_value=jnp.ones((2, 2))
        ),  # only 2 samples
        patch("colibri.blackjax_fit.resample_from_ns_posterior") as mock_resample,
        patch("colibri.blackjax_fit.anesthetic.NestedSamples"),
    ):
        mock_finalise.return_value = types.SimpleNamespace(
            particles=jnp.ones((5, 2)),
            loglikelihood=jnp.arange(5.0),
            loglikelihood_birth=jnp.zeros(5),
        )

        caplog.set_level("WARNING")

        fit_result = blackjax_fit(
            MOCK_PDF_MODEL,
            bayesian_prior,
            blackjax_settings,
            log_likelihood,
        )

    # --- assertions ---
    assert isinstance(fit_result, BlackJAXFit)

    # Warning was emitted
    assert any(
        "exceeds the number of posterior samples computed by BlackJAX" in record.message
        for record in caplog.records
    )

    # Truncated value (2) was used
    args, _ = mock_resample.call_args
    assert args[1] == 2


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
