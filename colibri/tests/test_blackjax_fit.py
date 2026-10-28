"""
colibri.tests.test_blackjax_fit.py
Tests for the BlackJAX fitting module.
"""

from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import pytest
from types import SimpleNamespace

from colibri.blackjax_samplers._common import SamplerOutput

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
    sample=lambda rng, n: jnp.zeros((n, len(MOCK_PDF_MODEL.param_names))),
)

integrability_penalty = lambda pdf: jnp.array([0.0])

blackjax_settings = {
    "blackjax_seed": 42,
    "n_live": 50,
    "delete_fraction": 0.5,
    "repeats": 2,
    "log_precision": -1.0,
    "n_posterior_samples": 10,
    "posterior_resampling_seed": 123,
    "log_dir": "test_logs",
    "sampler_plot": False,
}


@pytest.mark.parametrize("pos_penalty", [True, False])
def test_blackjax_fit(pos_penalty):
    forward_map = FKTableForwardMap(
        lambda pdf, fk: jnp.zeros(len(MOCK_PDF_MODEL.param_names)),
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

    fit_result = blackjax_fit(
        forward_map,
        bayesian_prior,
        blackjax_settings,
        mock_log_likelihood,
    )

    assert isinstance(fit_result, BlackJAXFit)


def test_blackjax_fit_truncates_posterior_and_warns(caplog):
    # The best explored point need not be among the posterior draws.
    output = SamplerOutput(
        samples=jnp.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ),
        log_likelihoods=jnp.array([-3.0, -5.0]),
        max_loglikelihood=-1.0,
        logz=-2.0,
        logz_err=0.1,
    )

    with patch(
        "colibri.blackjax_samplers.nested_sampling.run",
        return_value=output,
    ):
        fit = blackjax_fit(
            SimpleNamespace(param_names=["a", "b"]),
            bayesian_prior,
            blackjax_settings,
            SimpleNamespace(ndata=2),
        )

    assert isinstance(fit, BlackJAXFit)
    assert fit.blackjax_specs["algorithm"] == "nested_sampling"

    assert fit.full_posterior_samples.shape == (2, 2)
    assert fit.resampled_posterior.shape == (2, 2)

    assert "exceeds the number of posterior samples" in caplog.text

    assert fit.bayesian_metrics["min_chi2"] == 2.0
    assert fit.bayesian_metrics["avg_chi2"] == 8.0
    assert fit.bayesian_metrics["min_chi2_reduced"] == 1.0
    assert fit.bayesian_metrics["avg_chi2_reduced"] == 4.0
    assert fit.bayesian_metrics["bayes_complexity"] == 6.0

    assert fit.bayesian_metrics["logz"] == -2.0
    assert fit.blackjax_result["logZ_err"] == 0.1


def make_export_fit(tmp_path, algorithm, sampler_plot=False):
    settings = {
        "algorithm": algorithm,
        "log_dir": str(tmp_path / "logs"),
        "sampler_plot": sampler_plot,
    }

    return BlackJAXFit(
        blackjax_specs=settings,
        blackjax_result={},
        param_names=["a", "b"],
        full_posterior_samples=jnp.ones((20, 2)),
        resampled_posterior=jnp.ones((5, 2)),
        ndata=2,
        bayesian_metrics={
            "min_chi2": 2.0,
            "min_chi2_reduced": 1.0,
            "avg_chi2": 4.0,
            "avg_chi2_reduced": 2.0,
            "bayes_complexity": 2.0,
            "logz": None,
        },
    )


@pytest.mark.parametrize(
    "algorithm,result_filename",
    [
        ("nested_sampling", "ns_result"),
        ("nuts", "nuts_result"),
    ],
)
@pytest.mark.parametrize("sampler_plot", [False, True])
def test_run_blackjax_fit_exports(
    tmp_path,
    algorithm,
    result_filename,
    sampler_plot,
):
    fit = make_export_fit(
        tmp_path,
        algorithm,
        sampler_plot,
    )
    log_dir = tmp_path / "logs"

    with (
        patch("colibri.blackjax_fit.rank", 0),
        patch(f"colibri.blackjax_samplers.{algorithm}.export_logs") as logs,
        patch("colibri.blackjax_fit.export_bayes_results") as export,
        patch("colibri.blackjax_fit.plot_bayes_plots") as plot,
        patch("colibri.blackjax_fit.write_replicas") as replicas,
    ):
        run_blackjax_fit(
            fit,
            fit.blackjax_specs,
            tmp_path,
            MOCK_PDF_MODEL,
        )

    logs.assert_called_once_with(fit, log_dir)

    export.assert_called_once_with(
        fit,
        tmp_path,
        result_filename,
    )

    if sampler_plot:
        plot.assert_called_once_with(
            fit,
            tmp_path,
            str(log_dir.resolve()),
        )
    else:
        plot.assert_not_called()

    replicas.assert_called_once_with(
        fit,
        tmp_path,
        MOCK_PDF_MODEL,
    )

    assert (log_dir / "posterior_samples.csv").exists()


@pytest.mark.parametrize(
    "algorithm",
    ["nested_sampling", "nuts"],
)
def test_non_root_rank_only_writes_replicas(
    tmp_path,
    algorithm,
):
    fit = make_export_fit(
        tmp_path,
        algorithm,
        sampler_plot=True,
    )

    with (
        patch("colibri.blackjax_fit.rank", 1),
        patch(f"colibri.blackjax_samplers.{algorithm}.export_logs") as logs,
        patch("colibri.blackjax_fit.export_bayes_results") as export,
        patch("colibri.blackjax_fit.plot_bayes_plots") as plot,
        patch("colibri.blackjax_fit.write_replicas") as replicas,
    ):
        run_blackjax_fit(
            fit,
            fit.blackjax_specs,
            tmp_path,
            MOCK_PDF_MODEL,
        )

    logs.assert_not_called()
    export.assert_not_called()
    plot.assert_not_called()

    replicas.assert_called_once_with(
        fit,
        tmp_path,
        MOCK_PDF_MODEL,
    )

    assert not (tmp_path / "logs").exists()
