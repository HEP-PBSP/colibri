"""Tests for the independent BlackJAX NUTS fitting workflow."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from reportengine.configparser import ConfigError
from scipy.stats import truncnorm

from colibri.app import colibri_providers
from colibri.bayes_prior import bayesian_prior
from colibri.config import colibriConfig
from colibri.core import BayesianPrior, NUTSFit, PriorSettings
from colibri.nuts_fit import nuts_fit, run_nuts_fit

jax.config.update("jax_enable_x64", True)


def make_prior():
    forward_map = SimpleNamespace(param_names=["parameter"])
    settings = PriorSettings(
        prior_distribution="uniform_parameter_prior",
        prior_distribution_specs={"bounds": {"parameter": [-5.0, 5.0]}},
    )
    return forward_map, bayesian_prior(settings, forward_map)


def make_settings(tmp_path, **overrides):
    return colibriConfig({}).parse_nuts_settings(
        {
            "nuts_seed": 42,
            "posterior_resampling_seed": 12,
            "num_chains": 2,
            "num_warmup": 500,
            "num_samples": 1500,
            "target_acceptance_rate": 0.9,
            "max_num_doublings": 8,
            "n_posterior_samples": 100,
            "sampler_plot": False,
            **overrides,
        },
        tmp_path,
    )


def test_nuts_provider_registered():
    assert "colibri.nuts_fit" in colibri_providers


def test_uniform_density_includes_jacobian():
    _, prior = make_prior()
    position = jnp.array([0.7])

    physical = prior.from_unconstrained(position)
    jacobian = jax.jacfwd(prior.from_unconstrained)(position)
    expected = prior.log_prob(physical) + jnp.linalg.slogdet(jacobian)[1]

    np.testing.assert_allclose(
        prior.log_prob_unconstrained(position),
        expected,
    )

    for value in (-100.0, 100.0):
        gradient = jax.grad(prior.log_prob_unconstrained)(jnp.array([value]))
        assert np.isfinite(gradient).all()


def test_uniform_transforms_follow_forward_map_order():
    forward_map = SimpleNamespace(param_names=["second", "first"])
    settings = PriorSettings(
        prior_distribution="uniform_parameter_prior",
        prior_distribution_specs={
            "bounds": {
                "first": [0, 2],
                "second": [10, 20],
            }
        },
    )
    prior = bayesian_prior(settings, forward_map)

    np.testing.assert_allclose(
        prior.from_unconstrained(jnp.zeros(2)),
        [15, 1],
    )
    starts = prior.sample_unconstrained(jax.random.PRNGKey(0), 4)
    assert starts.shape == (4, 2)
    assert np.isfinite(starts).all()


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_chains", 1),
        ("num_warmup", 0),
        ("num_samples", 3),
        ("max_num_doublings", -1),
        ("n_posterior_samples", 0),
        ("target_acceptance_rate", 1.0),
        ("nuts_seed", -1),
        ("sampler_plot", "true"),
    ],
)
def test_invalid_settings(tmp_path, key, value):
    with pytest.raises(ConfigError):
        make_settings(tmp_path, **{key: value})


def test_prior_without_unconstrained_functions_is_rejected(tmp_path):
    prior = BayesianPrior(
        prior_transform=lambda value: value,
        log_prob=lambda value: -jnp.sum(value**2),
        sample=lambda key, count: jnp.zeros((count, 1)),
    )

    with pytest.raises(ValueError, match="unconstrained prior"):
        nuts_fit(
            SimpleNamespace(param_names=["parameter"]),
            prior,
            make_settings(tmp_path),
            SimpleNamespace(ndata=1),
        )


@pytest.mark.parametrize(
    "target_mean,target_sigma",
    [(0.0, 0.7), (4.8, 0.4)],
)
def test_nuts_recovers_truncated_normal(
    tmp_path,
    target_mean,
    target_sigma,
):
    forward_map, prior = make_prior()
    settings = make_settings(tmp_path)

    class Likelihood:
        ndata = 1

        def __call__(self, parameters):
            return -0.5 * jnp.sum(((parameters - target_mean) / target_sigma) ** 2)

    likelihood = Likelihood()
    result = nuts_fit(
        forward_map,
        prior,
        settings,
        likelihood,
    )

    distribution = truncnorm(
        (-5 - target_mean) / target_sigma,
        (5 - target_mean) / target_sigma,
        loc=target_mean,
        scale=target_sigma,
    )
    samples = np.asarray(result.full_posterior_samples[:, 0])

    assert isinstance(result, NUTSFit)
    assert result.full_posterior_samples.shape == (3000, 1)
    assert result.resampled_posterior.shape == (100, 1)
    assert result.nuts_result["chain_samples"].shape == (2, 1500, 1)
    assert result.bayesian_metrics["logz"] is None

    np.testing.assert_allclose(
        samples.mean(),
        distribution.mean(),
        atol=0.12,
    )
    np.testing.assert_allclose(
        samples.var(),
        distribution.var(),
        atol=0.12,
    )
    assert ((samples >= -5) & (samples <= 5)).all()

    expected_min = -2 * np.max(
        np.asarray(jax.vmap(likelihood)(result.full_posterior_samples))
    )
    np.testing.assert_allclose(
        result.bayesian_metrics["min_chi2"],
        expected_min,
        atol=1e-10,
    )

    # Fitting itself does not write files.
    assert not (tmp_path / "nuts_logs").exists()

    with (
        patch("colibri.nuts_fit.rank", 0),
        patch("colibri.nuts_fit.write_replicas") as write_replicas,
    ):
        run_nuts_fit(
            result,
            settings,
            tmp_path,
            SimpleNamespace(),
        )

    write_replicas.assert_called_once()
    assert (tmp_path / "nuts_result.csv").exists()
    assert (tmp_path / "full_posterior_sample.csv").exists()

    stored = pd.read_csv(tmp_path / "nuts_logs" / "nuts_samples.csv")
    assert stored["chain"].nunique() == 2
    assert stored.groupby("chain").size().tolist() == [1500, 1500]

    diagnostics_path = tmp_path / "nuts_logs" / "nuts_diagnostics.json"
    with diagnostics_path.open() as stream:
        diagnostics = json.load(stream)
    assert diagnostics["algorithm"] == "nuts"
    assert "rhat" in diagnostics
    assert "ess_bulk" in diagnostics
    assert "ess_tail" in diagnostics

    metrics = pd.read_csv(
        tmp_path / "bayes_metrics.csv",
        header=None,
        names=["metric", "value"],
    ).set_index("metric")
    assert pd.isna(metrics.loc["logz", "value"])
