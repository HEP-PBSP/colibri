"""Tests for the independent BlackJAX NUTS fitting workflow."""

import json
from dataclasses import replace
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
from colibri.nuts_fit import (
    _NUTS_DEFAULTS,
    _named_values,
    _thin_chains,
    _validate_nuts_settings,
    nuts_fit,
    run_nuts_fit,
)

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


@pytest.mark.parametrize(
    "requested,expected_indices",
    [
        (1, [0]),
        (2, [0, 10]),
        (7, [0, 5, 10, 15, 20, 25, 30]),
        (8, [0, 5, 10, 15, 20, 25, 30, 35]),
        (13, [0, 2, 5, 7, 10, 13, 16, 20, 23, 26, 30, 33, 36]),
        (40, list(range(40))),
        (41, list(range(40))),
    ],
)
def test_thin_chains_selects_spaced_replicas(requested, expected_indices):
    # Unique rows reveal both the source chain and draw for every selection.
    chains = jnp.arange(80).reshape(4, 10, 2)
    result = _thin_chains(chains, requested)
    expected = np.asarray(chains).reshape(40, 2)[expected_indices]

    assert result.shape == (min(requested, 40), 2)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(_thin_chains(chains, requested), result)


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
    np.testing.assert_array_equal(
        result.resampled_posterior,
        result.nuts_result["chain_samples"][:, ::30, :].reshape(100, 1),
    )
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


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("num_chains", 1, "at least two"),
        ("num_samples", 3, "at least four"),
        ("num_warmup", 0, "positive integer"),
        ("num_warmup", True, "positive integer"),
        ("num_warmup", 1.5, "positive integer"),
        ("nuts_seed", -1, "nuts_seed must be an integer"),
        ("nuts_seed", 2**32, "nuts_seed must be an integer"),
        ("nuts_seed", True, "nuts_seed must be an integer"),
        ("nuts_seed", 1.5, "nuts_seed must be an integer"),
        ("target_acceptance_rate", 1.0, "strictly between"),
        ("target_acceptance_rate", np.nan, "strictly between"),
        ("target_acceptance_rate", True, "strictly between"),
        ("target_acceptance_rate", "0.8", "strictly between"),
        ("sampler_plot", "true", "sampler_plot must be a boolean"),
    ],
)
def test_provider_rejects_invalid_settings(key, value, message):
    # Exercise provider validation directly, without the config parser rejecting first.
    with pytest.raises(ValueError, match=message):
        _validate_nuts_settings({**_NUTS_DEFAULTS, key: value})


def test_named_values_rejects_wrong_parameter_count():
    with pytest.raises(ValueError, match=r"Expected 2 diagnostic values, received 1\."):
        _named_values([1.0], ["first", "second"])


def test_named_values_handles_scalar_and_nonfinite_diagnostics():
    assert _named_values(1.25, ["parameter"]) == {"parameter": 1.25}
    assert _named_values([np.nan, np.inf], ["first", "second"]) == {
        "first": None,
        "second": None,
    }


@pytest.fixture
def small_nuts_inputs(tmp_path):
    forward_map, prior = make_prior()

    class Likelihood:
        ndata = 1

        def __call__(self, parameters):
            return -0.5 * jnp.sum(parameters**2)

    settings = make_settings(tmp_path, num_warmup=20, num_samples=4)
    return forward_map, prior, settings, Likelihood()


@pytest.mark.parametrize(
    "invalid_input,message",
    [
        ("parameters", "at least one fit parameter"),
        ("ndata", "at least one datapoint"),
        ("shape", "Initial prior draws do not match"),
        ("density", "non-finite density or gradient"),
        ("gradient", "non-finite density or gradient"),
    ],
)
def test_nuts_rejects_invalid_initialization(small_nuts_inputs, invalid_input, message):
    forward_map, prior, settings, likelihood = small_nuts_inputs
    if invalid_input == "parameters":
        forward_map = SimpleNamespace(param_names=[])
    elif invalid_input == "ndata":
        likelihood.ndata = 0
    elif invalid_input == "shape":
        prior = replace(
            prior, sample_unconstrained=lambda key, count: jnp.zeros((count, 2))
        )
    elif invalid_input == "density":
        prior = replace(prior, log_prob_unconstrained=lambda position: -jnp.inf)
    else:
        # Finite density at zero, but its gradient is undefined.
        prior = replace(
            prior,
            sample_unconstrained=lambda key, count: jnp.zeros((count, 1)),
            log_prob_unconstrained=lambda position: jnp.sqrt(jnp.sum(position**2)),
        )

    with pytest.raises(ValueError, match=message):
        nuts_fit(forward_map, prior, settings, likelihood)


@pytest.mark.parametrize(
    "phase,field",
    [
        ("warmup", "step_size_nonfinite"),
        ("warmup", "step_size_nonpositive"),
        ("warmup", "mass_nonfinite"),
        ("warmup", "mass_nonpositive"),
        ("warmup", "logdensity"),
        ("warmup", "logdensity_grad"),
        ("sampling", "positions"),
        ("sampling", "logdensity"),
        ("sampling", "acceptance"),
        ("sampling", "physical_samples"),
    ],
)
def test_nuts_rejects_failed_backend_results(small_nuts_inputs, phase, field):
    block_until_ready = jax.block_until_ready
    calls = 0

    def failed_result(result):
        nonlocal calls
        result = block_until_ready(result)
        calls += 1
        if phase == "warmup" and calls == 1:
            states, step_sizes, inverse_mass_matrices = result
            if field.startswith("step_size"):
                value = jnp.nan if field.endswith("nonfinite") else 0.0
                step_sizes = jnp.full_like(step_sizes, value)
            elif field.startswith("mass"):
                value = jnp.nan if field.endswith("nonfinite") else 0.0
                inverse_mass_matrices = jnp.full_like(inverse_mass_matrices, value)
            else:
                states = states._replace(
                    **{field: jnp.full_like(getattr(states, field), jnp.nan)}
                )
            return states, step_sizes, inverse_mass_matrices
        if phase == "sampling" and calls == 2:
            indices = {"positions": 0, "logdensity": 1, "acceptance": 3}
            if field in indices:
                index = indices[field]
                result = (
                    *result[:index],
                    jnp.full_like(result[index], jnp.nan),
                    *result[index + 1 :],
                )
        if phase == "sampling" and field == "physical_samples" and calls == 3:
            result = jnp.full_like(result, jnp.nan)
        return result

    message = (
        "NUTS warmup failed to produce finite states and usable parameters"
        if phase == "warmup"
        else "NUTS produced non-finite sampling results"
    )
    with (
        patch("colibri.nuts_fit.jax.block_until_ready", side_effect=failed_result),
        pytest.raises(RuntimeError, match=message),
    ):
        nuts_fit(*small_nuts_inputs)


def test_nuts_warns_for_insufficient_draws_and_nonconvergence(
    small_nuts_inputs, caplog
):
    forward_map, prior, settings, likelihood = small_nuts_inputs
    settings = {**settings, "n_posterior_samples": 9}

    with caplog.at_level("WARNING", logger="colibri.nuts_fit"):
        result = nuts_fit(forward_map, prior, settings, likelihood)

    assert "Requested 9 replicas but only 8 draws are available" in caplog.text
    assert "NUTS convergence checks did not pass" in caplog.text
    assert result.nuts_result["diagnostics"]["converged"] is False
    np.testing.assert_array_equal(
        result.resampled_posterior, result.full_posterior_samples
    )


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("sampler_plot", [False, True])
def test_run_nuts_fit_exports_on_primary_rank(tmp_path, rank, sampler_plot):
    fit, model = SimpleNamespace(), SimpleNamespace()
    log_dir = tmp_path / "nuts_logs"
    settings = {"log_dir": log_dir, "sampler_plot": sampler_plot}

    with (
        patch("colibri.nuts_fit.rank", rank),
        patch("colibri.nuts_fit._export_nuts_logs") as export_logs,
        patch("colibri.nuts_fit.export_bayes_results") as export_results,
        patch("colibri.nuts_fit.plot_bayes_plots") as plot,
        patch("colibri.nuts_fit.write_replicas") as write_replicas,
    ):
        run_nuts_fit(fit, settings, tmp_path, model)

    if rank == 0:
        export_logs.assert_called_once_with(fit, log_dir)
        export_results.assert_called_once_with(fit, tmp_path, "nuts_result")
    else:
        export_logs.assert_not_called()
        export_results.assert_not_called()
    if rank == 0 and sampler_plot:
        plot.assert_called_once_with(fit, tmp_path, str(log_dir.resolve()))
    else:
        plot.assert_not_called()
    write_replicas.assert_called_once_with(fit, tmp_path, model)
