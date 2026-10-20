"""
colibri.nuts_fit.py

Bayesian fitting with the BlackJAX No-U-Turn sampler.
"""

import json
import logging
import time
from numbers import Integral, Real
from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from blackjax.diagnostics import effective_sample_size
from mpi4py import MPI
from scipy.stats import norm, rankdata

from colibri.core import NUTSFit
from colibri.export_results import (
    export_bayes_results,
    plot_bayes_plots,
    write_replicas,
)
from colibri.utils import resample_from_ns_posterior

log = logging.getLogger(__name__)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

_NUTS_DEFAULTS = {
    "nuts_seed": 0,
    "num_chains": 4,
    "num_warmup": 1000,
    "num_samples": 2000,
    "target_acceptance_rate": 0.8,
    "max_num_doublings": 10,
    "n_posterior_samples": 1000,
    "posterior_resampling_seed": 123456,
    "sampler_plot": True,
}


def _validate_nuts_settings(settings):
    """Validate NUTS counts, seeds and acceptance settings."""
    count_keys = (
        "num_chains",
        "num_warmup",
        "num_samples",
        "max_num_doublings",
        "n_posterior_samples",
    )
    for name in count_keys:
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    if settings["num_chains"] < 2:
        raise ValueError("num_chains must be at least two for diagnostics.")

    if settings["num_samples"] < 4:
        raise ValueError("num_samples must be at least four per chain.")

    for name in ("nuts_seed", "posterior_resampling_seed"):
        value = settings[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or not 0 <= value < 2**32
        ):
            raise ValueError(f"{name} must be an integer in [0, 2**32).")

    target = settings["target_acceptance_rate"]
    if (
        isinstance(target, bool)
        or not isinstance(target, Real)
        or not np.isfinite(target)
        or not 0 < target < 1
    ):
        raise ValueError(
            "target_acceptance_rate must be strictly between zero and one."
        )

    if not isinstance(settings["sampler_plot"], bool):
        raise ValueError("sampler_plot must be a boolean.")


def _rank_normalize(chains):
    """Rank-normalize pooled draws separately for each parameter."""
    shape = chains.shape
    flat = chains.reshape(-1, shape[-1])
    normalized = np.empty_like(flat, dtype=float)

    for index in range(shape[-1]):
        ranks = rankdata(flat[:, index])
        normalized[:, index] = norm.ppf((ranks - 0.375) / (len(ranks) + 0.25))

    return normalized.reshape(shape)


def _rhat(chains):
    """Calculate R-hat for already split chains."""
    num_draws = chains.shape[1]
    within = np.var(chains, axis=1, ddof=1).mean(axis=0)
    between = num_draws * np.var(chains.mean(axis=1), axis=0, ddof=1)
    variance = (num_draws - 1) / num_draws * within + between / num_draws

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(variance / within)


def _named_values(values, param_names):
    """Convert scalar or array diagnostics to parameter mappings."""
    values = np.atleast_1d(np.asarray(values)).reshape(-1)

    if values.size != len(param_names):
        raise ValueError(
            f"Expected {len(param_names)} diagnostic values, "
            f"received {values.size}."
        )

    return {
        name: float(value) if np.isfinite(value) else None
        for name, value in zip(param_names, values)
    }


def _nuts_diagnostics(
    positions,
    sampler_info,
    step_sizes,
    param_names,
    max_num_doublings,
):
    """
    Calculate diagnostics from complete physical parameter chains.

    Parameters
    ----------
    positions: numpy.ndarray
        Draws with shape (chains, draws, parameters).
    sampler_info: dict
        Per-transition divergence, acceptance and trajectory information.
    step_sizes: jax.Array
        Adapted step sizes.
    param_names: list
        Parameter names in forward-map order.
    max_num_doublings: int
        Maximum number of trajectory expansions.

    Returns
    -------
    dict
        JSON-compatible convergence and sampling diagnostics.
    """
    positions = np.asarray(positions)
    num_chains, num_draws, _ = positions.shape

    half = num_draws // 2
    split = np.concatenate(
        [positions[:, :half], positions[:, -half:]],
        axis=0,
    )

    normalized = _rank_normalize(split)
    folded = _rank_normalize(np.abs(split - np.median(split, axis=(0, 1))))
    rhat = np.maximum(_rhat(normalized), _rhat(folded))

    bulk_ess = np.asarray(
        effective_sample_size(
            jnp.asarray(normalized),
            chain_axis=0,
            sample_axis=1,
        )
    )

    lower, upper = np.quantile(split, [0.05, 0.95], axis=(0, 1))
    lower_ess = np.asarray(
        effective_sample_size(
            jnp.asarray(split <= lower, dtype=float),
            chain_axis=0,
            sample_axis=1,
        )
    )
    upper_ess = np.asarray(
        effective_sample_size(
            jnp.asarray(split >= upper, dtype=float),
            chain_axis=0,
            sample_axis=1,
        )
    )
    tail_ess = np.minimum(lower_ess, upper_ess)

    divergent = np.asarray(sampler_info["is_divergent"])
    acceptance = np.asarray(sampler_info["acceptance_rate"])
    integration_steps = np.asarray(sampler_info["num_integration_steps"])
    expansions = np.asarray(sampler_info["num_trajectory_expansions"])

    finite = bool(
        np.isfinite(rhat).all()
        and np.isfinite(bulk_ess).all()
        and np.isfinite(tail_ess).all()
    )
    converged = bool(
        finite
        and np.all(rhat <= 1.01)
        and np.all(bulk_ess >= 100 * num_chains)
        and np.all(tail_ess >= 100 * num_chains)
        and not divergent.any()
    )

    return {
        "algorithm": "nuts",
        "num_chains": num_chains,
        "num_samples": num_draws,
        "rhat": _named_values(rhat, param_names),
        "ess_bulk": _named_values(bulk_ess, param_names),
        "ess_tail": _named_values(tail_ess, param_names),
        "divergences": int(divergent.sum()),
        "divergence_rate": float(divergent.mean()),
        "mean_acceptance_rate": float(acceptance.mean()),
        "step_sizes": np.asarray(step_sizes).tolist(),
        "mean_integration_steps": float(integration_steps.mean()),
        "max_integration_steps": int(integration_steps.max()),
        "mean_trajectory_expansions": float(expansions.mean()),
        "treedepth_saturation": float(np.mean(expansions >= max_num_doublings)),
        "finite_convergence_diagnostics": finite,
        "converged": converged,
    }


def nuts_fit(
    forward_map,
    bayesian_prior,
    nuts_settings,
    log_likelihood,
):
    """
    Run a Bayesian fit using BlackJAX NUTS.

    Parameters
    ----------
    forward_map: ForwardMap
        Forward map defining all parameter names and their ordering.
    bayesian_prior: BayesianPrior
        Prior with unconstrained transformation functions.
    nuts_settings: dict
        Settings for initialization, warmup, sampling and replica selection.
    log_likelihood: Callable
        Deterministic differentiable likelihood with an ``ndata`` attribute.

    Returns
    -------
    NUTSFit
        Full posterior, selected replicas, metrics and diagnostics.
    """
    settings = {**_NUTS_DEFAULTS, **nuts_settings}
    _validate_nuts_settings(settings)

    required = (
        "from_unconstrained",
        "log_prob_unconstrained",
        "sample_unconstrained",
    )
    if any(not callable(getattr(bayesian_prior, name, None)) for name in required):
        raise ValueError(
            "NUTS requires an unconstrained prior implementation. "
            "Currently supported: uniform_parameter_prior."
        )

    param_names = list(forward_map.param_names)
    num_parameters = len(param_names)
    if num_parameters == 0:
        raise ValueError("NUTS requires at least one fit parameter.")

    ndata = log_likelihood.ndata
    if ndata <= 0:
        raise ValueError("The likelihood must contain at least one datapoint.")

    num_chains = settings["num_chains"]
    num_draws = settings["num_samples"]

    log.info("Running NUTS with backend: %s", jax.default_backend())
    log.info("NUTS initialization seed: %d", settings["nuts_seed"])

    @jax.jit
    def logdensity(position):
        parameters = bayesian_prior.from_unconstrained(position)
        return bayesian_prior.log_prob_unconstrained(position) + log_likelihood(
            parameters
        )

    rng_key = jax.random.PRNGKey(settings["nuts_seed"])
    init_key, warmup_key, sample_key = jax.random.split(rng_key, 3)

    initial_positions = bayesian_prior.sample_unconstrained(init_key, num_chains)
    if initial_positions.shape != (num_chains, num_parameters):
        raise ValueError("Initial prior draws do not match forward_map.param_names.")

    values, gradients = jax.vmap(jax.value_and_grad(logdensity))(initial_positions)
    if not bool(jnp.isfinite(values).all() & jnp.isfinite(gradients).all()):
        raise ValueError(
            "NUTS initialization produced a non-finite density or gradient."
        )

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity,
        is_mass_matrix_diagonal=True,
        target_acceptance_rate=settings["target_acceptance_rate"],
        max_num_doublings=settings["max_num_doublings"],
        adaptation_info_fn=lambda state, info, adaptation_state: None,
    )

    def warmup_chain(key, position):
        adapted, _ = warmup.run(
            key,
            position,
            num_steps=settings["num_warmup"],
        )
        return (
            adapted.state,
            adapted.parameters["step_size"],
            adapted.parameters["inverse_mass_matrix"],
        )

    log.info(
        "Running %d chains with %d warmup steps and %d draws per chain.",
        num_chains,
        settings["num_warmup"],
        num_draws,
    )

    started = time.perf_counter()
    states, step_sizes, inverse_mass_matrices = jax.block_until_ready(
        jax.jit(jax.vmap(warmup_chain))(
            jax.random.split(warmup_key, num_chains),
            initial_positions,
        )
    )
    warmup_seconds = time.perf_counter() - started

    if not bool(
        jnp.isfinite(step_sizes).all()
        & (step_sizes > 0).all()
        & jnp.isfinite(inverse_mass_matrices).all()
        & (inverse_mass_matrices > 0).all()
        & jnp.isfinite(states.logdensity).all()
        & jnp.isfinite(states.logdensity_grad).all()
    ):
        raise RuntimeError(
            "NUTS warmup failed to produce finite states and usable parameters."
        )

    def sample_chain(key, state, step_size, inverse_mass_matrix):
        sampler = blackjax.nuts(
            logdensity,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            max_num_doublings=settings["max_num_doublings"],
        )

        def one_step(state, step_key):
            state, info = sampler.step(step_key, state)
            history = (
                state.position,
                state.logdensity,
                info.is_divergent,
                info.acceptance_rate,
                info.num_integration_steps,
                info.num_trajectory_expansions,
            )
            return state, history

        _, history = jax.lax.scan(
            one_step,
            state,
            jax.random.split(key, num_draws),
        )
        return history

    started = time.perf_counter()
    (
        unconstrained,
        logdensity_draws,
        divergent,
        acceptance,
        integration_steps,
        expansions,
    ) = jax.block_until_ready(
        jax.jit(jax.vmap(sample_chain))(
            jax.random.split(sample_key, num_chains),
            states,
            step_sizes,
            inverse_mass_matrices,
        )
    )
    sampling_seconds = time.perf_counter() - started

    flat_unconstrained = unconstrained.reshape(-1, num_parameters)
    full_samples = jax.block_until_ready(
        jax.vmap(bayesian_prior.from_unconstrained)(flat_unconstrained)
    )

    # Remove the transformed prior, including its Jacobian, to obtain
    # physical log-likelihood values without another forward evaluation.
    log_likelihoods = jax.block_until_ready(
        logdensity_draws.reshape(-1)
        - jax.vmap(bayesian_prior.log_prob_unconstrained)(flat_unconstrained)
    )

    if not bool(
        jnp.isfinite(flat_unconstrained).all()
        & jnp.isfinite(full_samples).all()
        & jnp.isfinite(log_likelihoods).all()
        & jnp.isfinite(acceptance).all()
    ):
        raise RuntimeError("NUTS produced non-finite sampling results.")

    chain_samples = full_samples.reshape(num_chains, num_draws, num_parameters)
    sampler_info = {
        "is_divergent": divergent,
        "acceptance_rate": acceptance,
        "num_integration_steps": integration_steps,
        "num_trajectory_expansions": expansions,
    }
    diagnostics = _nuts_diagnostics(
        chain_samples,
        sampler_info,
        step_sizes,
        param_names,
        settings["max_num_doublings"],
    )
    diagnostics.update(
        {
            "warmup_seconds": warmup_seconds,
            "sampling_seconds": sampling_seconds,
        }
    )

    requested = settings["n_posterior_samples"]
    selected_count = min(requested, full_samples.shape[0])
    if selected_count < requested:
        log.warning(
            "Requested %d replicas but only %d draws are available. "
            "Using all available draws.",
            requested,
            selected_count,
        )

    resampled_posterior = resample_from_ns_posterior(
        full_samples,
        selected_count,
        settings["posterior_resampling_seed"],
    )

    best_index = jnp.argmax(log_likelihoods)
    min_chi2 = -2 * log_likelihoods[best_index]
    avg_chi2 = -2 * jnp.mean(log_likelihoods)

    log.info(
        "NUTS finished: warmup %.2f s, sampling %.2f s, divergences %d.",
        warmup_seconds,
        sampling_seconds,
        diagnostics["divergences"],
    )
    if not diagnostics["converged"]:
        log.warning(
            "NUTS convergence checks did not pass. Inspect the diagnostics "
            "before using the posterior."
        )

    return NUTSFit(
        nuts_specs=settings,
        nuts_result={
            "chain_samples": chain_samples,
            "sampler_info": sampler_info,
            "step_sizes": step_sizes,
            "inverse_mass_matrices": inverse_mass_matrices,
            "best_point": full_samples[best_index],
            "max_loglikelihood": log_likelihoods[best_index],
            "diagnostics": diagnostics,
            "logZ": None,
            "logZ_err": None,
        },
        param_names=param_names,
        resampled_posterior=resampled_posterior,
        full_posterior_samples=full_samples,
        ndata=ndata,
        bayesian_metrics={
            "bayes_complexity": avg_chi2 - min_chi2,
            "avg_chi2": avg_chi2,
            "avg_chi2_reduced": avg_chi2 / ndata,
            "min_chi2": min_chi2,
            "min_chi2_reduced": min_chi2 / ndata,
            "logz": None,
        },
    )


def _export_nuts_logs(fit, log_dir):
    """Write complete chains and transition diagnostics."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    chains = np.asarray(fit.nuts_result["chain_samples"])
    num_chains, num_draws, num_parameters = chains.shape

    samples = pd.DataFrame(
        chains.reshape(-1, num_parameters),
        columns=fit.param_names,
    )
    samples.index = pd.MultiIndex.from_product(
        [range(num_chains), range(num_draws)],
        names=["chain", "draw"],
    )
    samples.to_csv(log_dir / "nuts_samples.csv")

    transitions = pd.DataFrame(
        {
            name: np.asarray(values).reshape(-1)
            for name, values in fit.nuts_result["sampler_info"].items()
        },
        index=samples.index,
    )
    transitions.to_csv(log_dir / "nuts_transitions.csv")

    with (log_dir / "nuts_diagnostics.json").open("w") as stream:
        json.dump(
            fit.nuts_result["diagnostics"],
            stream,
            indent=2,
            allow_nan=False,
        )


def run_nuts_fit(
    nuts_fit,
    nuts_settings,
    output_path,
    pdf_model,
):
    """
    Export the results of a NUTS fit.

    Parameters
    ----------
    nuts_fit: NUTSFit
        Results produced by the NUTS provider.
    nuts_settings: dict
        Settings including the log directory and plotting option.
    output_path: pathlib.Path
        Fit output directory.
    pdf_model: PDFModel
        Model used to write PDF replicas.
    """
    if rank == 0:
        log_dir = Path(nuts_settings["log_dir"])
        _export_nuts_logs(nuts_fit, log_dir)
        export_bayes_results(nuts_fit, output_path, "nuts_result")

        if nuts_settings["sampler_plot"]:
            plot_bayes_plots(
                nuts_fit,
                output_path,
                str(log_dir.resolve()),
            )

    # All MPI ranks participate in the existing replica-writing routine.
    write_replicas(nuts_fit, output_path, pdf_model)
