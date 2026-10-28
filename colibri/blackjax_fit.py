"""Shared interface for BlackJAX Bayesian fits."""

import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd
from mpi4py import MPI

from colibri.blackjax_samplers import get_sampler
from colibri.blackjax_samplers._common import (
    COMMON_SETTINGS,
    validate_common_settings,
)
from colibri.core import BlackJAXFit
from colibri.export_results import (
    export_bayes_results,
    plot_bayes_plots,
    write_replicas,
)
from colibri.utils import resample_from_ns_posterior

log = logging.getLogger(__name__)
rank = MPI.COMM_WORLD.Get_rank()


def blackjax_fit(
    forward_map,
    bayesian_prior,
    blackjax_settings,
    log_likelihood,
):
    """Run the selected sampler and construct a common fit result."""
    algorithm = blackjax_settings.get(
        "algorithm",
        "nested_sampling",
    )
    sampler = get_sampler(algorithm)

    # Defaults also support direct calls to this provider.
    settings = {
        **COMMON_SETTINGS,
        **sampler.SETTINGS,
        **blackjax_settings,
        "algorithm": algorithm,
    }

    validate_common_settings(settings)
    sampler.validate(settings)

    param_names = list(forward_map.param_names)
    ndata = log_likelihood.ndata

    if not param_names:
        raise ValueError("BlackJAX requires at least one fit parameter.")

    if ndata <= 0:
        raise ValueError("The likelihood must contain at least one datapoint.")

    log.info(
        "Running BlackJAX %s with backend: %s",
        algorithm,
        jax.default_backend(),
    )
    log.info(
        "BlackJAX initialization seed: %d",
        settings["blackjax_seed"],
    )

    output = sampler.run(
        jax.random.PRNGKey(settings["blackjax_seed"]),
        bayesian_prior,
        log_likelihood,
        settings,
        param_names=param_names,
    )

    samples = jnp.asarray(output.samples)
    log_likelihoods = jnp.asarray(output.log_likelihoods)
    max_loglikelihood = jnp.asarray(output.max_loglikelihood)

    if samples.ndim != 2 or samples.shape[1] != len(param_names) or len(samples) == 0:
        raise RuntimeError("Sampler returned invalid posterior sample dimensions.")

    if log_likelihoods.shape != (len(samples),):
        raise RuntimeError("Sampler log likelihoods do not match posterior samples.")

    if max_loglikelihood.shape != ():
        raise RuntimeError("Sampler maximum log likelihood must be a scalar.")

    if not bool(
        jnp.isfinite(samples).all()
        & jnp.isfinite(log_likelihoods).all()
        & jnp.isfinite(max_loglikelihood)
    ):
        raise RuntimeError("Sampler returned non-finite posterior results.")

    requested = settings["n_posterior_samples"]
    selected_count = min(
        requested,
        len(samples),
    )

    if selected_count < requested:
        log.warning(
            "The chosen number of posterior samples exceeds the "
            "number of posterior samples computed by BlackJAX. "
            "Setting the number of resampled posterior samples to %d",
            selected_count,
        )

    if output.resampled_posterior is None:
        selected = resample_from_ns_posterior(
            samples,
            selected_count,
            settings["posterior_resampling_seed"],
        )
    else:
        selected = jnp.asarray(output.resampled_posterior)
        if selected.shape != (selected_count, len(param_names)):
            raise RuntimeError("Sampler-selected replicas have invalid dimensions.")
        if not bool(jnp.isfinite(selected).all()):
            raise RuntimeError("Sampler-selected replicas are non-finite.")

    min_chi2 = -2 * max_loglikelihood
    avg_chi2 = -2 * jnp.mean(log_likelihoods)

    return BlackJAXFit(
        blackjax_specs=settings,
        blackjax_result={
            **output.result,
            "algorithm": algorithm,
            "max_loglikelihood": max_loglikelihood,
            "logZ": output.logz,
            "logZ_err": output.logz_err,
        },
        param_names=param_names,
        full_posterior_samples=samples,
        resampled_posterior=selected,
        ndata=ndata,
        bayesian_metrics={
            "min_chi2": min_chi2,
            "min_chi2_reduced": min_chi2 / ndata,
            "avg_chi2": avg_chi2,
            "avg_chi2_reduced": avg_chi2 / ndata,
            "bayes_complexity": avg_chi2 - min_chi2,
            "logz": output.logz,
        },
    )


def run_blackjax_fit(
    blackjax_fit,
    blackjax_settings,
    output_path,
    pdf_model,
):
    """Export results on rank zero and write replicas across ranks."""
    fit = blackjax_fit
    settings = fit.blackjax_specs

    sampler = get_sampler(settings["algorithm"])

    if rank == 0:
        log_dir = Path(settings["log_dir"])
        log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        sampler.export_logs(
            fit,
            log_dir,
        )

        pd.DataFrame(
            fit.resampled_posterior,
            columns=fit.param_names,
        ).to_csv(
            log_dir / "posterior_samples.csv",
            index=False,
        )

        export_bayes_results(
            fit,
            output_path,
            sampler.RESULT_FILENAME,
        )

        if settings["sampler_plot"]:
            plot_bayes_plots(
                fit,
                output_path,
                str(log_dir.resolve()),
            )

    # All ranks participate in the existing replica-writing routine.
    write_replicas(
        fit,
        output_path,
        pdf_model,
    )
