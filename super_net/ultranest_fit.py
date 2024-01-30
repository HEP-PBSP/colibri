"""
super_net.ultranest_fit.py

This module contains the main Bayesian fitting routine of super_net.

"""

import jax
import jax.numpy as jnp
import pandas as pd
import ultranest
import time
import logging

from super_net.constants import XGRID
from super_net.lhapdf import write_exportgrid
from super_net.utils import resample_from_ns_posterior

log = logging.getLogger(__name__)


def ultranest_fit(
    _chi2_with_positivity,
    pdf_model,
    bayesian_prior,
    ns_settings,
    output_path,
):
    """The complete Nested Sampling fitting routine, for any PDF model."""

    parameters = pdf_model.param_names
    log_dir = output_path / "ultranest"

    fit_grid_values_func = pdf_model.grid_values_func(XGRID)

    @jax.jit
    def log_likelihood(params):
        pdf = fit_grid_values_func(params)
        return -0.5 * _chi2_with_positivity(pdf)

    sampler = ultranest.ReactiveNestedSampler(
        parameters,
        log_likelihood,
        bayesian_prior,
        **ns_settings["ReactiveNS_settings"],
    )

    if ns_settings["SliceSampler_settings"]:
        import ultranest.stepsampler as ustepsampler

        sampler.stepsampler = ustepsampler.SliceSampler(
            generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
            **ns_settings["SliceSampler_settings"],
        )

    t0 = time.time()
    ultranest_result = sampler.run(**ns_settings["Run_settings"])
    t1 = time.time()
    log.info("ULTRANEST RUNNING TIME: %f" % (t1 - t0))

    n_posterior_samples = ns_settings["n_posterior_samples"]
    if n_posterior_samples > ultranest_result["samples"].shape[0]:
        n_posterior_samples = ultranest_result["samples"].shape[0]
        log.warning(
            f"The chosen number of posterior samples exceeds the number of posterior"
            "samples computed by ultranest. Setting the number of resampled posterior"
            f"samples to {n_posterior_samples}"
        )

    resampled_posterior = resample_from_ns_posterior(
        ultranest_result["samples"],
        n_posterior_samples,
        ns_settings["posterior_resampling_seed"],
    )

    # Store run plots to ultranest output folder
    sampler.plot()

    df = pd.DataFrame(resampled_posterior, columns=parameters)
    df.to_csv(str(output_path) + "/ns_result.csv")

    # Finish by writing the replicas to export grids, ready for evolution
    for i in range(n_posterior_samples):
        log.info(f"Writing exportgrid for replica {i+1}")
        write_exportgrid(
            jnp.array(df.iloc[i, :].tolist()), pdf_model, i + 1, output_path
        )
