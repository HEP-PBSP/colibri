"""
colibri.ultranest_fit.py

This module contains the main Bayesian fitting routine of colibri.

"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import pandas as pd
import ultranest
import time
import logging
import sys

from colibri.constants import XGRID
from colibri.lhapdf import write_exportgrid
from colibri.utils import resample_from_ns_posterior

log = logging.getLogger(__name__)

# Check if --debug flag is present
debug_flag = "--debug" in sys.argv

# Set the Ultranest logging level based on the presence of --debug flag
ultranest_logger = logging.getLogger("ultranest")
ultranest_logger.setLevel(logging.DEBUG if debug_flag else logging.WARNING)

# Configure the handler and formatter
handler = logging.StreamHandler(sys.stdout)
ultranest_logger.addHandler(handler)


@dataclass(frozen=True)
class UltranestFit:
    """
    Dataclass containing the results and specs of an Ultranest fit.

    Attributes
    ----------
    ultranest_specs: dict
        Dictionary containing the settings of the Ultranest fit.
    resampled_posterior: jnp.array
        Array containing the resampled posterior samples.
    ultranest_result: dict
        result from ultranest, can be used eg for corner plots
    """

    ultranest_specs: dict
    resampled_posterior: jnp.array
    ultranest_result: dict


def ultranest_fit(
    _chi2_with_positivity,
    pdf_model,
    bayesian_prior,
    ns_settings,
    output_path,
):
    """
    The complete Nested Sampling fitting routine, for any PDF model.

    Parameters
    ----------
    _chi2_with_positivity: @jax.jit CompiledFunction
        The chi2 function with positivity constraint.

    pdf_model: pdf_model.PDFModel
        The PDF model to fit.

    bayesian_prior: @jax.jit CompiledFunction
        The prior function for the model.

    ns_settings: dict
        Settings for the Nested Sampling fit.

    output_path: str
        Path to write the results to.

    Returns
    -------
    UltranestFit
        Dataclass containing the results and specs of an Ultranest fit.
    """

    parameters = pdf_model.param_names

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

    # Store run plots to ultranest_logs folder (within output_path folder)
    sampler.plot()

    df = pd.DataFrame(resampled_posterior, columns=parameters)
    df.to_csv(str(output_path) + "/ns_result.csv")

    # Finish by writing the replicas to export grids, ready for evolution
    for i in range(n_posterior_samples):
        log.info(f"Writing exportgrid for replica {i+1}")
        write_exportgrid(
            jnp.array(df.iloc[i, :].tolist()), pdf_model, i + 1, output_path
        )

    return UltranestFit(
        ultranest_specs=ns_settings,
        resampled_posterior=resampled_posterior,
        ultranest_result=ultranest_result,
    )
