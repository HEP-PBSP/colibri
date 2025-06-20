"""
colibri.ultranest_fit.py

This module contains the main Bayesian fitting routine of colibri.

"""

import logging
import sys
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import ultranest
import ultranest.popstepsampler as popstepsampler
import ultranest.stepsampler as ustepsampler
from mpi4py import MPI

from colibri.export_results import BayesianFit, export_bayes_results, write_replicas
from colibri.utils import resample_from_ns_posterior

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

log = logging.getLogger(__name__)

# Check if --debug flag is present
debug_flag = "--debug" in sys.argv

# Set the Ultranest logging level based on the presence of --debug flag
ultranest_logger = logging.getLogger("ultranest")
ultranest_logger.setLevel(logging.DEBUG if debug_flag else logging.INFO)

# Configure the handler and formatter
handler = logging.StreamHandler(sys.stdout)
ultranest_logger.addHandler(handler)


@dataclass(frozen=True)
class UltranestFit(BayesianFit):
    """
    Dataclass containing the results and specs of an Ultranest fit.

    Attributes
    ----------
    ultranest_specs: dict
        Dictionary containing the settings of the Ultranest fit.
    ultranest_result: dict
        result from ultranest, can be used eg for corner plots
    """

    ultranest_specs: dict
    ultranest_result: dict


def ultranest_fit(
    pdf_model,
    bayesian_prior,
    ns_settings,
    log_likelihood,
):
    """
    The complete Nested Sampling fitting routine, for any PDF model.

    Parameters
    ----------
    pdf_model: pdf_model.PDFModel
        The PDF model to fit.

    bayesian_prior: @jax.jit CompiledFunction
        The prior function for the model.

    ns_settings: dict
        Settings for the Nested Sampling fit.

    log_likelihood: Callable
        The log likelihood function for the model.

    Returns
    -------
    UltranestFit
        Dataclass containing the results and specs of an Ultranest fit.
    """

    log.info(f"Running fit with backend: {jax.lib.xla_bridge.get_backend().platform}")

    # set the ultranest seed
    np.random.seed(ns_settings["ultranest_seed"])

    parameters = pdf_model.param_names

    sampler = ultranest.ReactiveNestedSampler(
        parameters,
        log_likelihood,
        bayesian_prior["prior_transform"],
        **ns_settings["ReactiveNS_settings"],
    )

    if ns_settings["SliceSampler_settings"]:
        if ns_settings["popstepsampler"]:

            sampler.stepsampler = popstepsampler.PopulationSliceSampler(
                generate_direction=ultranest.popstepsampler.generate_mixture_random_direction,
                **ns_settings["SliceSampler_settings"],
            )
        else:

            sampler.stepsampler = ustepsampler.SliceSampler(
                generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
                **ns_settings["SliceSampler_settings"],
            )

    t0 = time.time()
    ultranest_result = sampler.run(**ns_settings["Run_settings"])
    t1 = time.time()

    if rank == 0:
        log.info("ULTRANEST RUNNING TIME: %f" % (t1 - t0))

    n_posterior_samples = ns_settings["n_posterior_samples"]

    # Initialize fit_result to avoid UnboundLocalError
    fit_result = None

    # The following block is only executed by the master process
    if rank == 0:
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

        if ns_settings["sampler_plot"]:
            log.info("Plotting sampler plots")
            # Store run plots to ultranest_logs folder (within output_path folder)
            sampler.plot()

        # Get the full samples
        full_samples = ultranest_result["samples"]

        # Compute bayesian metrics
        min_chi2 = -2 * ultranest_result["maximum_likelihood"]["logl"]

        # the log_likelihood function here should never be vectorized as the samples do not come in batches
        if ns_settings["ReactiveNS_settings"]["vectorized"]:
            avg_chi2 = jnp.array([-2 * log_likelihood(full_samples)]).mean()
        else:
            avg_chi2 = jnp.array(
                [
                    -2 * log_likelihood(jnp.array(sample)).item()
                    for sample in full_samples
                ]
            ).mean()
        Cb = avg_chi2 - min_chi2

        fit_result = UltranestFit(
            ultranest_specs=ns_settings,
            ultranest_result=ultranest_result,
            param_names=parameters,
            resampled_posterior=resampled_posterior,
            full_posterior_samples=full_samples,
            bayesian_metrics={
                "bayes_complexity": Cb,
                "avg_chi2": avg_chi2,
                "min_chi2": min_chi2,
                "logz": ultranest_result["logz"],
            },
        )

    # Synchronize to ensure all processes have finished
    comm.Barrier()

    # Broadcast the result to all processes
    fit_result = comm.bcast(fit_result, root=0)

    return fit_result


def run_ultranest_fit(ultranest_fit, output_path, pdf_model):
    """
    Export the results of an Ultranest fit.

    Parameters
    ----------
    ultranest_fit: UltranestFit
        The results of the Ultranest fit.
    output_path: pathlib.PosixPath
        Path to the output folder.
    pdf_model: pdf_model.PDFModel
        The PDF model used in the fit.
    """

    if rank == 0:
        export_bayes_results(ultranest_fit, output_path, "ns_result")

    write_replicas(ultranest_fit, output_path, pdf_model)
