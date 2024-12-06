"""
colibri.ultranest_fit.py

This module contains the main Bayesian fitting routine of colibri.

"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import ultranest
import ultranest.popstepsampler as popstepsampler
import ultranest.stepsampler as ustepsampler
import time
import logging
import sys
from functools import partial

from colibri.utils import resample_from_ns_posterior
from colibri.export_results import BayesianFit, write_replicas, export_bayes_results
from colibri.loss_functions import chi2

import numpy as np
from mpi4py import MPI

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


class UltraNestLogLikelihood(object):
    def __init__(
        self,
        central_inv_covmat_index,
        pdf_model,
        fit_xgrid,
        forward_map,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        ns_settings,
        chi2,
        penalty_posdata,
        positivity_penalty_settings,
        integrability_penalty,
    ):
        """
        Parameters
        ----------
        central_inv_covmat_index: commondata_utils.CentralInvCovmatIndex

        pdf_model: pdf_model.PDFModel

        fit_xgrid: np.ndarray

        forward_map: Callable

        fast_kernel_arrays: tuple

        positivity_fast_kernel_arrays: tuple

        ns_settings: dict

        chi2: Callable

        penalty_posdata: Callable

        positivity_penalty_settings: dict, default {}

        integrability_penalty: Callable

        """
        self.central_values = central_inv_covmat_index.central_values
        self.inv_covmat = central_inv_covmat_index.inv_covmat
        self.pdf_model = pdf_model
        self.chi2 = chi2
        self.penalty_posdata = penalty_posdata
        self.positivity_penalty_settings = positivity_penalty_settings
        self.integrability_penalty = integrability_penalty

        self.pred_and_pdf = pdf_model.pred_and_pdf_func(
            fit_xgrid, forward_map=forward_map
        )

        if ns_settings["ReactiveNS_settings"]["vectorized"]:
            self.pred_and_pdf = jax.vmap(
                self.pred_and_pdf, in_axes=(0, None), out_axes=(0, 0)
            )

            self.chi2 = jax.vmap(self.chi2, in_axes=(None, 0, None), out_axes=0)
            self.penalty_posdata = jax.vmap(
                self.penalty_posdata, in_axes=(0, None, None, None), out_axes=0
            )
            self.integrability_penalty = jax.vmap(
                self.integrability_penalty, in_axes=(0,), out_axes=0
            )

        self.fast_kernel_arrays = fast_kernel_arrays
        self.positivity_fast_kernel_arrays = positivity_fast_kernel_arrays

    def __call__(self, params):
        """
        Note that this function is called by the ultranest sampler, and it must be
        a function of the model parameters only.

        Parameters
        ----------
        params: jnp.array
            The model parameters.
        """
        return self.log_likelihood(
            params,
            self.central_values,
            self.inv_covmat,
            self.fast_kernel_arrays,
            self.positivity_fast_kernel_arrays,
        )

    @partial(jax.jit, static_argnames=("self",))
    def log_likelihood(
        self,
        params,
        central_values,
        inv_covmat,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
    ):
        predictions, pdf = self.pred_and_pdf(params, fast_kernel_arrays)

        if self.positivity_penalty_settings["positivity_penalty"]:
            pos_penalty = jnp.sum(
                self.penalty_posdata(
                    pdf,
                    self.positivity_penalty_settings["alpha"],
                    self.positivity_penalty_settings["lambda_positivity"],
                    positivity_fast_kernel_arrays,
                ),
                axis=-1,
            )
        else:
            pos_penalty = 0

        integ_penalty = jnp.sum(
            self.integrability_penalty(
                pdf,
            ),
            axis=-1,
        )

        return -0.5 * (
            self.chi2(central_values, predictions, inv_covmat)
            + pos_penalty
            + integ_penalty
        )


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


def log_likelihood(
    central_inv_covmat_index,
    pdf_model,
    FIT_XGRID,
    _pred_data,
    fast_kernel_arrays,
    positivity_fast_kernel_arrays,
    ns_settings,
    _penalty_posdata,
    positivity_penalty_settings,
    integrability_penalty,
):
    """
    Instantiates the UltraNestLogLikelihood class.
    This function is used to create the log likelihood function for the UltraNest sampler.
    The function, being a node of the reportengine graph, can be overriden by the user for
    model specific applications by changing the log_likelihood method of the UltraNestLogLikelihood class.
    """
    return UltraNestLogLikelihood(
        central_inv_covmat_index,
        pdf_model,
        FIT_XGRID,
        _pred_data,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        ns_settings,
        chi2,
        _penalty_posdata,
        positivity_penalty_settings,
        integrability_penalty,
    )


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
        bayesian_prior,
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
