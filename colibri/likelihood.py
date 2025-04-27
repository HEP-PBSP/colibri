"""
colibri.likelihood.py

Module containing the likelihood class for the colibri fit.
"""

import logging
import sys
from functools import partial

import jax
import jax.numpy as jnp
from mpi4py import MPI

from colibri.loss_functions import chi2

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


class LogLikelihood(object):
    """
    TODO: describe class.
    """

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
        TODO: Instantiation of the ...

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

        # TODO: is ultranest specific and should be changed at some point
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
        Note that this function is called by the samplers, and it must be
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
    Instantiates the LogLikelihood class.
    This function is used to create the log likelihood function for the UltraNest sampler.
    The function, being a node of the reportengine graph, can be overriden by the user for
    model specific applications by changing the log_likelihood method of the UltraNestLogLikelihood class.
    """
    return LogLikelihood(
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
