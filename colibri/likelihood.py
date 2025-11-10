"""
colibri.likelihood.py

Module containing the likelihood class for the colibri fit.
"""

from functools import partial

import jax
import jax.numpy as jnp
from colibri.loss_functions import chi2
from colibri.commondata_utils import CentralCovmatIndex


class LogLikelihood(object):
    """
    This class takes care of constructing the log-likelihood that is passed to
    the various fit routines.
    """

    def __init__(
        self,
        central_covmat_index,
        pdf_model,
        fit_xgrid,
        forward_map,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        penalty_posdata,
        positivity_penalty_settings,
        integrability_penalty,
    ):
        """
        Parameters
        ----------
        central_covmat_index: commondata_utils.CentralCovmatIndex

        pdf_model: pdf_model.PDFModel

        fit_xgrid: np.ndarray

        forward_map: Callable

        fast_kernel_arrays: tuple

        positivity_fast_kernel_arrays: tuple

        penalty_posdata: Callable

        positivity_penalty_settings: dict, default {}

        integrability_penalty: Callable

        """
        self.central_values = central_covmat_index.central_values
        self.covmat = central_covmat_index.covmat
        self.inv_covmat = jnp.linalg.inv(self.covmat)
        self.central_values_idx = central_covmat_index.central_values_idx
        self.pdf_model = pdf_model
        self.penalty_posdata = penalty_posdata
        self.positivity_penalty_settings = positivity_penalty_settings
        self.integrability_penalty = integrability_penalty

        self.pred_and_pdf = pdf_model.pred_and_pdf_func(
            fit_xgrid, forward_map=forward_map
        )

        self.fast_kernel_arrays = fast_kernel_arrays
        self.positivity_fast_kernel_arrays = positivity_fast_kernel_arrays

    def __call__(self, params, batch_idx=None):
        """
        Note that this function is called by the samplers, and it must be
        a function of the model parameters only.

        Parameters
        ----------
        params: jnp.ndarray
            The model parameters.

        batch_idx: jnp.ndarray, optional
            If provided, computes the log-likelihood only for the subset of data
            indexed by batch_idx.

        Returns
        -------
        jnp.ndarray
            The log-likelihood value.
        """
        return self.log_likelihood(
            params,
            self.central_values,
            self.inv_covmat,
            self.fast_kernel_arrays,
            self.positivity_fast_kernel_arrays,
            batch_idx=batch_idx,
        )

    @partial(jax.jit, static_argnames=("self",))
    def log_likelihood(
        self,
        params: jnp.ndarray,
        central_values: jnp.ndarray,
        inv_covmat: jnp.ndarray,
        fast_kernel_arrays: tuple,
        positivity_fast_kernel_arrays: tuple,
        batch_idx: jnp.ndarray = None,
    ) -> jnp.array:
        """
        This function takes care of computing the log_likelihood that is defined in LogLikelihood.
        Function is jax.jit compiled for better performance.

        Parameters
        ----------
        params: jnp.ndarray
        central_values: jnp.ndarray
        inv_covmat: jnp.ndarray
        fast_kernel_arrays: tuple
        positivity_fast_kernel_arrays: tuple

        Returns
        -------
        jnp.ndarray
            jax array with the value of the log-likelihood.
        """
        predictions, pdf = self.pred_and_pdf(params, fast_kernel_arrays)
        # Select only the data relevant for this likelihood
        # Especially important when using a training/validation split
        predictions = predictions[self.central_values_idx]

        if batch_idx is not None:
            predictions = predictions[batch_idx]
            central_values = central_values[batch_idx]
            batched_covmat = self.covmat[batch_idx][:, batch_idx]
            inv_covmat = jnp.linalg.inv(batched_covmat)

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
            chi2(central_values, predictions, inv_covmat) + pos_penalty + integ_penalty
        )


def log_likelihood(
    central_covmat_index,
    pdf_model,
    FIT_XGRID,
    _pred_data,
    fast_kernel_arrays,
    positivity_fast_kernel_arrays,
    _penalty_posdata,
    positivity_penalty_settings,
    integrability_penalty,
):
    """
    Instantiates the LogLikelihood class.
    This function is used to create the log likelihood function for the sampler.
    The function, being a node of the reportengine graph, can be overriden by the user for
    model specific applications by changing the log_likelihood method of the LogLikelihood class.
    """
    return LogLikelihood(
        central_covmat_index,
        pdf_model,
        FIT_XGRID,
        _pred_data,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        _penalty_posdata,
        positivity_penalty_settings,
        integrability_penalty,
    )


def mc_log_likelihood(
    mc_pseudodata,
    fit_covariance_matrix,
    pdf_model,
    FIT_XGRID,
    _pred_data,
    fast_kernel_arrays,
    positivity_fast_kernel_arrays,
    _penalty_posdata,
    positivity_penalty_settings,
    integrability_penalty,
):
    """
    Instantiates the LogLikelihood class for training and validation datasets
    when using Monte Carlo pseudodata with a training/validation split.
    The function, being a node of the reportengine graph, can be overriden by the user for
    model specific applications by changing the log_likelihood method of the LogLikelihood class.
    """

    tr_idx = mc_pseudodata.training_indices
    central_values_train = mc_pseudodata.pseudodata[tr_idx]
    covmat_train = fit_covariance_matrix[tr_idx][:, tr_idx]

    central_covmat_index_train = CentralCovmatIndex(
        central_values=central_values_train,
        covmat=covmat_train,
        central_values_idx=tr_idx,
    )

    train_loglike = LogLikelihood(
        central_covmat_index_train,
        pdf_model,
        FIT_XGRID,
        _pred_data,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        _penalty_posdata,
        positivity_penalty_settings,
        integrability_penalty,
    )

    if not mc_pseudodata.trval_split:
        val_loglike = lambda params: jnp.nan

    else:
        val_idx = mc_pseudodata.validation_indices
        central_values_val = mc_pseudodata.pseudodata[val_idx]
        covmat_val = fit_covariance_matrix[val_idx][:, val_idx]

        central_covmat_index_val = CentralCovmatIndex(
            central_values=central_values_val,
            covmat=covmat_val,
            central_values_idx=val_idx,
        )

        val_loglike = LogLikelihood(
            central_covmat_index_val,
            pdf_model,
            FIT_XGRID,
            _pred_data,
            fast_kernel_arrays,
            positivity_fast_kernel_arrays,
            _penalty_posdata,
            positivity_penalty_settings,
            integrability_penalty,
        )

    return train_loglike, val_loglike
