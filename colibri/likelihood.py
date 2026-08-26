"""
colibri.likelihood.py

Module containing the likelihood class for the colibri fit.
"""

from functools import partial

import jax
import jax.numpy as jnp
from colibri.loss_functions import chi2
from colibri.commondata_utils import CentralCovmatIndex
from colibri.data_batch import BatchSpec

THRESHOLD_POS = 1e-6


class LogLikelihood(object):
    """
    This class takes care of constructing the log-likelihood that is passed to
    the various fit routines.
    """

    def __init__(
        self,
        central_covmat_index,
        pdf_model,
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

        self.forward_map = forward_map

        self.fast_kernel_arrays = fast_kernel_arrays
        self.positivity_fast_kernel_arrays = positivity_fast_kernel_arrays

    def get_pos_pass(self, params):
        _, pdf = self.pred_and_pdf(params, self.fast_kernel_arrays)
        pos_pass, _ = self.positivity_check_and_penalty(
            pdf,
            self.positivity_fast_kernel_arrays,
        )
        return pos_pass

    def __call__(self, params, batch: BatchSpec | None = None):
        """
        Note that this function is called by the samplers, and it must be
        a function of the model parameters only by default.
        If a batch is provided, the log-likelihood is computed only for the
        subset of data indexed by batch.

        Parameters
        ----------
        params: jnp.ndarray
            The model parameters.

        batch: BatchSpec, optional
            If provided, computes the log-likelihood only for the subset of data
            indexed by batch.idx, using precomputed batch.inv_cov if available.

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
            batch=batch,
        )

    def positivity_check_and_penalty(self, pdf, positivity_fast_kernel_arrays):
        if self.positivity_penalty_settings["positivity_penalty"]:
            pos_penalties = self.penalty_posdata(
                pdf,
                self.positivity_penalty_settings["alpha"],
                self.positivity_penalty_settings["lambda_positivity"],
                positivity_fast_kernel_arrays,
            )
            pos_pass = jnp.all(pos_penalties < THRESHOLD_POS)

            pos_penalty = jnp.sum(
                pos_penalties,
                axis=-1,
            )
        else:
            pos_penalty = 0
            pos_pass = True
        return pos_pass, pos_penalty

    @partial(jax.jit, static_argnames=("self",))
    def log_likelihood(
        self,
        params: jnp.ndarray,
        central_values: jnp.ndarray,
        inv_covmat: jnp.ndarray,
        fast_kernel_arrays: tuple,
        positivity_fast_kernel_arrays: tuple,
        batch: BatchSpec | None = None,
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
        predictions, pdf = self.forward_map(fast_kernel_arrays, params)
        # Select only the data relevant for this likelihood
        # Especially important when using a training/validation split
        predictions = predictions[self.central_values_idx]

        if batch is not None:
            predictions = predictions[batch.idx]
            central_values = central_values[batch.idx]
            if batch.inv_cov is None:
                batched_covmat = self.covmat[batch.idx][:, batch.idx]
                inv_covmat = jnp.linalg.inv(batched_covmat)
            else:
                inv_covmat = batch.inv_cov

        _, pos_penalty = self.positivity_check_and_penalty(
            pdf,
            positivity_fast_kernel_arrays,
        )

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
    forward_map,
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
        forward_map,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        _penalty_posdata,
        positivity_penalty_settings,
        integrability_penalty,
    )


def mc_log_likelihood(
    mc_pseudodata,
    general_covariance_matrix,
    pdf_model,
    forward_map,
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
    covmat_train = general_covariance_matrix[tr_idx][:, tr_idx]

    central_covmat_index_train = CentralCovmatIndex(
        central_values=central_values_train,
        covmat=covmat_train,
        central_values_idx=tr_idx,
    )

    train_loglike = LogLikelihood(
        central_covmat_index_train,
        pdf_model,
        forward_map,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        _penalty_posdata,
        positivity_penalty_settings,
        integrability_penalty,
    )

    if not mc_pseudodata.trval_split:
        # Match n3fit's no-validation behaviour: use the full training set as
        # the monitoring set.  This evaluates the same objective after the
        # epoch update and allows best-epoch selection without a held-out set.
        val_loglike = train_loglike

    else:
        val_idx = mc_pseudodata.validation_indices
        central_values_val = mc_pseudodata.pseudodata[val_idx]
        covmat_val = general_covariance_matrix[val_idx][:, val_idx]

        central_covmat_index_val = CentralCovmatIndex(
            central_values=central_values_val,
            covmat=covmat_val,
            central_values_idx=val_idx,
        )

        val_loglike = LogLikelihood(
            central_covmat_index_val,
            pdf_model,
            forward_map,
            fast_kernel_arrays,
            positivity_fast_kernel_arrays,
            _penalty_posdata,
            positivity_penalty_settings,
            integrability_penalty,
        )

    return train_loglike, val_loglike
