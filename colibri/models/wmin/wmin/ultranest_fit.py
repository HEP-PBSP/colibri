"""
wmin.ultranest_fit.py

This module overrides the log likelihood defined in colibri.ultranest_fit so as to allow
the user to add model dependent terms to the likelihood

"""

import jax
import jax.numpy as jnp
from functools import partial
from colibri.ultranest_fit import UltraNestLogLikelihood
from colibri.loss_functions import chi2


class WminUltraNestLogLikelihood(UltraNestLogLikelihood):
    """
    UltraNest log likelihood with additional terms for the wmin model.
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
        alpha,
        lambda_positivity,
        wmin_l2_reg=False,
        wmin_l2_reg_factor=10,
    ):
        super().__init__(
            central_inv_covmat_index,
            pdf_model,
            fit_xgrid,
            forward_map,
            fast_kernel_arrays,
            positivity_fast_kernel_arrays,
            ns_settings,
            chi2,
            penalty_posdata,
            alpha,
            lambda_positivity,
        )
        self.wmin_l2_reg = wmin_l2_reg
        self.wmin_l2_reg_factor = wmin_l2_reg_factor

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

        if self.wmin_l2_reg:
            wmin_l2_reg_penalty = self.wmin_l2_reg_factor * jnp.sum(params**2)
        else:
            wmin_l2_reg_penalty = 0

        return -0.5 * (
            self.chi2(central_values, predictions, inv_covmat)
            + jnp.sum(
                self.penalty_posdata(
                    pdf,
                    self.alpha,
                    self.lambda_positivity,
                    positivity_fast_kernel_arrays,
                ),
                axis=-1,
            )
            + wmin_l2_reg_penalty
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
    alpha,
    lambda_positivity,
    wmin_l2_reg=False,
    wmin_l2_reg_factor=10,
):
    """
    TODO
    """
    return WminUltraNestLogLikelihood(
        central_inv_covmat_index,
        pdf_model,
        FIT_XGRID,
        _pred_data,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        ns_settings,
        chi2,
        _penalty_posdata,
        alpha,
        lambda_positivity,
        wmin_l2_reg,
        wmin_l2_reg_factor,
    )
