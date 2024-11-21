"""
colibri.analytic_fit.py

For a linear model, this module allows for an analytic Bayesian fit of the
model.

"""

from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
import numpy as np
import scipy.special as special

from colibri.export_results import BayesianFit, write_replicas, export_bayes_results
from colibri.checks import check_pdf_model_is_linear
from colibri.utils import compute_determinants_of_principal_minors

import logging

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalyticFit(BayesianFit):
    """
    Dataclass containing the results and specs of an analytic fit.

    Attributes
    ----------
    analytic_specs: dict
        Dictionary containing the settings of the analytic fit.
    resampled_posterior: jnp.array
        Array containing the resampled posterior samples.
    """

    analytic_specs: dict


def analytic_evidence_uniform_prior(sol_covmat, sol_mean, max_logl, a_vec, b_vec):
    """
    Compute the log of the evidence for Gaussian likelihood and uniform prior.
    The implementation is based on the following paper: https://arxiv.org/pdf/2301.13783
    and consists in a small improvement of the Laplace approximation.

    Parameters
    ----------
    sol_covmat: array
        Covariance matrix of the posterior (X^T Sigma^-1 X)^-1.

    sol_mean

    a_vec: np.ndarray
        Lower bounds of the Uniform prior.

    b_vec: np.ndarray
        Upper bounds of the Uniform prior.

    Returns
    -------
    float: The log evidence.
    """

    # Take into account change of variables of type (x - mu) -> x
    b_vec -= sol_mean
    a_vec -= sol_mean

    determinants = compute_determinants_of_principal_minors(sol_covmat)

    sqrt_det_ratios = np.sqrt(determinants[:-1] / determinants[1:])

    erf_arg_a = a_vec / np.sqrt(2) * sqrt_det_ratios
    erf_arg_b = b_vec / np.sqrt(2) * sqrt_det_ratios

    erf_a = special.erf(erf_arg_a)
    erf_b = special.erf(erf_arg_b)

    log_erf_terms = np.log(0.5 * (erf_b - erf_a)).sum()

    occam_factor_num = np.sqrt(jla.det(sol_covmat))
    occam_factor_denom = np.prod((b_vec - a_vec))

    log_occam_factor = np.log(occam_factor_num / occam_factor_denom)

    log_evidence = (
        max_logl
        + sol_covmat.shape[0] / 2 * np.log(2 * np.pi)
        + log_occam_factor
        + log_erf_terms
    )

    return log_evidence, log_occam_factor


@check_pdf_model_is_linear
def analytic_fit(
    central_inv_covmat_index,
    _pred_data,
    pdf_model,
    analytic_settings,
    prior_settings,
    FIT_XGRID,
    fast_kernel_arrays,
):
    """
    Analytic fits, for any *linear* PDF model.

    The assumption is that the model is linear with an intercept:
    T(w) = T(0) + X w.
    The linear problem to solve is through minimisation of the chi2:
    chi2 = (D - (T(0) + X w))^T Sigma^-1 (D - (T(0) + X w)) = (Y - X w)^T Sigma^-1 (Y - X w)
    with Y = D - T(0).

    Parameters
    ----------
    central_inv_covmat_index: commondata_utils.CentralInvCovmatIndex
        dataclass containing central values and inverse covmat.

    _pred_data: @jax.jit CompiledFunction
        Prediction function for the fit.

    pdf_model: pdf_model.PDFModel
        PDF model to fit.

    analytic_settings: dict
        Settings for the analytic fit.

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.
    """

    log.warning("The prior is assumed to be flat in the parameters.")
    log.warning(
        "Assuming that the prior is wide enough to fully cover the gaussian likelihood."
    )

    parameters = pdf_model.param_names
    pred_and_pdf = pdf_model.pred_and_pdf_func(FIT_XGRID, forward_map=_pred_data)

    # Precompute predictions for the basis of the model
    bases = jnp.identity(len(parameters))
    predictions = jnp.array(
        [pred_and_pdf(basis, fast_kernel_arrays)[0] for basis in bases]
    )
    intercept = pred_and_pdf(jnp.zeros(len(parameters)), fast_kernel_arrays)[0]

    # Construct the analytic solution
    central_values = central_inv_covmat_index.central_values
    inv_covmat = central_inv_covmat_index.inv_covmat

    # Solve chi2 analytically for the mean
    Y = central_values - intercept
    Sigma = inv_covmat
    X = predictions.T - intercept[:, None]

    # * Check that cov mat is positive definite
    if jnp.any(jla.eigh(X.T @ Sigma @ X)[0] <= 0.0):
        raise ValueError(
            "The obtained covariance matrix for the analytic solution is not positive definite."
        )

    t0 = time.time()
    sol_mean = jla.inv(X.T @ Sigma @ X) @ X.T @ Sigma @ Y
    sol_covmat = jla.inv(X.T @ Sigma @ X)

    key = jax.random.PRNGKey(analytic_settings["sampling_seed"])

    # full samples with no cuts from the prior bounds
    full_samples = jax.random.multivariate_normal(
        key,
        sol_mean,
        sol_covmat,
        shape=(analytic_settings["full_sample_size"],),
    )
    t1 = time.time()
    log.info("ANALYTIC SAMPLING RUNTIME: %f s" % (t1 - t0))

    # Compute the evidence
    # This is the log of the evidence, which is the log of the integral of the likelihood
    # over the prior. The prior is uniform with width prior_width.
    log.info("Computing the evidence...")

    if prior_settings.prior_distribution["n_sigma_prior"]:
        nsigma = prior_settings.prior_distribution_specs["n_sigma_value"]

        log.info(f"Using +- {nsigma} sigma of covmat")
        diags = np.sqrt(np.diag(sol_covmat))

        prior_lower = sol_mean - nsigma * diags
        prior_upper = sol_mean + nsigma * diags

    elif prior_settings.prior_distribution["custom_uniform_parameter_prior"]:
        log.info("Using custom uniform prior")
        prior_lower = prior_settings.prior_distribution_specs["lower_bounds"]
        prior_upper = prior_settings.prior_distribution_specs["upper_bounds"]

    elif prior_settings.prior_distribution["min_max_prior"]:
        log.info("Using min-max prior")
        prior_lower = full_samples.min(axis=0)
        prior_upper = full_samples.max(axis=0)

    else:
        # Extract lower and upper bounds of the prior
        prior_lower = prior_settings.prior_distribution_specs["min_val"] * jnp.ones(
            len(parameters)
        )
        prior_upper = prior_settings.prior_distribution_specs["max_val"] * jnp.ones(
            len(parameters)
        )

    prior_width = prior_upper - prior_lower

    # Check that the prior is wide enough
    if jnp.any(full_samples < prior_lower) or jnp.any(full_samples > prior_upper):
        log.error(
            "The prior is not wide enough to cover the posterior samples. Increase the prior width."
        )

    log.warning(f"Discarding samples outside the prior bounds.")

    # discard samples outside the prior
    full_samples = full_samples[
        (full_samples > prior_lower).all(axis=1)
        & (full_samples < prior_upper).all(axis=1)
    ]

    gaussian_integral = jnp.log(jnp.sqrt(jla.det(2 * jnp.pi * sol_covmat)))
    log_prior = jnp.log(1 / prior_width).sum()
    # Compute maximum log likelihood
    max_logl = -0.5 * (Y @ Sigma @ Y - Y @ Sigma @ X @ sol_mean)

    logZ_laplace = gaussian_integral + max_logl + log_prior

    log.info(f"LogZ (Laplace approximation) = {logZ_laplace}")

    # computation of the evidence (analytic approximation)
    logZ_analytical, log_occam_factor = analytic_evidence_uniform_prior(
        sol_covmat, sol_mean, max_logl, prior_lower, prior_upper
    )

    log.info(f"LogZ (Analytic approximation) = {logZ_analytical}")
    log.info(f"Log Occam factor = {log_occam_factor}")
    log.info(f"Maximal log likelihood = {max_logl}")

    # Compute minimum chi2
    min_chi2 = -2 * max_logl
    log.info(f"Minimum chi2 = {min_chi2}")

    BIC = min_chi2 + sol_covmat.shape[0] * np.log(Sigma.shape[0])
    AIC = min_chi2 + 2 * sol_covmat.shape[0]

    # Compute average chi2
    diffs = Y[:, None] - X @ full_samples.T
    avg_chi2 = jnp.einsum("ij,jk,ki->i", diffs.T, Sigma, diffs).mean()

    log.info(f"Average chi2 = {avg_chi2}")

    # Compute the Bayesian complexity
    Cb = avg_chi2 - min_chi2
    log.info(f"Bayesian complexity = {Cb}")

    # Resample the posterior for PDF set
    samples = full_samples[: analytic_settings["n_posterior_samples"]]

    return AnalyticFit(
        analytic_specs=analytic_settings,
        resampled_posterior=samples,
        param_names=parameters,
        full_posterior_samples=full_samples,
        bayesian_metrics={
            "bayes_complexity": Cb,
            "avg_chi2": avg_chi2,
            "min_chi2": min_chi2,
            "logZ_laplace": logZ_laplace,
            "logz": logZ_analytical,
            "log_occam_factor": log_occam_factor,
            "BIC": BIC,
            "AIC": AIC,
        },
    )


def run_analytic_fit(analytic_fit, output_path, pdf_model):
    """
    Export the results of an analytic fit.

    Parameters
    ----------
    analytic_fit: AnalyticFit
        The results of the analytic fit.
    output_path: pathlib.PosixPath
        Path to the output folder.
    pdf_model: pdf_model.PDFModel
        The PDF model used in the fit.
    """

    export_bayes_results(analytic_fit, output_path, "analytic_result")

    write_replicas(analytic_fit, output_path, pdf_model)
