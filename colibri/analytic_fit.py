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

from colibri.export_results import BayesianFit, write_replicas, export_bayes_results
from colibri.checks import check_pdf_model_is_linear

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


@check_pdf_model_is_linear
def analytic_fit(
    central_inv_covmat_index,
    _pred_data,
    pdf_model,
    analytic_settings,
    bayesian_prior,
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

    if analytic_settings["optimal_prior"]:
        log.info("Using optimal prior")
        prior_lower = full_samples.min(axis=0)
        prior_upper = full_samples.max(axis=0)
    else:
        # Extract lower and upper bounds of the prior
        prior_lower = bayesian_prior(jnp.zeros(len(parameters)))
        prior_upper = bayesian_prior(jnp.ones(len(parameters)))

    prior_width = prior_upper - prior_lower

    gaussian_integral = jnp.log(jnp.sqrt(jla.det(2 * jnp.pi * sol_covmat)))
    log_prior = jnp.log(1 / prior_width).sum()
    # Compute maximum log likelihood
    max_logl = -0.5 * (Y @ Sigma @ Y - Y @ Sigma @ X @ sol_mean)

    logZ = gaussian_integral + max_logl + log_prior

    log.info(f"LogZ = {logZ}")
    log.info(f"Maximum log likelihood = {max_logl}")

    # Compute minimum chi2
    min_chi2 = -2 * max_logl
    log.info(f"Minimum chi2 = {min_chi2}")

    # Check that the prior is wide enough
    if jnp.any(full_samples < prior_lower) or jnp.any(full_samples > prior_upper):
        log.error(
            "The prior is not wide enough to cover the posterior samples. Increase the prior width."
        )

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
            "logz": logZ,
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
