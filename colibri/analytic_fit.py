"""
colibri.analytic_fit.py

For a linear model, this module allows for an analytic Bayesian fit of the
model.

"""

import time

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
import jax.lax.linalg as jlinalg
import numpy as np
import scipy.special as special
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler

from colibri.core import AnalyticFit
from colibri.export_results import write_replicas, export_bayes_results
from colibri.checks import check_pdf_model_is_linear
from colibri.utils import compute_determinants_of_principal_minors

import logging

log = logging.getLogger(__name__)


def analytic_evidence_uniform_prior(sol_covmat, sol_mean, max_logl, a_vec, b_vec):
    """
    Compute the log of the evidence for Gaussian likelihood and uniform prior.
    The implementation is based on the following paper: https://arxiv.org/pdf/2301.13783
    and consists in a small improvement of the Laplace approximation.

    Parameters
    ----------
    sol_covmat: jnp.ndarray
        Covariance matrix of the posterior (X^T Sigma^-1 X)^-1.

    sol_mean: jnp.ndarray
        Posterior mean vector.

    a_vec: np.ndarray
        Lower bounds of the Uniform prior.

    b_vec: np.ndarray
        Upper bounds of the Uniform prior.

    Returns
    -------
    tuple[float, float]
        The log evidence and the log Occam factor.
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
    central_covmat_index,
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
    central_covmat_index: commondata_utils.CentralCovmatIndex
        dataclass containing central values and covariance matrix.

    _pred_data: @jax.jit CompiledFunction
        Prediction function for the fit.

    pdf_model: pdf_model.PDFModel
        PDF model to fit.

    analytic_settings: dict
        Settings for the analytic fit.

    prior_settings: PriorSettings
        Settings for the prior.

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    fast_kernel_arrays: tuple
        Tuple containing the fast kernel arrays.
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
    central_values = central_covmat_index.central_values
    covmat = central_covmat_index.covmat

    # Solve chi2 analytically for the mean
    Y = central_values - intercept
    X = predictions.T - intercept[:, None]

    t0 = time.time()

    ridge_alpha = analytic_settings.get("ridge_alpha", 0.0)
    ridge_cv_alphas = analytic_settings.get("ridge_cv_alphas", None)
    ridge_cv_folds = analytic_settings.get("ridge_cv_folds", 5)

    elasticnet_alpha = analytic_settings.get("elasticnet_alpha", 0.0)
    elasticnet_l1_ratio = analytic_settings.get("elasticnet_l1_ratio", 0.5)
    elasticnet_cv_alphas = analytic_settings.get("elasticnet_cv_alphas", None)
    elasticnet_cv_l1_ratios = analytic_settings.get("elasticnet_cv_l1_ratios", None)
    elasticnet_cv_folds = analytic_settings.get("elasticnet_cv_folds", 5)

    # Cholesky factorization: S = L L^T
    # upper False means that we want the lower triangular matrix L
    L = jla.cholesky(covmat, upper=False)

    # Whiten the problem: Y' = L^-1 Y, X' = L^-1 X
    # (X^T Sigma^-1 X) = (X'^T X'), (X^T Sigma^-1 Y) = (X'^T Y')
    Y_tilde = jlinalg.triangular_solve(L, Y, left_side=True, lower=True)
    X_tilde = jlinalg.triangular_solve(L, X, left_side=True, lower=True)

    if ridge_cv_alphas is not None or ridge_alpha > 0.0:
        # Ridge regression with sklearn StandardScaler for scale-invariant regularisation.
        #
        # StandardScaler (with_mean=False) divides each column of X_tilde by its
        # standard deviation s_j, so the Ridge penalty ||v||^2 on the scaled
        # coefficients v treats every parameter equally regardless of feature scale.
        # After fitting, we back-transform to the original parameter space:
        #   sol_mean   = v_ridge / s
        #   sol_covmat = D^{-1} (X_scaled^T X_scaled + alpha I)^{-1} D^{-1},  D = diag(s)
        # Equivalent to a Bayesian MAP with Gaussian prior N(0, (1/alpha) I) on v.
        log.warning(
            "With Ridge regularisation the Bayesian evidence metrics assume a "
            "Gaussian prior N(0, (1/alpha)*I) on the scaled parameters rather than a uniform prior."
        )
        n_params = len(parameters)
        X_tilde_np = np.array(X_tilde)
        Y_tilde_np = np.array(Y_tilde)

        scaler = StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(X_tilde_np)
        col_scales = scaler.scale_  # per-feature standard deviations, shape (n_params,)

        if ridge_cv_alphas is not None:
            # Select alpha via k-fold cross-validation on the whitened, scaled problem.
            log.info(
                f"Selecting Ridge alpha via {ridge_cv_folds}-fold CV "
                f"over candidates {ridge_cv_alphas}."
            )
            ridge_cv = RidgeCV(
                alphas=ridge_cv_alphas,
                cv=ridge_cv_folds,
                fit_intercept=False,
            )
            ridge_cv.fit(X_scaled, Y_tilde_np)
            best_alpha = float(ridge_cv.alpha_)
            v_ridge = ridge_cv.coef_
            log.info(f"RidgeCV selected alpha={best_alpha}.")
        else:
            best_alpha = ridge_alpha
            log.info(f"Using Ridge regression with alpha={best_alpha}.")
            ridge = Ridge(alpha=best_alpha, fit_intercept=False)
            ridge.fit(X_scaled, Y_tilde_np)
            v_ridge = ridge.coef_

        # Posterior covariance in the scaled space: (X_scaled^T X_scaled + alpha I)^{-1}
        A_scaled = X_scaled.T @ X_scaled + best_alpha * np.eye(n_params)
        v_covmat = np.linalg.inv(A_scaled)

        # Back-transform to original parameter space
        sol_mean = jnp.array(v_ridge / col_scales)
        sol_covmat = jnp.array(v_covmat / col_scales[:, None] / col_scales[None, :])
    elif elasticnet_alpha > 0.0 or elasticnet_cv_alphas is not None:
        # Elastic Net regularisation with sklearn StandardScaler for scale-invariant
        # regularisation. As with Ridge we work in the whitened+scaled space so that
        # the L1 and L2 penalties treat every feature on equal footing, and we
        # back-transform afterwards.
        #
        # Elastic Net minimises (in the scaled space)
        #   (1/(2n)) ||Y_tilde - X_scaled v||^2
        #     + alpha * l1_ratio * ||v||_1
        #     + 0.5 * alpha * (1 - l1_ratio) * ||v||_2^2
        # which has no closed-form posterior covariance, so we use the Laplace
        # approximation around the MAP solution v* (see below).
        log.warning(
            "With Elastic Net regularisation the Bayesian evidence metrics assume a "
            "Laplace approximation around the MAP solution; this can be inaccurate "
            "for very sparse solutions (many near-zero coefficients)."
        )
        n_params = len(parameters)
        X_tilde_np = np.array(X_tilde)
        Y_tilde_np = np.array(Y_tilde)

        scaler = StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(X_tilde_np)
        col_scales = scaler.scale_  # per-feature standard deviations, shape (n_params,)

        if elasticnet_cv_alphas is not None:
            # Select alpha (and possibly l1_ratio) via k-fold CV on the whitened,
            # scaled problem.
            log.info(
                f"Selecting Elastic Net alpha via {elasticnet_cv_folds}-fold CV "
                f"over alpha candidates {elasticnet_cv_alphas} "
                f"and l1_ratio candidates {elasticnet_cv_l1_ratios}."
            )
            # ElasticNetCV defaults l1_ratio to 0.5 if None is passed; sklearn requires
            # a list/scalar so fall back to the fixed-alpha default.
            cv_l1_ratios = (
                elasticnet_cv_l1_ratios
                if elasticnet_cv_l1_ratios is not None
                else elasticnet_l1_ratio
            )
            enet_cv = ElasticNetCV(
                alphas=elasticnet_cv_alphas,
                l1_ratio=cv_l1_ratios,
                cv=elasticnet_cv_folds,
                fit_intercept=False,
            )
            enet_cv.fit(X_scaled, Y_tilde_np)
            best_alpha = float(enet_cv.alpha_)
            best_l1_ratio = float(enet_cv.l1_ratio_)
            v_enet = enet_cv.coef_
            log.info(
                f"ElasticNetCV selected alpha={best_alpha}, l1_ratio={best_l1_ratio}."
            )
        else:
            best_alpha = elasticnet_alpha
            best_l1_ratio = elasticnet_l1_ratio
            log.info(
                f"Using Elastic Net regression with alpha={best_alpha}, "
                f"l1_ratio={best_l1_ratio}."
            )
            enet = ElasticNet(
                alpha=best_alpha,
                l1_ratio=best_l1_ratio,
                fit_intercept=False,
            )
            enet.fit(X_scaled, Y_tilde_np)
            v_enet = enet.coef_

        # Posterior covariance via the Laplace approximation around v*.
        # The Hessian of the Elastic Net objective in the scaled space at v* is
        #   H = X_scaled^T X_scaled + diag(lambda2 + lambda1 / |v*_i|)
        # with lambda1 = alpha * l1_ratio and lambda2 = alpha * (1 - l1_ratio).
        # Near-zero coefficients are clamped (|v*_i| < 1e-10 -> 1e-10) to avoid
        # division by zero; the penalty there becomes very large, which correctly
        # reflects that the L1 term sharply pins these coefficients to zero.
        lambda1 = best_alpha * best_l1_ratio
        lambda2 = best_alpha * (1.0 - best_l1_ratio)
        abs_v = np.maximum(np.abs(v_enet), 1e-10)
        diag_penalty = lambda2 + lambda1 / abs_v
        H = X_scaled.T @ X_scaled + np.diag(diag_penalty)
        v_covmat = np.linalg.inv(H)

        # Back-transform to original parameter space (same formula as Ridge)
        sol_mean = jnp.array(v_enet / col_scales)
        sol_covmat = jnp.array(v_covmat / col_scales[:, None] / col_scales[None, :])
    else:
        if jnp.any(jla.eigh(X_tilde.T @ X_tilde)[0] <= 0.0):
            raise ValueError(
                "The obtained covariance matrix for the analytic solution is not positive definite."
            )
        # OLS: use QR decomposition of X_tilde for numerical stability
        Q, R = jla.qr(X_tilde)
        # NOTE: R is upper triangular in QR decomposition, so lower=False
        sol_mean = jlinalg.triangular_solve(R, Q.T @ Y_tilde, left_side=True, lower=False)
        I_R = jnp.eye(R.shape[0])
        R_inv = jlinalg.triangular_solve(R, I_R, left_side=True, lower=False)
        sol_covmat = R_inv @ R_inv.T

    key = jax.random.PRNGKey(analytic_settings["sampling_seed"])

    # full samples with no cuts from the prior bounds
    full_samples = jax.random.multivariate_normal(
        key,
        sol_mean,
        sol_covmat,
        shape=(analytic_settings["full_sample_size"],),
    )

    # Compute the evidence
    # This is the log of the evidence, which is the log of the integral of the likelihood
    # over the prior. The prior is uniform with width prior_width.
    log.info("Computing the evidence...")

    if prior_settings.prior_distribution == "n_sigma_prior":
        nsigma = prior_settings.prior_distribution_specs["n_sigma_value"]

        log.info(f"Using +- {nsigma} sigma of covmat")
        diags = np.sqrt(np.diag(sol_covmat))

        prior_lower = sol_mean - nsigma * diags
        prior_upper = sol_mean + nsigma * diags

    elif prior_settings.prior_distribution == "custom_uniform_parameter_prior":
        log.info("Using custom uniform prior")
        prior_lower = jnp.array(prior_settings.prior_distribution_specs["lower_bounds"])
        prior_upper = jnp.array(prior_settings.prior_distribution_specs["upper_bounds"])

    elif prior_settings.prior_distribution == "min_max_prior":
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
    # Compute maximum log likelihood in the whitened basis
    min_chi2 = (Y_tilde - X_tilde @ sol_mean).T @ (Y_tilde - X_tilde @ sol_mean)
    # Compute the log likelihood
    max_logl = -0.5 * min_chi2

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

    BIC = min_chi2 + sol_covmat.shape[0] * np.log(covmat.shape[0])
    AIC = min_chi2 + 2 * sol_covmat.shape[0]

    # Compute average chi2 (in whitened basis)
    diffs = Y_tilde[:, None] - X_tilde @ full_samples.T
    avg_chi2 = jnp.mean(jnp.sum(diffs**2, axis=0))

    log.info(f"Average chi2 = {avg_chi2}")

    # Compute the Bayesian complexity
    Cb = avg_chi2 - min_chi2
    log.info(f"Bayesian complexity = {Cb}")

    # Resample the posterior for PDF set
    samples = full_samples[: analytic_settings["n_posterior_samples"]]

    t1 = time.time()
    log.info("ANALYTIC SAMPLING RUNTIME: %f s" % (t1 - t0))

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


def run_analytic_fit(analytic_fit, output_path, pdf_model, Q0):
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
    Q0: float
        The scale at which to export the PDFs.
    """

    export_bayes_results(analytic_fit, output_path, "analytic_result")

    write_replicas(analytic_fit, output_path, pdf_model, Q0)
