"""
colibri.analytic_fit.py

For a linear model, this module allows for an analytic Bayesian fit of the
model.

"""

from dataclasses import dataclass
import time
import os

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla

import pandas as pd

from colibri.lhapdf import write_exportgrid

import logging

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalyticFit:
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
    resampled_posterior: jnp.array


def analytic_fit(
    central_covmat_index,
    _pred_data,
    pdf_model,
    analytic_settings,
    output_path,
    FIT_XGRID,
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
        dataclass containing central values and covmat.

    _pred_data: @jax.jit CompiledFunction
        Prediction function for the fit.

    pdf_model: pdf_model.PDFModel
        PDF model to fit.

    analytic_settings: dict
        Settings for the analytic fit.

    output_path: str
        Path to write the results to.

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.
    """

    parameters = pdf_model.param_names
    pred_and_pdf = pdf_model.pred_and_pdf_func(FIT_XGRID, forward_map=_pred_data)

    # Precompute predictions for the basis of the model
    bases = jnp.identity(len(parameters))
    predictions = jnp.array([pred_and_pdf(basis)[0] for basis in bases])
    intercept = pred_and_pdf(jnp.zeros(len(parameters)))[0]

    # Construct the analytic solution
    central_values = central_covmat_index.central_values
    covmat = central_covmat_index.covmat

    # Invert the covmat
    inv_covmat = jla.inv(covmat)

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

    # Compute evidence logZ
    extra = -0.5 * (Y @ Sigma @ Y - Y @ Sigma @ X @ sol_mean)
    logZ = jnp.log(jnp.sqrt(jla.det(2 * jnp.pi * sol_covmat))) + extra
    log.info(f"LogZ = {logZ}")

    key = jax.random.PRNGKey(analytic_settings["sampling_seed"])

    samples = jax.random.multivariate_normal(
        key,
        sol_mean,
        sol_covmat,
        shape=(analytic_settings["n_posterior_samples"],),
    )
    t1 = time.time()
    log.info("ANALYTIC SAMPLING RUNTIME: %f s" % (t1 - t0))

    # Save the results
    df = pd.DataFrame(samples, columns=parameters)
    df.to_csv(str(output_path) + "/analytic_result.csv")

    # create replicas folder if it does not exist
    replicas_path = str(output_path) + "/replicas"
    if not os.path.exists(replicas_path):
        os.mkdir(replicas_path)

    # Finish by writing the replicas to export grids, ready for evolution
    for i in range(analytic_settings["n_posterior_samples"]):
        log.info(f"Writing exportgrid for replica {i+1}")
        write_exportgrid(
            jnp.array(df.iloc[i, :].tolist()), pdf_model, i + 1, output_path
        )

    return AnalyticFit(
        analytic_specs=analytic_settings,
        resampled_posterior=samples,
    )
