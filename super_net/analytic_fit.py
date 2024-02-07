"""
super_net.analytic_fit.py

For a linear model, this module allows for an analytic Bayesian fit of the
model.

"""

from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla

import pandas as pd

from super_net.constants import XGRID
from super_net.lhapdf import write_exportgrid

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
    _data_values,
    _pred_data,
    pdf_model,
    analytic_settings,
    output_path,
):
    """
    Analytic fits, for any *linear* PDF model.

    Parameters
    ----------
    _data_values: MakeDataValues
        Data values for the fit.

    _pred_data: @jax.jit CompiledFunction
        Prediction function for the fit.

    pdf_model: pdf_model.PDFModel
        PDF model to fit.

    analytic_settings: dict
        Settings for the analytic fit.

    output_path: str
        Path to write the results to.
    """

    parameters = pdf_model.param_names
    fit_grid_values_func = pdf_model.grid_values_func(XGRID)

    # Precompute predictions for the basis of the model
    bases = jnp.identity(len(parameters))
    pdf_bases = [fit_grid_values_func(basis) for basis in bases]
    predictions = jnp.array([_pred_data(pdf_basis) for pdf_basis in pdf_bases])

    # Construct the analytic solution
    training_data = _data_values.training_data
    central_values = training_data.central_values
    covmat = training_data.covmat
    central_values_idx = training_data.central_values_idx

    # Invert the covmat
    inv_covmat = jla.inv(covmat)

    # Solve chi2 analytically for the mean
    Y = central_values
    Sigma = inv_covmat
    X = (predictions[:, central_values_idx]).T

    t0 = time.time()
    sol_mean = jla.inv(X.T @ Sigma @ X) @ X.T @ Sigma @ Y
    sol_covmat = jla.inv(X.T @ Sigma @ X)

    # * Check that cov mat is semi-positive definite
    if jnp.any(jla.eigh(sol_covmat)[0] < 0.0):
        raise ValueError(
            "The obtained covariance matrix for the analytic solution is not semi-postive definite."
        )

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
