from validphys.convolution import FK_FLAVOURS
from validphys.loader import Loader
from validphys.lhio import generate_replica0

from grid_pdf.grid_pdf_lhapdf import lhapdf_grid_pdf_from_samples

import jax
import jax.scipy.linalg as jla
import jax.numpy as jnp
import pandas as pd

import time
import logging

log = logging.getLogger(__name__)


def analytic_gridpdf_fit(
    _data_values,
    flavour_indices,
    reduced_xgrids,
    precomputed_predictions,
    pdfgrid_sampling_index=123456,
    n_posterior_samples=100,
):
    """
    Valid only for DIS datasets.
    """
    parameters = [
        f"{FK_FLAVOURS[i]}({j})" for i in flavour_indices for j in reduced_xgrids[i]
    ]

    training_data = _data_values.training_data
    central_values = training_data.central_values
    covmat = training_data.covmat
    central_values_idx = training_data.central_values_idx

    # Invert the covmat
    inv_covmat = jla.inv(covmat)

    # Solve chi2 analytically for the mean
    Y = central_values
    Sigma = inv_covmat
    X = (precomputed_predictions[:, central_values_idx]).T

    t0 = time.time()
    gridpdf_mean = jla.inv(X.T @ Sigma @ X) @ X.T @ Sigma @ Y
    gridpdf_covmat = jla.inv(X.T @ Sigma @ X)

    # * Check that cov mat is semi-positive definite
    if jnp.any(jla.eigh(gridpdf_covmat, eigvals_only=True) < 0.0):
        raise ValueError(
            "The obtained covariance matrix for the gridpdf is not semi-postive definite."
        )

    key = jax.random.PRNGKey(pdfgrid_sampling_index)

    samples = jax.random.multivariate_normal(
        key,
        gridpdf_mean,
        gridpdf_covmat,
        shape=(n_posterior_samples,),
    )
    t1 = time.time()
    log.info("ANALYTIC SAMPLING RUNNING TIME: %f s" % (t1 - t0))

    return (parameters, samples)


def perform_analytic_gridpdf_fit(
    analytic_gridpdf_fit,
    reduced_xgrids,
    flavour_indices,
    length_reduced_xgrids,
    n_posterior_samples,
    lhapdf_path,
    output_path,
    theoryid,
):
    """
    Performs an Analytic fit using the grid.
    """

    # Save the resampled posterior as a pandas df
    parameter_names, analytic_gridpdf_fit = analytic_gridpdf_fit
    df = pd.DataFrame(analytic_gridpdf_fit, columns=parameter_names)
    df.to_csv(str(output_path) + "/analytic_result.csv")

    # Produce the LHAPDF grid
    lhapdf_grid_pdf_from_samples(
        analytic_gridpdf_fit,
        reduced_xgrids,
        flavour_indices,
        length_reduced_xgrids,
        n_posterior_samples,
        theoryid,
        folder=lhapdf_path,
        output_path=output_path,
    )

    # Produce the central replica
    l = Loader()
    pdf = l.check_pdf(str(output_path).split("/")[-1])
    generate_replica0(pdf)

    log.info("Analytic grid PDF fit completed!")


def analyticmc_gridpdf_fit(
    _data_values,
    flavour_indices,
    reduced_xgrids,
    precomputed_predictions,
    pdfgrid_sampling_index=123456,
    n_replicas=100,
):
    """
    Valid only for DIS datasets.
    """
    parameters = [
        f"{FK_FLAVOURS[i]}({j})" for i in flavour_indices for j in reduced_xgrids[i]
    ]

    training_data = _data_values.training_data
    central_values = training_data.central_values
    covmat = training_data.covmat
    central_values_idx = training_data.central_values_idx

    # Invert the covmat
    inv_covmat = jla.inv(covmat)

    key = jax.random.PRNGKey(pdfgrid_sampling_index)

    mc_replicas = jax.random.multivariate_normal(
        key,
        central_values,
        covmat,
        shape=(n_replicas,),
    )
    t0 = time.time()
    samples = []
    for replica in mc_replicas:
        # Solve chi2 analytically for the mean
        Y = replica
        Sigma = inv_covmat
        X = (precomputed_predictions[:, central_values_idx]).T

        gridpdf_replica = jla.inv(X.T @ Sigma @ X) @ X.T @ Sigma @ Y

        samples.append(gridpdf_replica)

    t1 = time.time()
    log.info("ANALYTIC MC SAMPLING RUNNING TIME: %f s" % (t1 - t0))

    return (parameters, samples)


def perform_analyticmc_gridpdf_fit(
    analyticmc_gridpdf_fit,
    reduced_xgrids,
    flavour_indices,
    length_reduced_xgrids,
    n_replicas,
    lhapdf_path,
    output_path,
    theoryid,
):
    """
    Performs an Analytic fit using the grid.
    """

    # Save the resampled posterior as a pandas df
    parameter_names, analyticmc_gridpdf_fit = analyticmc_gridpdf_fit
    df = pd.DataFrame(analyticmc_gridpdf_fit, columns=parameter_names)
    df.to_csv(str(output_path) + "/analyticmc_result.csv")

    # Produce the LHAPDF grid
    lhapdf_grid_pdf_from_samples(
        analyticmc_gridpdf_fit,
        reduced_xgrids,
        flavour_indices,
        length_reduced_xgrids,
        n_replicas,
        theoryid,
        folder=lhapdf_path,
        output_path=output_path,
    )

    # Produce the central replica
    l = Loader()
    pdf = l.check_pdf(str(output_path).split("/")[-1])
    generate_replica0(pdf)

    log.info("Analytic MC grid PDF fit completed!")
