from validphys.convolution import FK_FLAVOURS
from validphys.loader import Loader
from validphys.lhio import generate_replica0

from super_net.mc_utils import mc_pseudodata
from grid_pdf.grid_pdf_lhapdf import (
    write_exportgrid_from_fit_samples,
)

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
    output_path,
):
    """
    Performs an Analytic fit using the grid.
    """

    # Save the resampled posterior as a pandas df
    parameter_names, analytic_gridpdf_fit = analytic_gridpdf_fit
    df = pd.DataFrame(analytic_gridpdf_fit, columns=parameter_names)
    df.to_csv(str(output_path) + "/analytic_result.csv")

    # Produce exportgrid files for each posterior sample
    write_exportgrid_from_fit_samples(
        samples=analytic_gridpdf_fit,
        n_posterior_samples=n_posterior_samples,
        reduced_xgrids=reduced_xgrids,
        length_reduced_xgrids=length_reduced_xgrids,
        flavour_indices=flavour_indices,
        output_path=output_path,
    )

    log.info("Analytic grid PDF fit completed!")


def analyticmc_gridpdf_fit(
    pseudodata_central_covmat_index,
    fit_covariance_matrix,
    flavour_indices,
    reduced_xgrids,
    precomputed_predictions,
    shuffle_indices=True,
    n_replicas=100,
    mc_validation_fraction=0.2,
):
    """
    Valid only for DIS datasets.
    """
    parameters = [
        f"{FK_FLAVOURS[i]}({j})" for i in flavour_indices for j in reduced_xgrids[i]
    ]

    covmat = fit_covariance_matrix

    mc_replicas = []
    for i in range(n_replicas):
        trval_key = jax.random.PRNGKey(i)
        replica = mc_pseudodata(
            pseudodata_central_covmat_index,
            i,
            trval_seed=trval_key,
            shuffle_indices=shuffle_indices,
            mc_validation_fraction=mc_validation_fraction,
        )

        central_values_idx = replica.training_indices
        central_values = replica.pseudodata[central_values_idx]
        replica_covmat = covmat[central_values_idx][:, central_values_idx]
        replica_inv_covmat = jla.inv(replica_covmat)

        mc_replicas.append((central_values, central_values_idx, replica_inv_covmat))

    t0 = time.time()
    samples = []
    for central_values, idx, inv_covmat in mc_replicas:
        # Solve chi2 analytically for the mean
        Y = central_values
        Sigma = inv_covmat
        X = (precomputed_predictions[:, idx]).T

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
    output_path,
):
    """
    Performs an Analytic fit using the grid.
    """

    # Save the resampled posterior as a pandas df
    parameter_names, analyticmc_gridpdf_fit = analyticmc_gridpdf_fit
    df = pd.DataFrame(analyticmc_gridpdf_fit, columns=parameter_names)
    df.to_csv(str(output_path) + "/analyticmc_result.csv")

    # Produce exportgrid files for each posterior sample
    write_exportgrid_from_fit_samples(
        samples=analyticmc_gridpdf_fit,
        n_posterior_samples=n_replicas,
        reduced_xgrids=reduced_xgrids,
        length_reduced_xgrids=length_reduced_xgrids,
        flavour_indices=flavour_indices,
        output_path=output_path,
    )

    log.info("Analytic MC grid PDF fit completed!")
