import pathlib

import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from colibri.api import API as colibriAPI
from colibri.covmats import sqrt_covmat_jax
from colibri.tests.conftest import T0_PDFSET, TEST_DATASETS

TEST_COVMATS_FOLDER = pathlib.Path(__file__).with_name("test_covmats")


def test_sqrt_covmat_jax():
    """
    Test that sqrt_covmat_jax actually computes the square root of a matrix.
    """

    test_matrix = jnp.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])

    # This matrix has square root [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
    sqrt_matrix = sqrt_covmat_jax(test_matrix)
    actual_sqrt = jnp.array([[2, 0, 0], [6, 1, 0], [-8, 5, 3]])
    assert_allclose(sqrt_matrix, actual_sqrt, rtol=1e-5)


def test_dataset_inputs_covmat_from_systematics():
    """
    Test that dataset_inputs_covmat_from_systematics correctly produces the covariance matrix.
    """

    result = colibriAPI.dataset_inputs_covmat_from_systematics(**TEST_DATASETS)

    # Check result is a JAX array
    assert isinstance(result, jnp.ndarray)

    # Check result is symmetric and positive definite (i.e. all evalues are positive)
    assert_allclose(result, result.T)
    assert jnp.all(jnp.linalg.eigvals(result) > 0)

    # Check result is correct for given datasets
    path = TEST_COVMATS_FOLDER / "test_ds_inputs_covmat.csv"
    assert_allclose(
        result,
        pd.read_csv(path).to_numpy(dtype=float),
    )


def test_colibri_dataset_inputs_t0_predictions():
    """
    Test the t0_predictions.
    """

    result = colibriAPI.colibri_dataset_inputs_t0_predictions(
        **{**TEST_DATASETS, **T0_PDFSET}
    )

    # Check that the result is a list of numpy arrays
    assert isinstance(result, list)
    for pred in result:
        assert isinstance(pred, np.ndarray)

    # Check that the result is correct for the given datasets
    path = TEST_COVMATS_FOLDER / "NMC_NC_NOTFIXED_P_EM-SIGMARED_t0_predictions.csv"
    assert_allclose(
        result[0], pd.read_csv(path)["data"].to_numpy(dtype=float), rtol=1e-5
    )


def test_dataset_inputs_t0_covmat_from_systematics():
    """
    Test the t0 covmat.
    """

    result = colibriAPI.dataset_inputs_t0_covmat_from_systematics(
        **{**TEST_DATASETS, **T0_PDFSET}
    )

    # Check result is a JAX array
    assert isinstance(result, jnp.ndarray)

    # Check result is symmetric and positive definite (i.e. all evalues are positive)
    assert_allclose(result, result.T)
    assert jnp.all(jnp.linalg.eigvals(result) > 0)

    # Check result is correct for given datasets
    path = TEST_COVMATS_FOLDER / "test_t0_covmat.csv"
    assert_allclose(
        result,
        pd.read_csv(path, header=None).to_numpy(dtype=float),
        rtol=1e-4,
    )
