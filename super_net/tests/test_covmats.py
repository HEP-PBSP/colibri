import pathlib
import pandas as pd
import numpy as np

from super_net.covmats import sqrt_covmat_jax 

from super_net.api import API as SuperNetAPI
from super_net.tests.conftest import TEST_DATASETS, T0_PDFSET

import jax
import jax.numpy as jnp

from numpy.testing import assert_allclose

TEST_COVMATS_FOLDER = pathlib.Path(__file__).with_name('test_covmats')

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

    result = SuperNetAPI.dataset_inputs_covmat_from_systematics(**TEST_DATASETS)

    # Check result is a JAX array
    assert isinstance(result, jnp.ndarray)

    # Check result is symmetric and positive definite (i.e. all evalues are positive)
    assert_allclose(result, result.T)
    assert jnp.all(jnp.linalg.eigvals(result) > 0)

    # Check result is correct for given datasets
    path = TEST_COVMATS_FOLDER/'test_ds_inputs_covmat.csv'
    assert_allclose(
        result,
        pd.read_csv(path).to_numpy(dtype=float),
    )    

def test_super_net_dataset_inputs_t0_predictions():
    """
    Test the t0_predictions.
    """

    result = SuperNetAPI.super_net_dataset_inputs_t0_predictions(**{**TEST_DATASETS, **T0_PDFSET})

    # Check that the result is a list of numpy arrays
    assert isinstance(result, list)
    for pred in result:
        assert isinstance(pred, np.ndarray)

    # Check that the result is correct for the given datasets
    path = TEST_COVMATS_FOLDER/'NMC_t0_predictions.csv'
    assert_allclose(
        result,
        pd.read_csv(path).to_numpy(dtype=float),
        rtol=1e-5
    )

def test_dataset_inputs_t0_covmat_from_systematics():
    """
    Test the t0 covmat.
    """

    result = SuperNetAPI.dataset_inputs_t0_covmat_from_systematics(**{**TEST_DATASETS, **T0_PDFSET})

    # Check result is a JAX array
    assert isinstance(result, jnp.ndarray)

    # Check result is symmetric and positive definite (i.e. all evalues are positive)
    assert_allclose(result, result.T)
    assert jnp.all(jnp.linalg.eigvals(result) > 0)

    # Check result is correct for given datasets
    path = TEST_COVMATS_FOLDER/'test_t0_covmat.csv'
    assert_allclose(
        result,
        pd.read_csv(path).to_numpy(dtype=float),
        rtol=1e-5,
    )
