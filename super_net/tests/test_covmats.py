import pathlib
import pandas as pd

from super_net.covmats import sqrt_covmat_jax, dataset_inputs_covmat_from_systematics

from super_net.api import API as SuperNetAPI
from super_net.tests.conftest import TEST_DATASETS

from super_net.commondata_utils import experimental_commondata_tuple

import jax
import jax.numpy as jnp

from numpy.testing import assert_allclose

TEST_COVMATS_FOLDER = pathlib.Path(__file__).with_name('test_covmats')

def test_sqrt_covmat_jax():
    """
    Test that sqrt_covmat_jax actually computes the square root of a matrix.
    """

    # Is this okay?
    jax.config.update("jax_enable_x64", True)

    test_matrix = jnp.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])

    # This matrix has square root [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
    sqrt_matrix = sqrt_covmat_jax(test_matrix)
    actual_sqrt = jnp.array([[2, 0, 0], [6, 1, 0], [-8, 5, 3]])
    assert_allclose(sqrt_matrix, actual_sqrt, rtol=1e-6)

def test_dataset_inputs_covmat_from_systematics():
    """
    Test that dataset_inputs_covmat_from_systematics correctly produces the covariance matrix.
    """

    data = SuperNetAPI.data(**TEST_DATASETS)
    exp_tuple = experimental_commondata_tuple(data)
    result = dataset_inputs_covmat_from_systematics(data, exp_tuple)

    # Check result is a JAX array
    assert isinstance(result, jnp.ndarray)

    # Check result is correct size
    # TODO

    # Check result is symmetric and positive definite (i.e. all evalues are positive)
    assert_allclose(result, result.T)
    assert jnp.all(jnp.linalg.eigvals(result) > 0)

    # Check result is correct for given datasets
    path = TEST_COVMATS_FOLDER/'test_ds_inputs_covmat.csv'
    assert_allclose(
        result,
        pd.read_csv(path).to_numpy(dtype=float),
    )    
