import pandas as pd
import pathlib
import jax.numpy as jnp

from super_net.api import API as SuperNetAPI

from super_net.commondata_utils import (
    experimental_commondata_tuple,
    CentralCovmatIndex,
)

from validphys.coredata import CommonData

from super_net.tests.conftest import (
    TEST_DATASETS,
    T0_PDFSET,
)

from numpy.testing import assert_allclose

TEST_COMMONDATA_FOLDER = pathlib.Path(__file__).with_name("test_commondata")


def test_experimental_commondata_tuple():
    """
    Test that experimental_commondata_tuple returns a tuple of CommonData objects.
    """

    data = SuperNetAPI.data(**TEST_DATASETS)
    result = experimental_commondata_tuple(data)

    # Check that experimental_commondata_tuple produces a tuple
    assert isinstance(result, tuple)

    # Check that experimental_commondata_tuple produces a tuple of CommonData objects
    for commondata_instance in result:
        assert isinstance(commondata_instance, CommonData)

    # Test that the correct values have been loaded
    for i in range(len(result)):
        path = TEST_COMMONDATA_FOLDER / (data.datasets[i].name + "_commondata.csv")
        assert_allclose(
            result[i].commondata_table.iloc[:, 1:].to_numpy(dtype=float),
            pd.read_csv(path).iloc[:, 1:].to_numpy(dtype=float),
        )


def test_central_covmat_index():
    """
    Test that CentralCovmatIndex object is produced correctly.
    """

    result = SuperNetAPI.central_covmat_index(**{**TEST_DATASETS, **T0_PDFSET})

    # Check that central_covmat_index produces a CentralCovmatIndex object
    assert isinstance(result, CentralCovmatIndex)

    # Check that CentralCovmatIndex has the required attributes, of the correct types
    assert hasattr(result, "central_values")
    assert isinstance(result.central_values, jnp.ndarray)
    assert hasattr(result, "covmat")
    assert isinstance(result.covmat, jnp.ndarray)
    assert hasattr(result, "central_values_idx")
    assert isinstance(result.central_values_idx, jnp.ndarray)

    # Check that dimensions of attributes are correct
    assert result.central_values.shape[0] == result.covmat.shape[0]
    assert result.central_values_idx.shape[0] == result.central_values.shape[0]
