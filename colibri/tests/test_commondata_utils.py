"""
colibri.tests.test_commondata_utils

Module for testing commondata_utils functions in the colibri package.
"""

import jax.numpy as jnp
import jax.scipy.linalg as jla
import pandas as pd
from nnpdf_data.coredata import CommonData
from numpy.testing import assert_allclose

from colibri.api import API as colibriAPI
from colibri.commondata_utils import CentralCovmatIndex, experimental_commondata_tuple
from colibri.tests.conftest import (
    CLOSURE_TEST_PDFSET,
    PSEUDODATA_SEED,
    T0_PDFSET,
    TEST_DATASETS,
    TEST_COMMONDATA_FOLDER,
)


def test_experimental_commondata_tuple():
    """
    Test that experimental_commondata_tuple returns a tuple of CommonData objects.
    """

    data = colibriAPI.data(**TEST_DATASETS)
    result = experimental_commondata_tuple(data)

    # Check that experimental_commondata_tuple produces a tuple
    assert isinstance(result, tuple)

    # Check that experimental_commondata_tuple produces a tuple of CommonData objects
    for commondata_instance in result:
        assert isinstance(commondata_instance, CommonData)

    # Test that the correct values have been loaded
    for i in range(len(result)):
        path = TEST_COMMONDATA_FOLDER / (
            TEST_DATASETS["dataset_inputs"][i]["dataset"] + "_commondata.csv"
        )

        assert_allclose(
            result[i].commondata_table.iloc[:, 1:]["data"].values,
            pd.read_csv(path).iloc[:, 1:]["data"].values,
        )


def test_central_covmat_index():
    """
    Test that CentralCovmatIndex object is produced correctly.
    """

    result = colibriAPI.central_covmat_index(**{**TEST_DATASETS, **T0_PDFSET})
    # Check that central_covmat_index produces a CentralCovmatIndex object
    assert isinstance(result, CentralCovmatIndex)

    # Check that CentralCovmatIndex has the required attributes, of the correct types
    assert hasattr(result, "central_values")
    assert isinstance(result.central_values, jnp.ndarray)
    assert hasattr(result, "covmat")
    assert isinstance(result.covmat, jnp.ndarray)
    assert hasattr(result, "central_values_idx")
    assert isinstance(result.central_values_idx, jnp.ndarray)

    # Check that the to_dict method works as expected
    result_dict = result.to_dict()
    assert isinstance(result_dict, dict)

    # Check that dimensions of attributes are correct
    assert result.central_values.shape[0] == result.covmat.shape[0]
    assert result.central_values_idx.shape[0] == result.central_values.shape[0]


def test_level0_commondata_tuple():
    """
    Regression test, testing that the generation of Level 0
    data is consistent with main.
    Note that level0 data is generated using jax.config.update("jax_enable_x64", True).
    """

    reference_level0_commondata = pd.read_csv(
        TEST_COMMONDATA_FOLDER
        / "NMC_NC_NOTFIXED_P_EM-SIGMARED_level0_central_values.csv"
    )

    current_level0_commondata = colibriAPI.level_0_commondata_tuple(
        **{**TEST_DATASETS, **CLOSURE_TEST_PDFSET}
    )

    assert_allclose(
        reference_level0_commondata["data"].values,
        current_level0_commondata[0].central_values,
    )


def test_level1_commondata_tuple():
    """
    Regression test, testing that the generation of Level 1
    data is consistent with main.
    Note that level1 data is generated using jax.config.update("jax_enable_x64", True)
    """
    reference_level1_central_values = pd.read_csv(
        TEST_COMMONDATA_FOLDER
        / "NMC_NC_NOTFIXED_P_EM-SIGMARED_level1_central_values.csv"
    )

    current_level1_central_values = colibriAPI.level_1_commondata_tuple(
        **{**TEST_DATASETS, **CLOSURE_TEST_PDFSET, "level_1_seed": PSEUDODATA_SEED}
    )

    assert_allclose(
        reference_level1_central_values["data"].values,
        current_level1_central_values[0].central_values,
    )


def test_central_inv_covmat_index():
    """
    Tests that the central_inv_covmat_index object is produced correctly.
    """
    cci = colibriAPI.central_covmat_index(**{**TEST_DATASETS, **T0_PDFSET})

    cici = colibriAPI.central_inv_covmat_index(**{**TEST_DATASETS, **T0_PDFSET})

    # check that central_inv_covmat_index computes inverse covariance matrix correctly
    assert_allclose(cici.inv_covmat, jla.inv(cci.covmat))

    # check that central values and indices are the same
    assert_allclose(cici.central_values, cci.central_values)
    assert_allclose(cici.central_values_idx, cci.central_values_idx)

    # check that the to_dict method works as expected
    cici_dict = cici.to_dict()
    assert isinstance(cici_dict, dict)
