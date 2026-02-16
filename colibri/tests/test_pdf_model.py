"""
colibri.tests.test_pdf_model

Tests for the PDFModel class.
"""

import jax.numpy as jnp
from numpy.testing import assert_array_equal

from colibri.tests.conftest import (
    TEST_FK_ARRAYS,
    TEST_FORWARD_MAP_DIS,
    TEST_PDF_GRID,
    TEST_XGRID,
    TestPDFModel,
)

model = TestPDFModel(n_parameters=2)


def test_param_names():
    """
    Tests that the param_names property returns the correct names.
    """
    assert model.param_names == ["w_1", "w_2"]


def test_grid_values_func():
    """
    Tests that the grid_values_func returns the correct values.
    """
    func = model.grid_values_func(TEST_XGRID)
    params = jnp.array([2, 3])

    expected_output = sum([param * TEST_PDF_GRID for param in params])

    assert_array_equal(func(params), expected_output)
