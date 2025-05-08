"""
colibri.tests.test_pdf_model

Tests for the PDFModel class.
"""

import jax.numpy as jnp
from numpy.testing import assert_array_equal
from colibri.tests.conftest import (
    TestPDFModel,
    TEST_FORWARD_MAP_DIS,
    TEST_FK_ARRAYS,
    TEST_XGRID,
    TEST_PDF_GRID,
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


def test_pred_and_pdf_func():
    """
    Tests that the pred_and_pdf_func returns the correct values.
    """
    pred_and_pdf = model.pred_and_pdf_func(TEST_XGRID, TEST_FORWARD_MAP_DIS)

    params = jnp.array([2, 3])
    predictions, pdf = pred_and_pdf(params, TEST_FK_ARRAYS[0])

    expected_predictions = jnp.einsum("ijk,jk->i", TEST_FK_ARRAYS[0], pdf)

    assert jnp.allclose(predictions, expected_predictions)
