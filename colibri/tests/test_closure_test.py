"""
colibri.tests.test_closure_test

Module for testing the closure test functions in the colibri package.
"""

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest
import validphys
from validphys import convolution

from colibri.api import API as cAPI
from colibri.closure_test import (
    closure_test_central_pdf_grid,
    closure_test_colibri_model_pdf,
    closure_test_pdf_grid,
)
from colibri.tests.conftest import (
    CLOSURE_TEST_PDFSET,
    MOCK_PDF_MODEL,
    TEST_N_FL,
    TEST_N_XGRID,
    TEST_XGRID,
)


@pytest.fixture
def mock_pdf():
    pdf = MagicMock()
    pdf.grid_values = jnp.ones((1, 13, 5, 1))
    return pdf


def test_closure_test_pdf_grid():
    """
    Test the closure_test_pdf_grid function.

    Verifies:
    - Type of "closure_test_pdf" is validphys.core.PDF
    - Output type is a jnp.array.
    - The output shape is (N_rep, N_fl, N_x)
    """
    cltest_pdf_set = cAPI.closure_test_pdf(**CLOSURE_TEST_PDFSET)

    # Check 1: closure_test_pdf is an instance of validphys.core.PDF
    assert isinstance(cltest_pdf_set, validphys.core.PDF)

    # call the function
    grid = closure_test_pdf_grid(cltest_pdf_set, TEST_XGRID, Q0=1.65)

    # Check 2: type of the output is a jnp.array
    assert isinstance(grid, jnp.ndarray)

    # Check 3: shape of the output
    N_rep = cltest_pdf_set.get_members()  #   number of replicas
    N_fl = len(convolution.FK_FLAVOURS)  # number of flavours

    assert grid.shape == (N_rep, N_fl, TEST_N_XGRID)


@patch("colibri.closure_test.pdf_model_from_colibri_model")
def test_closure_test_pdf_grid_with_colibri_model(mock_pdf_model_from_colibri_model):
    # Mock the pdf model
    mock_pdf_model_from_colibri_model.return_value = MOCK_PDF_MODEL

    settings = {"parameters": [1, 2, 3], "model": "test_model"}
    grid = closure_test_pdf_grid(
        "colibri_model", TEST_XGRID, closure_test_model_settings=settings
    )
    assert grid.shape == (1, TEST_N_FL, TEST_N_XGRID)


@patch("validphys.convolution.evolution.grid_values")
def test_closure_test_pdf_grid_with_pdf_object(mock_grid_values, mock_pdf):
    mock_grid_values.return_value = mock_pdf.grid_values
    grid = closure_test_pdf_grid(mock_pdf, TEST_XGRID)
    assert grid.shape == (1, 13, 5)
    assert jnp.all(grid == 1)


def test_closure_test_central_pdf_grid():
    grid = jnp.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    central = closure_test_central_pdf_grid(grid)
    assert central.shape == (2, 3)
    assert jnp.array_equal(central, jnp.array([[1, 2, 3], [4, 5, 6]]))


@patch("colibri.closure_test.pdf_model_from_colibri_model")
def test_closure_test_colibri_model_pdf(
    mock_pdf_model_from_colibri_model,
):
    # Mock the pdf model
    mock_pdf_model_from_colibri_model.return_value = MOCK_PDF_MODEL
    settings = {"parameters": [1, 2, 3], "model": "test_model"}
    pdf_grid = closure_test_colibri_model_pdf(settings, TEST_XGRID)
    assert pdf_grid.shape == (TEST_N_FL, TEST_N_XGRID)
