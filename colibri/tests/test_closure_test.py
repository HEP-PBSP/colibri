import pytest
import jax.numpy as jnp
from unittest.mock import MagicMock
from unittest.mock import patch
from colibri.closure_test import (
    closure_test_pdf_grid,
    closure_test_central_pdf_grid,
    closure_test_colibri_model_pdf,
)


@pytest.fixture
def sample_xgrid():
    return jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])


@pytest.fixture
def mock_pdf():
    pdf = MagicMock()
    pdf.grid_values = jnp.ones((1, 13, 5, 1))
    return pdf


@pytest.fixture
def mock_colibri_model():
    model = MagicMock()
    model.grid_values_func = MagicMock(
        return_value=lambda params: jnp.array(
            [[p * x for x in range(1, 6)] for p in params]
        )
    )
    return model


@patch("colibri.closure_test.pdf_model_from_colibri_model")
def test_closure_test_pdf_grid_with_colibri_model(
    mock_pdf_model_from_colibri_model, sample_xgrid, mock_colibri_model
):
    # Mock the pdf model
    mock_pdf_model_from_colibri_model.return_value = mock_colibri_model

    settings = {"parameters": [1, 2, 3], "model": "test_model"}
    grid = closure_test_pdf_grid(
        "colibri_model", sample_xgrid, closure_test_model_settings=settings
    )
    assert grid.shape == (1, 3, 5)
    assert jnp.array_equal(grid[0][0], jnp.array([1, 2, 3, 4, 5]))


@patch("validphys.convolution.evolution.grid_values")
def test_closure_test_pdf_grid_with_pdf_object(
    mock_grid_values, sample_xgrid, mock_pdf
):
    mock_grid_values.return_value = mock_pdf.grid_values
    grid = closure_test_pdf_grid(mock_pdf, sample_xgrid)
    assert grid.shape == (1, 13, 5)
    assert jnp.all(grid == 1)


def test_closure_test_central_pdf_grid():
    grid = jnp.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    central = closure_test_central_pdf_grid(grid)
    assert central.shape == (2, 3)
    assert jnp.array_equal(central, jnp.array([[1, 2, 3], [4, 5, 6]]))


@patch("colibri.closure_test.pdf_model_from_colibri_model")
def test_closure_test_colibri_model_pdf(
    mock_pdf_model_from_colibri_model, sample_xgrid, mock_colibri_model
):
    # Mock the pdf model
    mock_pdf_model_from_colibri_model.return_value = mock_colibri_model
    settings = {"parameters": [1, 2, 3], "model": "test_model"}
    pdf_grid = closure_test_colibri_model_pdf(settings, sample_xgrid)
    assert pdf_grid.shape == (3, 5)
    assert jnp.array_equal(pdf_grid[0], jnp.array([1, 2, 3, 4, 5]))
