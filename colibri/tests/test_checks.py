"""
colibri.tests.test_checks

Tests for the checks module of the colibri package.
"""

from unittest.mock import MagicMock, mock_open, patch

import jax.numpy as jnp
import pytest

from colibri.forward_map import FKTableForwardMap

from colibri.checks import check_pdf_model_is_linear, check_pdf_models_equal
from colibri.core import PriorSettings


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="theoryid: 123\nt0pdfset: t0pdfset1",
)
@patch("os.path.exists", return_value=True)
@patch("colibri.checks.get_pdf_model")
def test_check_pdf_models_equal_true(mock_get_pdf_model, mock_exists, mock_open):
    # Setup
    prior_settings = PriorSettings(
        **{
            "prior_distribution": "prior_from_gauss_posterior",
            "prior_distribution_specs": {"prior_fit": "fit1"},
        }
    )

    # The prior model returned by get_pdf_model must have matching param_names
    mock_prior_model = MagicMock()
    mock_prior_model.param_names = ["param1", "param2"]
    mock_get_pdf_model.return_value = mock_prior_model

    forward_map = MagicMock()
    forward_map.pdf_param_names = ["param1", "param2"]

    theoryid = MagicMock()
    theoryid.id = 123

    # Act — should not raise
    check_pdf_models_equal.__wrapped__(prior_settings, forward_map, theoryid)


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="theoryid: 456\nt0pdfset: t0pdfset1",
)
@patch("os.path.exists", return_value=True)
@patch("colibri.checks.get_pdf_model")
def test_check_pdf_models_equal_false_theoryid(
    mock_get_pdf_model, mock_exists, mock_open
):
    # Setup
    prior_settings = PriorSettings(
        **{
            "prior_distribution": "prior_from_gauss_posterior",
            "prior_distribution_specs": {"prior_fit": "fit1"},
        }
    )

    mock_prior_model = MagicMock()
    mock_prior_model.param_names = ["param1", "param2"]
    mock_get_pdf_model.return_value = mock_prior_model

    forward_map = MagicMock()
    forward_map.pdf_param_names = ["param1", "param2"]

    theoryid = MagicMock()
    theoryid.id = 123

    # Theory ID mismatch (file says 456, fit says 123)
    with pytest.raises(Exception):
        check_pdf_models_equal.__wrapped__(prior_settings, forward_map, theoryid)


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="theoryid: 123\nt0pdfset: t0pdfset2",
)
@patch("os.path.exists", return_value=True)
@patch("colibri.checks.get_pdf_model")
def test_check_pdf_models_equal_false_param_names(
    mock_get_pdf_model, mock_exists, mock_open
):
    # Setup — param names mismatch between prior model and forward_map
    prior_settings = PriorSettings(
        **{
            "prior_distribution": "prior_from_gauss_posterior",
            "prior_distribution_specs": {"prior_fit": "fit1"},
        }
    )

    mock_prior_model = MagicMock()
    mock_prior_model.param_names = ["param1", "param2", "param3"]  # different
    mock_get_pdf_model.return_value = mock_prior_model

    forward_map = MagicMock()
    forward_map.pdf_param_names = ["param1", "param2"]

    theoryid = MagicMock()
    theoryid.id = 123

    with pytest.raises(ValueError):
        check_pdf_models_equal.__wrapped__(prior_settings, forward_map, theoryid)


def test_check_pdf_model_is_linear():
    # Create test data
    FIT_XGRID = jnp.array([1.0, 2.0, 3.0])
    fk = jnp.array([0.3, 0.1, 0.6])

    # Create a mock for the PDF model
    mock_pdf_model = MagicMock()
    mock_pdf_model.param_names = ["a", "b", "c"]

    # Mock the behavior of pdf_grid to return a linear model
    def pdf_linear_model(params):
        return params

    # Set the mock's grid_values_func to return the linear_model function
    mock_pdf_model.grid_values_func.return_value = pdf_linear_model

    forward_map_lin = FKTableForwardMap(
        # Simulating a simple linear model: f(x) = a*x + b*y + c*z + 3.0, where pdf = [a, b, c]
        lambda pdf, fk: jnp.dot(pdf, fk) + 3.0,
        pdf_model=mock_pdf_model,
        pdf_grid_func=mock_pdf_model.grid_values_func(FIT_XGRID),
    )

    # Test for linear model (should not raise an exception)
    check_pdf_model_is_linear(forward_map_lin, fk)

    # Now mock a non-linear model to ensure the ValueError is raised
    non_linear_model = FKTableForwardMap(
        # Introduce some non-linearity
        lambda pdf, fk: jnp.dot(pdf**2, FIT_XGRID) + fk,
        pdf_model=mock_pdf_model,
        pdf_grid_func=mock_pdf_model.grid_values_func(FIT_XGRID),
    )

    # Ensure ValueError is raised for non-linear model
    with pytest.raises(ValueError):
        check_pdf_model_is_linear(non_linear_model, fk)
