"""
colibri.tests.test_checks

Tests for the checks module of the colibri package.
"""

from colibri.checks import check_pdf_models_equal, check_pdf_model_is_linear
from colibri.core import PriorSettings
from unittest.mock import patch, MagicMock, mock_open
import pytest
import jax.numpy as jnp


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="theoryid: 123\nt0pdfset: t0pdfset1",
)
@patch("os.path.exists", return_value=True)
@patch("colibri.checks.get_pdf_model", return_value="model1")
@patch("colibri.checks.pdf_models_equal")
def test_check_pdf_models_equal_true(
    mock_pdf_models_equal, mock_get_pdf_model, mock_exists, mock_open
):
    # Setup
    prior_settings = PriorSettings(
        **{
            "prior_distribution": "prior_from_gauss_posterior",
            "prior_distribution_specs": {"prior_fit": "fit1"},
        }
    )
    pdf_model = "model1"

    theoryid = MagicMock()
    theoryid.id = 123

    # Configure mock behavior
    mock_pdf_models_equal.side_effect = lambda x, y: x == y

    # Act
    check_pdf_models_equal.__wrapped__(prior_settings, pdf_model, theoryid)


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="theoryid: 456\nt0pdfset: t0pdfset1",
)
@patch("os.path.exists", return_value=True)
@patch("colibri.checks.get_pdf_model", return_value="model1")
@patch("colibri.checks.pdf_models_equal")
def test_check_pdf_models_equal_false_theoryid(
    mock_pdf_models_equal, mock_get_pdf_model, mock_exists, mock_open
):
    # Setup
    prior_settings = PriorSettings(
        **{
            "prior_distribution": "prior_from_gauss_posterior",
            "prior_distribution_specs": {"prior_fit": "fit1"},
        }
    )
    pdf_model = "model1"

    theoryid = MagicMock()
    theoryid.id = 123

    t0pdfset = MagicMock()
    t0pdfset.name = "t0pdfset1"

    # Configure mock behavior
    mock_pdf_models_equal.side_effect = lambda x, y: x == y

    with pytest.raises(Exception):
        check_pdf_models_equal.__wrapped__(
            prior_settings, pdf_model, theoryid, t0pdfset
        )


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="theoryid: 123\nt0pdfset: t0pdfset2",
)
@patch("os.path.exists", return_value=True)
@patch("colibri.checks.get_pdf_model", return_value="model1")
@patch("colibri.checks.pdf_models_equal")
def test_check_pdf_models_equal_false_t0pdf(
    mock_pdf_models_equal, mock_get_pdf_model, mock_exists, mock_open
):
    # Setup
    prior_settings = {
        "prior_distribution": "prior_from_gauss_posterior",
        "prior_distribution_specs": {"prior_fit": "fit1"},
    }
    pdf_model = "model1"

    theoryid = MagicMock()
    theoryid.id = 123

    t0pdfset = MagicMock()
    t0pdfset.name = "t0pdfset1"

    # Configure mock behavior
    mock_pdf_models_equal.side_effect = lambda x, y: x == y

    with pytest.raises(Exception):
        check_pdf_models_equal.__wrapped__(
            prior_settings, pdf_model, theoryid, t0pdfset
        )


@patch("colibri.checks.make_pred_data")
@patch("colibri.checks.fast_kernel_arrays")
def test_check_pdf_model_is_linear(mock_fast_kernel_arrays, mock_make_pred_data):
    # Create test data
    FIT_XGRID = jnp.array([1.0, 2.0, 3.0])
    data = jnp.array([0.1, 0.2, 0.3])
    fk = jnp.array([0.3, 0.1, 0.6])

    mock_fast_kernel_arrays.return_value = fk

    # Create a mock for the PDF model
    mock_pdf_model = MagicMock()
    mock_pdf_model.param_names = ["a", "b", "c"]

    # Mock the behavior of pred_and_pdf_func to return a linear model
    def linear_model(params, fk):
        # Simulating a simple linear model: f(x) = a*x + b*y + c*z + 3.0, where params = [a, b, c]
        return (jnp.dot(params, fk) + 3.0, params)

    # Set the mock's pred_and_pdf_func to return the linear_model function
    mock_pdf_model.pred_and_pdf_func.return_value = linear_model

    # Test for linear model (should not raise an exception)
    check_pdf_model_is_linear.__wrapped__(mock_pdf_model, FIT_XGRID, data)

    # Now mock a non-linear model to ensure the ValueError is raised
    def non_linear_model(params, fk):
        # Introduce some non-linearity
        return (jnp.dot(params**2, FIT_XGRID) + fk, params)

    mock_pdf_model.pred_and_pdf_func.return_value = non_linear_model

    # Ensure ValueError is raised for non-linear model
    with pytest.raises(ValueError):
        check_pdf_model_is_linear.__wrapped__(mock_pdf_model, FIT_XGRID, data)
