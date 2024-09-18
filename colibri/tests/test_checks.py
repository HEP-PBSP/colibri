from colibri.checks import check_pdf_models_equal
from unittest.mock import patch, MagicMock, mock_open
import pytest


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
    prior_settings = {
        "type": "prior_from_gauss_posterior",
        "prior_fit": "fit1",
    }
    pdf_model = "model1"

    theoryid = MagicMock()
    theoryid.id = 123

    t0pdfset = MagicMock()
    t0pdfset.name = "t0pdfset1"

    # Configure mock behavior
    mock_pdf_models_equal.side_effect = lambda x, y: x == y

    # Act
    check_pdf_models_equal.__wrapped__(prior_settings, pdf_model, theoryid, t0pdfset)


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
    prior_settings = {
        "type": "prior_from_gauss_posterior",
        "prior_fit": "fit1",
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
        "type": "prior_from_gauss_posterior",
        "prior_fit": "fit1",
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
