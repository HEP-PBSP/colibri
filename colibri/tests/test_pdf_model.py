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


def test_n_parameters():
    """
    Tests that n_parameters returns correct number.
    """
    assert model.n_parameters == 2


def test_grid_values_func():
    """
    Tests that the grid_values_func returns the correct values.
    """
    func = model.grid_values_func(TEST_XGRID)
    params = jnp.array([2, 3])

    expected_output = sum([param * TEST_PDF_GRID for param in params])

    assert_array_equal(func(params), expected_output)


# ---------------------------------------------------------------------------
# FixedPDFModel and the fixed_pdf_model provider
# ---------------------------------------------------------------------------

import pathlib
from unittest.mock import Mock, patch

import dill
import numpy as np
import pytest

from colibri.core import IntegrabilitySettings
from colibri.pdf_model import (
    FixedPDFModel,
    fixed_pdf_model,
    no_pdf_model,
)

FIXED_PDF_NAME = "NNPDF40_nlo_as_01180"


@patch("colibri.utils.lhapdf_grid_values")
def test_fixed_pdf_model(mock_grid):
    # 6 members; member m is filled with the value m
    mock_grid.return_value = jnp.arange(6.0)[:, None, None] * jnp.ones((6, 14, 3))
    model = FixedPDFModel(FIXED_PDF_NAME, Q0=1.65, member=4)

    assert model.param_names == []
    assert model.n_parameters == 0

    func = model.grid_values_func(TEST_XGRID)
    (pdf, xgrid, Q0), _ = mock_grid.call_args
    assert (pdf.name, Q0) == (FIXED_PDF_NAME, 1.65)
    assert xgrid is TEST_XGRID
    # Member 4 is selected, computed once, params ignored
    assert_array_equal(func(jnp.array([])), 4 * np.ones((14, 3)))
    assert_array_equal(func(jnp.array([1.0, 2.0])), 4 * np.ones((14, 3)))
    assert mock_grid.call_count == 1


def test_export_grids_flag():
    from colibri.tests.conftest import TestPDFModel

    assert FixedPDFModel(FIXED_PDF_NAME, Q0=1.65).export_grids is False
    assert TestPDFModel(n_parameters=0).export_grids is True


def test_fixed_pdf_model_pickles():
    model = FixedPDFModel(FIXED_PDF_NAME, Q0=1.65, member=3)
    loaded = dill.loads(dill.dumps(model))
    assert vars(loaded) == vars(model)


def test_no_pdf_model():
    assert no_pdf_model() is None


def _mock_pdf(error_type="replicas", n_members=101):
    pdf = Mock()
    pdf.name = FIXED_PDF_NAME
    pdf.error_type = error_type
    pdf.info = {"NumMembers": n_members}
    return pdf


def test_fixed_pdf_model_provider_central(tmp_path):
    """Without -rep the central member is used, for any error type."""
    model = fixed_pdf_model(_mock_pdf("hessian"), 1.65, tmp_path)

    assert isinstance(model, FixedPDFModel)
    assert (model.pdf_name, model.Q0, model.member) == (FIXED_PDF_NAME, 1.65, 0)
    with open(tmp_path / "pdf_model.pkl", "rb") as f:
        assert vars(dill.load(f)) == vars(model)


def test_fixed_pdf_model_provider_replica(tmp_path):
    model = fixed_pdf_model(_mock_pdf(), 1.65, tmp_path, replica_index=7)
    assert model.member == 7


@pytest.mark.parametrize(
    "pdf, kwargs, match",
    [
        (_mock_pdf("hessian"), {"replica_index": 1}, "Monte Carlo"),
        (_mock_pdf(n_members=11), {"replica_index": 20}, "not available"),
    ],
)
def test_fixed_pdf_model_provider_errors(pdf, kwargs, match):
    with pytest.raises(ValueError, match=match):
        fixed_pdf_model(pdf, 1.65, pathlib.Path("."), dump_model=False, **kwargs)


@patch("colibri.pdf_model.log.warning")
def test_fixed_pdf_model_provider_warns_on_penalties(mock_warning):
    fixed_pdf_model(
        _mock_pdf(),
        1.65,
        pathlib.Path("."),
        positivity_penalty_settings={"positivity_penalty": True},
        integrability_settings=IntegrabilitySettings(True, {}),
        dump_model=False,
    )
    assert mock_warning.call_count == 2


@patch("colibri.pdf_model.log.warning")
def test_fixed_pdf_model_provider_no_warning_without_penalties(mock_warning):
    fixed_pdf_model(
        _mock_pdf(),
        1.65,
        pathlib.Path("."),
        positivity_penalty_settings={"positivity_penalty": False},
        integrability_settings=IntegrabilitySettings(False, {}),
        dump_model=False,
    )
    mock_warning.assert_not_called()
