"""
colibri.tests.test_forward_map

Tests for the ForwardMap abstract base class, FKTableForwardMap, and the
forward_map provider function.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from unittest.mock import Mock
from numpy.testing import assert_array_almost_equal

from colibri.forward_map import ForwardMap, FKTableForwardMap, forward_map
from colibri.tests.conftest import (
    TEST_FK_ARRAYS,
    TEST_PDF_GRID,
    TEST_N_DATA,
    TEST_N_FL,
    TEST_N_XGRID,
    MOCK_PDF_MODEL,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_pred_func(pdf, fk_tables):
    """DIS-like prediction: einsum over the first FK table."""
    return jnp.einsum("ijk,jk->i", fk_tables[0], pdf)


def _make_pdf_grid_func(pdf_grid):
    """Return a callable that ignores params and always returns pdf_grid."""
    return lambda params: pdf_grid


# ---------------------------------------------------------------------------
# ForwardMap (abstract base class)
# ---------------------------------------------------------------------------


def test_forward_map_cannot_be_instantiated():
    """ForwardMap is abstract; direct instantiation must raise TypeError."""
    with pytest.raises(TypeError):
        ForwardMap(pdf_param_names=["a", "b"])


def test_forward_map_subclass_without_call_cannot_be_instantiated():
    """A subclass that does not implement __call__ must also raise TypeError."""

    class NoCallSubclass(ForwardMap):
        pass

    with pytest.raises(TypeError):
        NoCallSubclass(pdf_param_names=["a", "b"])


def test_forward_map_abstract_call_raises_not_implemented():
    """Calling super().__call__() must hit the raise NotImplementedError body."""

    class SuperCallingForwardMap(ForwardMap):
        def __call__(self, pdf_grid_func, fk_tables, params):
            return super().__call__(pdf_grid_func, fk_tables, params)

    fm = SuperCallingForwardMap(pdf_param_names=["a", "b"])
    with pytest.raises(NotImplementedError):
        fm(_make_pdf_grid_func(TEST_PDF_GRID), TEST_FK_ARRAYS, jnp.array([1.0, 2.0]))


def test_forward_map_subclass_stores_pdf_param_names():
    """pdf_param_names passed to super().__init__ must be stored on the instance."""

    class MinimalForwardMap(ForwardMap):
        def __call__(self, pdf_grid_func, fk_tables, params):
            pdf = pdf_grid_func(params[: self.n_pdf_params])
            return _simple_pred_func(pdf, fk_tables), pdf

    fm = MinimalForwardMap(pdf_param_names=["p0", "p1", "p2", "p3", "p4"])
    assert fm.pdf_param_names == ["p0", "p1", "p2", "p3", "p4"]
    assert fm.n_pdf_params == 5


def test_forward_map_extra_param_names_default():
    """extra_param_names defaults to an empty tuple."""

    class MinimalForwardMap(ForwardMap):
        def __call__(self, pdf_grid_func, fk_tables, params):
            return None

    fm = MinimalForwardMap(pdf_param_names=["a", "b"])
    assert list(fm.extra_param_names) == []
    assert fm.param_names == ["a", "b"]


def test_forward_map_extra_param_names():
    """extra_param_names are stored and appear in param_names after pdf_param_names."""

    class MinimalForwardMap(ForwardMap):
        def __call__(self, pdf_grid_func, fk_tables, params):
            return None

    fm = MinimalForwardMap(
        pdf_param_names=["a", "b"], extra_param_names=["norm", "scale"]
    )
    assert list(fm.extra_param_names) == ["norm", "scale"]
    assert fm.param_names == ["a", "b", "norm", "scale"]
    assert fm.n_pdf_params == 2


# ---------------------------------------------------------------------------
# FKTableForwardMap.__init__
# ---------------------------------------------------------------------------


def test_fktable_forward_map_stores_pdf_param_names():
    """FKTableForwardMap.__init__ must store pdf_param_names via the base class."""
    fm = FKTableForwardMap(pred_func=_simple_pred_func, pdf_param_names=["a", "b", "c"])
    assert fm.pdf_param_names == ["a", "b", "c"]
    assert fm.n_pdf_params == 3


def test_fktable_forward_map_extra_param_names():
    """FKTableForwardMap must accept and store extra_param_names."""
    fm = FKTableForwardMap(
        pred_func=_simple_pred_func,
        pdf_param_names=["a", "b"],
        extra_param_names=["norm"],
    )
    assert fm.param_names == ["a", "b", "norm"]
    assert fm.n_pdf_params == 2
    assert list(fm.extra_param_names) == ["norm"]


def test_fktable_forward_map_stores_pred_func():
    """FKTableForwardMap.__init__ must store the pred_func."""
    fm = FKTableForwardMap(pred_func=_simple_pred_func, pdf_param_names=["a", "b", "c"])
    assert fm._pred_func is _simple_pred_func


# ---------------------------------------------------------------------------
# FKTableForwardMap.__call__
# ---------------------------------------------------------------------------


def test_fktable_forward_map_returns_tuple():
    """__call__ must return a 2-tuple (predictions, pdf)."""
    fm = FKTableForwardMap(pred_func=_simple_pred_func, pdf_param_names=["a", "b"])
    pdf_grid_func = _make_pdf_grid_func(TEST_PDF_GRID)
    params = jnp.array([1.0, 2.0])

    result = fm(pdf_grid_func, TEST_FK_ARRAYS, params)

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_fktable_forward_map_predictions_shape():
    """Predictions returned by __call__ must have shape (N_data,)."""
    fm = FKTableForwardMap(pred_func=_simple_pred_func, pdf_param_names=["a", "b"])
    pdf_grid_func = _make_pdf_grid_func(TEST_PDF_GRID)
    params = jnp.array([1.0, 2.0])

    predictions, _ = fm(pdf_grid_func, TEST_FK_ARRAYS, params)

    assert predictions.shape == (TEST_N_DATA,)


def test_fktable_forward_map_pdf_shape():
    """PDF returned by __call__ must have shape (N_fl, N_x)."""
    fm = FKTableForwardMap(pred_func=_simple_pred_func, pdf_param_names=["a", "b"])
    pdf_grid_func = _make_pdf_grid_func(TEST_PDF_GRID)
    params = jnp.array([1.0, 2.0])

    _, pdf = fm(pdf_grid_func, TEST_FK_ARRAYS, params)

    assert pdf.shape == (TEST_N_FL, TEST_N_XGRID)


def test_fktable_forward_map_slices_pdf_params():
    """
    __call__ must pass only params[:n_pdf_params] to pdf_grid_func; extra
    parameters appended to params must not affect the PDF or predictions.
    """
    n_pdf = 2
    fm = FKTableForwardMap(pred_func=_simple_pred_func, pdf_param_names=["a", "b"])
    pdf_grid_func = _make_pdf_grid_func(TEST_PDF_GRID)

    pdf_params = jnp.array([1.0, 2.0])
    extra_params = jnp.array([99.0, -99.0])  # should be ignored

    params_no_extra = pdf_params
    params_with_extra = jnp.concatenate([pdf_params, extra_params])

    preds_no_extra, pdf_no_extra = fm(pdf_grid_func, TEST_FK_ARRAYS, params_no_extra)
    preds_with_extra, pdf_with_extra = fm(
        pdf_grid_func, TEST_FK_ARRAYS, params_with_extra
    )

    assert_array_almost_equal(preds_no_extra, preds_with_extra)
    assert_array_almost_equal(pdf_no_extra, pdf_with_extra)


def test_fktable_forward_map_uses_pdf_grid_func():
    """
    __call__ must feed the pdf returned by pdf_grid_func into pred_func.
    We verify this by using a pdf_grid_func that scales by a known factor.
    """
    scale = 3.0
    n_pdf = 2
    fm = FKTableForwardMap(pred_func=_simple_pred_func, pdf_param_names=["a", "b"])

    params = jnp.array([1.0, 2.0])
    base_pdf_grid_func = _make_pdf_grid_func(TEST_PDF_GRID)
    scaled_pdf_grid_func = lambda p: scale * base_pdf_grid_func(p)  # noqa: E731

    preds_base, _ = fm(base_pdf_grid_func, TEST_FK_ARRAYS, params)
    preds_scaled, _ = fm(scaled_pdf_grid_func, TEST_FK_ARRAYS, params)

    np.testing.assert_allclose(preds_scaled, scale * preds_base, rtol=1e-5)


def test_fktable_forward_map_correct_values():
    """
    __call__ must produce predictions equal to pred_func(pdf_grid_func(params), fk).
    """
    n_pdf = 2
    fm = FKTableForwardMap(pred_func=_simple_pred_func, pdf_param_names=["a", "b"])

    params = jnp.array([1.0, 2.0])
    pdf_grid_func = _make_pdf_grid_func(TEST_PDF_GRID)

    predictions, pdf = fm(pdf_grid_func, TEST_FK_ARRAYS, params)

    expected_pdf = pdf_grid_func(params)
    expected_preds = _simple_pred_func(expected_pdf, TEST_FK_ARRAYS)

    assert_array_almost_equal(predictions, expected_preds)
    assert_array_almost_equal(pdf, expected_pdf)


# ---------------------------------------------------------------------------
# forward_map provider function
# ---------------------------------------------------------------------------


def test_forward_map_provider_returns_fktable_forward_map():
    """forward_map() must return an FKTableForwardMap instance."""
    result = forward_map(_pred_data=_simple_pred_func, pdf_model=MOCK_PDF_MODEL)
    assert isinstance(result, FKTableForwardMap)


def test_forward_map_provider_infers_pdf_param_names():
    """
    forward_map() must set pdf_param_names equal to pdf_model.param_names,
    and n_pdf_params must equal len(pdf_model.param_names).
    """
    result = forward_map(_pred_data=_simple_pred_func, pdf_model=MOCK_PDF_MODEL)
    assert result.pdf_param_names == MOCK_PDF_MODEL.param_names
    assert result.n_pdf_params == len(MOCK_PDF_MODEL.param_names)


def test_forward_map_provider_stores_pred_func():
    """forward_map() must wire _pred_data into the FKTableForwardMap."""
    result = forward_map(_pred_data=_simple_pred_func, pdf_model=MOCK_PDF_MODEL)
    assert result._pred_func is _simple_pred_func


def test_forward_map_provider_functional():
    """
    The FKTableForwardMap built by forward_map() must produce correct results
    when called.
    """
    fm = forward_map(_pred_data=_simple_pred_func, pdf_model=MOCK_PDF_MODEL)
    pdf_grid_func = _make_pdf_grid_func(TEST_PDF_GRID)
    params = jnp.array([1.0, 2.0])

    predictions, pdf = fm(pdf_grid_func, TEST_FK_ARRAYS, params)

    assert predictions.shape == (TEST_N_DATA,)
    assert pdf.shape == (TEST_N_FL, TEST_N_XGRID)


def test_forward_map_provider_with_different_param_counts():
    """
    forward_map() must correctly handle pdf_models with different numbers of
    parameters.
    """
    for n in [1, 3, 7]:
        mock_model = Mock()
        mock_model.param_names = [f"p_{i}" for i in range(n)]
        fm = forward_map(_pred_data=_simple_pred_func, pdf_model=mock_model)
        assert fm.pdf_param_names == mock_model.param_names
        assert fm.n_pdf_params == n


def test_forward_map_provider_with_extra_param_names():
    """
    forward_map() must forward extra_param_names to FKTableForwardMap
    and expose them via param_names.
    """
    extra = ["norm", "scale"]
    fm = forward_map(
        _pred_data=_simple_pred_func,
        pdf_model=MOCK_PDF_MODEL,
        extra_param_names=extra,
    )
    assert fm.param_names == MOCK_PDF_MODEL.param_names + extra
    assert list(fm.extra_param_names) == extra
