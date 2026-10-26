"""
colibri.tests.test_forward_map

Tests for the ForwardMap abstract base class, FKTableForwardMap, SMEFTForwardMap
and the forward map provider functions.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from unittest.mock import Mock
from numpy.testing import assert_array_almost_equal

from colibri.checks import check_pdf_model_is_linear
from colibri.forward_map import (
    ForwardMap,
    FKTableForwardMap,
    SMEFTForwardMap,
    fktable_forward_map,
    smeft_forward_map,
)
from colibri.smeft import SMEFTCorrections
from colibri.tests.conftest import (
    TEST_FK_ARRAYS,
    TEST_PDF_GRID,
    TEST_N_DATA,
    TEST_N_FL,
    TEST_N_XGRID,
    TEST_XGRID,
    MOCK_PDF_MODEL,
)

# colibri runs in float64; the linearity check fails with float32 round-off
jax.config.update("jax_enable_x64", True)

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


def _mock_pdf_model(param_names):
    """Create a mock pdf_model with the given param_names."""
    model = Mock()
    model.param_names = param_names
    return model


def test_forward_map_cannot_be_instantiated():
    """ForwardMap is abstract; direct instantiation must raise TypeError."""
    with pytest.raises(TypeError):
        ForwardMap(pdf_model=_mock_pdf_model(["a", "b"]))


def test_forward_map_subclass_without_call_cannot_be_instantiated():
    """A subclass that does not implement __call__ must also raise TypeError."""

    class NoCallSubclass(ForwardMap):
        pass

    with pytest.raises(TypeError):
        NoCallSubclass(pdf_model=_mock_pdf_model(["a", "b"]))


def test_forward_map_abstract_call_raises_not_implemented():
    """Calling super().__call__() must hit the raise NotImplementedError body."""

    class SuperCallingForwardMap(ForwardMap):
        def __call__(self, fk_tables, params):
            return super().__call__(fk_tables, params)

    fm = SuperCallingForwardMap(pdf_model=_mock_pdf_model(["a", "b"]))
    with pytest.raises(NotImplementedError):
        fm(TEST_FK_ARRAYS, jnp.array([1.0, 2.0]))


def test_forward_map_none_pdf_model_sets_empty_param_names():
    """When pdf_model is None, pdf_param_names must be set to an empty list."""

    class MinimalForwardMap(ForwardMap):
        def __call__(self, fk_tables, params):
            return None

    fm = MinimalForwardMap(pdf_model=None)
    assert fm.pdf_param_names == []
    assert fm.n_pdf_params == 0
    assert fm.pdf_model is None


def test_forward_map_subclass_stores_pdf_param_names():
    """pdf_param_names passed to super().__init__ must be stored on the instance."""

    class MinimalForwardMap(ForwardMap):
        def __call__(self, fk_tables, params):
            return None

    mock_model = _mock_pdf_model(["p0", "p1", "p2", "p3", "p4"])
    fm = MinimalForwardMap(pdf_model=mock_model)
    assert fm.pdf_param_names == ["p0", "p1", "p2", "p3", "p4"]
    assert fm.pdf_model is mock_model
    assert fm.n_pdf_params == 5


def test_forward_map_extra_param_names_default():
    """extra_param_names defaults to an empty tuple."""

    class MinimalForwardMap(ForwardMap):
        def __call__(self, fk_tables, params):
            return None

    fm = MinimalForwardMap(pdf_model=_mock_pdf_model(["a", "b"]))
    assert list(fm.extra_param_names) == []
    assert fm.param_names == ["a", "b"]


def test_forward_map_extra_param_names():
    """extra_param_names are stored and appear in param_names after pdf_param_names."""

    class MinimalForwardMap(ForwardMap):
        def __call__(self, fk_tables, params):
            return None

    fm = MinimalForwardMap(
        pdf_model=_mock_pdf_model(["a", "b"]), extra_param_names=["norm", "scale"]
    )
    assert list(fm.extra_param_names) == ["norm", "scale"]
    assert fm.param_names == ["a", "b", "norm", "scale"]
    assert fm.n_pdf_params == 2


# ---------------------------------------------------------------------------
# FKTableForwardMap.__init__
# ---------------------------------------------------------------------------


def test_fktable_forward_map_stores_pdf_param_names():
    """FKTableForwardMap.__init__ must store pdf_param_names via the base class."""
    pdf_grid_func = _make_pdf_grid_func(TEST_PDF_GRID)
    fm = FKTableForwardMap(
        pred_func=_simple_pred_func,
        pdf_model=_mock_pdf_model(["a", "b", "c"]),
        pdf_grid_func=pdf_grid_func,
    )
    assert fm.pdf_param_names == ["a", "b", "c"]
    assert fm.n_pdf_params == 3


def test_fktable_forward_map_extra_param_names():
    """FKTableForwardMap must accept and store extra_param_names."""
    pdf_grid_func = _make_pdf_grid_func(TEST_PDF_GRID)
    fm = FKTableForwardMap(
        pred_func=_simple_pred_func,
        pdf_model=_mock_pdf_model(["a", "b"]),
        pdf_grid_func=pdf_grid_func,
        extra_param_names=["norm"],
    )
    assert fm.param_names == ["a", "b", "norm"]
    assert fm.n_pdf_params == 2
    assert list(fm.extra_param_names) == ["norm"]


def test_fktable_forward_map_stores_pred_func():
    """FKTableForwardMap.__init__ must store the pred_func."""
    pdf_grid_func = _make_pdf_grid_func(TEST_PDF_GRID)
    fm = FKTableForwardMap(
        pred_func=_simple_pred_func,
        pdf_model=_mock_pdf_model(["a", "b", "c"]),
        pdf_grid_func=pdf_grid_func,
    )
    assert fm._pred_func is _simple_pred_func


def test_fktable_forward_map_stores_pdf_grid_func():
    """FKTableForwardMap.__init__ must store the pdf_grid_func."""
    pdf_grid_func = _make_pdf_grid_func(TEST_PDF_GRID)
    fm = FKTableForwardMap(
        pred_func=_simple_pred_func,
        pdf_model=_mock_pdf_model(["a", "b"]),
        pdf_grid_func=pdf_grid_func,
    )
    assert fm._pdf_grid_func is pdf_grid_func


# ---------------------------------------------------------------------------
# FKTableForwardMap.__call__
# ---------------------------------------------------------------------------


def test_fktable_forward_map_returns_tuple():
    """__call__ must return a 2-tuple (predictions, pdf)."""
    pdf_grid_func = _make_pdf_grid_func(TEST_PDF_GRID)
    fm = FKTableForwardMap(
        pred_func=_simple_pred_func,
        pdf_model=_mock_pdf_model(["a", "b"]),
        pdf_grid_func=pdf_grid_func,
    )
    params = jnp.array([1.0, 2.0])

    result = fm(TEST_FK_ARRAYS, params)

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_fktable_forward_map_predictions_shape():
    """Predictions returned by __call__ must have shape (N_data,)."""
    pdf_grid_func = _make_pdf_grid_func(TEST_PDF_GRID)
    fm = FKTableForwardMap(
        pred_func=_simple_pred_func,
        pdf_model=_mock_pdf_model(["a", "b"]),
        pdf_grid_func=pdf_grid_func,
    )
    params = jnp.array([1.0, 2.0])

    predictions, _ = fm(TEST_FK_ARRAYS, params)

    assert predictions.shape == (TEST_N_DATA,)


def test_fktable_forward_map_pdf_shape():
    """PDF returned by __call__ must have shape (N_fl, N_x)."""
    pdf_grid_func = _make_pdf_grid_func(TEST_PDF_GRID)
    fm = FKTableForwardMap(
        pred_func=_simple_pred_func,
        pdf_model=_mock_pdf_model(["a", "b"]),
        pdf_grid_func=pdf_grid_func,
    )
    params = jnp.array([1.0, 2.0])

    _, pdf = fm(TEST_FK_ARRAYS, params)

    assert pdf.shape == (TEST_N_FL, TEST_N_XGRID)


def test_fktable_forward_map_slices_pdf_params():
    """
    __call__ must pass only params[:n_pdf_params] to pdf_grid_func; extra
    parameters appended to params must not affect the PDF or predictions.
    """
    pdf_grid_func = _make_pdf_grid_func(TEST_PDF_GRID)
    fm = FKTableForwardMap(
        pred_func=_simple_pred_func,
        pdf_model=_mock_pdf_model(["a", "b"]),
        pdf_grid_func=pdf_grid_func,
    )

    pdf_params = jnp.array([1.0, 2.0])
    extra_params = jnp.array([99.0, -99.0])  # should be ignored

    params_no_extra = pdf_params
    params_with_extra = jnp.concatenate([pdf_params, extra_params])

    preds_no_extra, pdf_no_extra = fm(TEST_FK_ARRAYS, params_no_extra)
    preds_with_extra, pdf_with_extra = fm(TEST_FK_ARRAYS, params_with_extra)

    assert_array_almost_equal(preds_no_extra, preds_with_extra)
    assert_array_almost_equal(pdf_no_extra, pdf_with_extra)


def test_fktable_forward_map_uses_pdf_grid_func():
    """
    __call__ must feed the pdf returned by the bound pdf_grid_func into pred_func.
    We verify this by constructing two forward maps with differently scaled pdf_grid_funcs.
    """
    scale = 3.0
    base_pdf_grid_func = _make_pdf_grid_func(TEST_PDF_GRID)
    scaled_pdf_grid_func = lambda p: scale * base_pdf_grid_func(p)  # noqa: E731

    fm_base = FKTableForwardMap(
        pred_func=_simple_pred_func,
        pdf_model=_mock_pdf_model(["a", "b"]),
        pdf_grid_func=base_pdf_grid_func,
    )
    fm_scaled = FKTableForwardMap(
        pred_func=_simple_pred_func,
        pdf_model=_mock_pdf_model(["a", "b"]),
        pdf_grid_func=scaled_pdf_grid_func,
    )

    params = jnp.array([1.0, 2.0])
    preds_base, _ = fm_base(TEST_FK_ARRAYS, params)
    preds_scaled, _ = fm_scaled(TEST_FK_ARRAYS, params)

    np.testing.assert_allclose(preds_scaled, scale * preds_base, rtol=1e-5)


def test_fktable_forward_map_correct_values():
    """
    __call__ must produce predictions equal to pred_func(pdf_grid_func(params), fk).
    """
    pdf_grid_func = _make_pdf_grid_func(TEST_PDF_GRID)
    fm = FKTableForwardMap(
        pred_func=_simple_pred_func,
        pdf_model=_mock_pdf_model(["a", "b"]),
        pdf_grid_func=pdf_grid_func,
    )

    params = jnp.array([1.0, 2.0])
    predictions, pdf = fm(TEST_FK_ARRAYS, params)

    expected_pdf = pdf_grid_func(params)
    expected_preds = _simple_pred_func(expected_pdf, TEST_FK_ARRAYS)

    assert_array_almost_equal(predictions, expected_preds)
    assert_array_almost_equal(pdf, expected_pdf)


# ---------------------------------------------------------------------------
# fktable_forward_map provider function
# ---------------------------------------------------------------------------


def test_forward_map_provider_returns_fktable_forward_map():
    """fktable_forward_map() must return an FKTableForwardMap instance."""
    result = fktable_forward_map(
        _pred_data=_simple_pred_func, pdf_model=MOCK_PDF_MODEL, FIT_XGRID=TEST_XGRID
    )
    assert isinstance(result, FKTableForwardMap)


def test_forward_map_provider_infers_pdf_param_names():
    """
    fktable_forward_map() must set pdf_param_names equal to pdf_model.param_names,
    and n_pdf_params must equal len(pdf_model.param_names).
    """
    result = fktable_forward_map(
        _pred_data=_simple_pred_func, pdf_model=MOCK_PDF_MODEL, FIT_XGRID=TEST_XGRID
    )
    assert result.pdf_param_names == MOCK_PDF_MODEL.param_names
    assert result.n_pdf_params == len(MOCK_PDF_MODEL.param_names)


def test_forward_map_provider_stores_pred_func():
    """fktable_forward_map() must wire _pred_data into the FKTableForwardMap."""
    result = fktable_forward_map(
        _pred_data=_simple_pred_func, pdf_model=MOCK_PDF_MODEL, FIT_XGRID=TEST_XGRID
    )
    assert result._pred_func is _simple_pred_func


def test_forward_map_provider_binds_pdf_grid_func():
    """fktable_forward_map() must call pdf_model.grid_values_func(FIT_XGRID) and store the result."""
    result = fktable_forward_map(
        _pred_data=_simple_pred_func, pdf_model=MOCK_PDF_MODEL, FIT_XGRID=TEST_XGRID
    )
    assert callable(result._pdf_grid_func)


def test_forward_map_provider_functional():
    """
    The FKTableForwardMap built by fktable_forward_map() must produce correct results
    when called.
    """
    fm = fktable_forward_map(
        _pred_data=_simple_pred_func, pdf_model=MOCK_PDF_MODEL, FIT_XGRID=TEST_XGRID
    )
    params = jnp.array([1.0, 2.0])

    predictions, pdf = fm(TEST_FK_ARRAYS, params)

    assert predictions.shape == (TEST_N_DATA,)
    assert pdf.shape == (TEST_N_FL, TEST_N_XGRID)


def test_forward_map_provider_with_different_param_counts():
    """
    fktable_forward_map() must correctly handle pdf_models with different numbers of
    parameters.
    """
    for n in [1, 3, 7]:
        mock_model = Mock()
        mock_model.param_names = [f"p_{i}" for i in range(n)]
        fm = fktable_forward_map(
            _pred_data=_simple_pred_func, pdf_model=mock_model, FIT_XGRID=TEST_XGRID
        )
        assert fm.pdf_param_names == mock_model.param_names
        assert fm.n_pdf_params == n


def test_forward_map_provider_with_extra_param_names():
    """
    fktable_forward_map() must forward extra_param_names to FKTableForwardMap
    and expose them via param_names.
    """
    extra = ["norm", "scale"]
    fm = fktable_forward_map(
        _pred_data=_simple_pred_func,
        pdf_model=MOCK_PDF_MODEL,
        FIT_XGRID=TEST_XGRID,
        extra_param_names=extra,
    )
    assert fm.param_names == MOCK_PDF_MODEL.param_names + extra
    assert list(fm.extra_param_names) == extra


# ---------------------------------------------------------------------------
# SMEFTForwardMap
# ---------------------------------------------------------------------------


def _smeft_corrections(linear, scales=None, quadratic=None, names=None):
    linear = np.asarray(linear, dtype=float)
    n_params = linear.shape[0]
    return SMEFTCorrections(
        parameter_names=tuple(names or [f"c{i}" for i in range(n_params)]),
        scales=np.ones(n_params) if scales is None else np.asarray(scales),
        linear=linear,
        quadratic=quadratic,
    )


def _smeft_fm(corrections, pdf_param_names=("a", "b")):
    return SMEFTForwardMap(
        pred_func=_simple_pred_func,
        pdf_model=_mock_pdf_model(list(pdf_param_names)),
        pdf_grid_func=_make_pdf_grid_func(TEST_PDF_GRID),
        smeft_corrections=corrections,
    )


def _sm_predictions():
    return _simple_pred_func(TEST_PDF_GRID, TEST_FK_ARRAYS)


def test_smeft_forward_map_param_names():
    """SMEFT parameters are appended to the PDF parameters as extra parameters."""
    linear = np.zeros((2, TEST_N_DATA))
    fm = _smeft_fm(_smeft_corrections(linear, names=["OtG", "Y"]))
    assert fm.param_names == ["a", "b", "OtG", "Y"]
    assert fm.n_pdf_params == 2
    assert list(fm.extra_param_names) == ["OtG", "Y"]


def test_smeft_forward_map_zero_coefficients_give_sm():
    """With all Wilson coefficients at zero the SM predictions are recovered."""
    rng = np.random.default_rng(0)
    linear = rng.normal(size=(2, TEST_N_DATA))
    quadratic = rng.normal(size=(2, 2, TEST_N_DATA))
    fm = _smeft_fm(_smeft_corrections(linear, quadratic=quadratic))

    predictions, pdf = fm(TEST_FK_ARRAYS, jnp.array([1.0, 2.0, 0.0, 0.0]))

    assert_array_almost_equal(predictions, _sm_predictions())
    assert_array_almost_equal(pdf, TEST_PDF_GRID)


def test_smeft_forward_map_linear_correction():
    """Predictions are T_SM * (1 + sum_i c_i L_i)."""
    rng = np.random.default_rng(1)
    linear = rng.normal(size=(2, TEST_N_DATA))
    fm = _smeft_fm(_smeft_corrections(linear))
    c = np.array([0.3, -1.2])

    predictions, _ = fm(TEST_FK_ARRAYS, jnp.concatenate([jnp.array([1.0, 2.0]), c]))

    expected = _sm_predictions() * (1 + c @ linear)
    np.testing.assert_allclose(predictions, expected, rtol=1e-6)


def test_smeft_forward_map_quadratic_correction():
    """Predictions are T_SM * (1 + c.L + c.Q.c) when quadratic responses exist."""
    rng = np.random.default_rng(2)
    linear = rng.normal(size=(2, TEST_N_DATA))
    quadratic = rng.normal(size=(2, 2, TEST_N_DATA))
    fm = _smeft_fm(_smeft_corrections(linear, quadratic=quadratic))
    c = np.array([0.5, 2.0])

    predictions, _ = fm(TEST_FK_ARRAYS, jnp.concatenate([jnp.array([1.0, 2.0]), c]))

    quad = np.einsum("i,j,ijn->n", c, c, quadratic)
    expected = _sm_predictions() * (1 + c @ linear + quad)
    np.testing.assert_allclose(predictions, expected, rtol=1e-6)


def test_smeft_forward_map_scales():
    """Fit parameters are scale * c: dividing by the scale recovers the physics."""
    rng = np.random.default_rng(3)
    linear = rng.normal(size=(2, TEST_N_DATA))
    scales = np.array([50.0, 200.0])
    c = np.array([0.01, -0.002])

    fm_scaled = _smeft_fm(_smeft_corrections(linear, scales=scales))
    fm_unscaled = _smeft_fm(_smeft_corrections(linear))
    pdf_params = jnp.array([1.0, 2.0])

    preds_scaled, _ = fm_scaled(
        TEST_FK_ARRAYS, jnp.concatenate([pdf_params, scales * c])
    )
    preds_unscaled, _ = fm_unscaled(TEST_FK_ARRAYS, jnp.concatenate([pdf_params, c]))

    np.testing.assert_allclose(preds_scaled, preds_unscaled, rtol=1e-6)
    np.testing.assert_allclose(
        fm_scaled.physical_coefficients(jnp.concatenate([pdf_params, scales * c])),
        c,
        rtol=1e-6,
    )


def test_smeft_forward_map_is_jittable():
    """The SMEFT forward map can be traced by jax.jit with dynamic FK tables."""
    linear = np.ones((1, TEST_N_DATA))
    fm = _smeft_fm(_smeft_corrections(linear))
    params = jnp.array([1.0, 2.0, 0.5])

    jitted, _ = jax.jit(fm)(TEST_FK_ARRAYS, params)
    eager, _ = fm(TEST_FK_ARRAYS, params)

    np.testing.assert_allclose(jitted, eager, rtol=1e-6)


def test_smeft_forward_map_provider():
    """smeft_forward_map() builds an SMEFTForwardMap bound to the PDF model grid."""
    linear = np.zeros((1, TEST_N_DATA))
    fm = smeft_forward_map(
        _pred_data=_simple_pred_func,
        pdf_model=MOCK_PDF_MODEL,
        FIT_XGRID=TEST_XGRID,
        smeft_corrections=_smeft_corrections(linear, names=["OtG"]),
        fast_kernel_arrays=TEST_FK_ARRAYS,
    )
    assert isinstance(fm, SMEFTForwardMap)
    assert fm.param_names == MOCK_PDF_MODEL.param_names + ["OtG"]

    predictions, pdf = fm(TEST_FK_ARRAYS, jnp.array([1.0, 2.0, 0.0]))
    assert predictions.shape == (TEST_N_DATA,)
    assert pdf.shape == (TEST_N_FL, TEST_N_XGRID)


# ---------------------------------------------------------------------------
# SMEFTForwardMap with a fixed PDF (PDF model without parameters)
# ---------------------------------------------------------------------------


def _fixed_smeft_fm(corrections, fk_tables=TEST_FK_ARRAYS):
    return SMEFTForwardMap(
        pred_func=_simple_pred_func,
        pdf_model=_mock_pdf_model([]),
        pdf_grid_func=_make_pdf_grid_func(TEST_PDF_GRID),
        smeft_corrections=corrections,
        fk_tables=fk_tables,
    )


def test_fixed_pdf_smeft_forward_map_precomputes_sm():
    """Without PDF parameters the SM predictions and PDF grid are computed once."""
    fm = _fixed_smeft_fm(_smeft_corrections(np.ones((1, TEST_N_DATA))))

    assert fm.param_names == ["c0"]
    assert fm.n_pdf_params == 0
    assert_array_almost_equal(fm.sm_predictions, _sm_predictions())
    assert_array_almost_equal(fm.pdf_grid, TEST_PDF_GRID)


def test_fitted_pdf_smeft_forward_map_does_not_precompute():
    fm = _smeft_fm(_smeft_corrections(np.ones((1, TEST_N_DATA))))
    assert fm.sm_predictions is None
    assert fm.pdf_grid is None


def test_fixed_pdf_smeft_forward_map_predictions():
    """Predictions are T_SM * (1 + c.L + c.Q.c) with the precomputed T_SM."""
    rng = np.random.default_rng(4)
    linear = rng.normal(size=(2, TEST_N_DATA))
    quadratic = rng.normal(size=(2, 2, TEST_N_DATA))
    scales = np.array([2.0, 5.0])
    fixed = _fixed_smeft_fm(
        _smeft_corrections(linear, scales=scales, quadratic=quadratic)
    )
    params = np.array([0.4, -1.3])

    preds_fixed, pdf_fixed = fixed(TEST_FK_ARRAYS, jnp.array(params))

    c = params / scales
    quad = np.einsum("i,j,ijn->n", c, c, quadratic)
    expected = _sm_predictions() * (1 + c @ linear + quad)
    np.testing.assert_allclose(preds_fixed, expected, rtol=1e-6)
    assert_array_almost_equal(pdf_fixed, TEST_PDF_GRID)


def test_fixed_pdf_smeft_forward_map_ignores_fk_tables():
    fm = _fixed_smeft_fm(_smeft_corrections(np.ones((1, TEST_N_DATA))))
    params = jnp.array([0.5])
    np.testing.assert_allclose(
        fm(None, params)[0], fm(TEST_FK_ARRAYS, params)[0], rtol=1e-12
    )


def test_fixed_pdf_smeft_forward_map_needs_fk_tables():
    with pytest.raises(ValueError, match="fk_tables"):
        _fixed_smeft_fm(_smeft_corrections(np.ones((1, TEST_N_DATA))), fk_tables=None)


def test_fixed_pdf_smeft_forward_map_is_linear():
    """Linear-only K-factors with a fixed PDF pass the analytic-fit linearity check."""
    rng = np.random.default_rng(5)
    fm = _fixed_smeft_fm(_smeft_corrections(rng.normal(size=(2, TEST_N_DATA))))
    check_pdf_model_is_linear(fm, TEST_FK_ARRAYS)


def test_smeft_forward_map_provider_fixed_pdf():
    """With a PDF model without parameters the provider precomputes the SM."""
    model = _mock_pdf_model([])
    model.grid_values_func = lambda xgrid: _make_pdf_grid_func(TEST_PDF_GRID)
    fm = smeft_forward_map(
        _pred_data=_simple_pred_func,
        pdf_model=model,
        FIT_XGRID=TEST_XGRID,
        smeft_corrections=_smeft_corrections(np.zeros((1, TEST_N_DATA))),
        fast_kernel_arrays=TEST_FK_ARRAYS,
    )
    assert isinstance(fm, SMEFTForwardMap)
    assert_array_almost_equal(fm.sm_predictions, _sm_predictions())
