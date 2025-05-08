"""
colibri.tests.test_theory_predictions.py

Test module for theory_predictions.py
"""

import jax.numpy as jnp
import jaxlib
from numpy.testing import assert_allclose
from validphys.fkparser import load_fktable

from colibri.api import API as colibriAPI
from colibri.tests.conftest import (
    CLOSURE_TEST_PDFSET,
    TEST_DATASET,
    TEST_DATASET_HAD,
    TEST_DATASETS,
    TEST_DATASETS_HAD,
)
from colibri.theory_predictions import (
    fktable_xgrid_indices,
    make_dis_prediction,
    make_had_prediction,
)


# Mock FKTableData class to simulate the 'fktable' object
class FKTableDataMock:
    def __init__(self, xgrid):
        self.xgrid = xgrid


def test_fktable_xgrid_indices_fill_with_zeros():
    # Case where fill_fk_xgrid_with_zeros is True
    fktable = FKTableDataMock(xgrid=jnp.array([0.1, 0.2, 0.3]))
    FIT_XGRID = jnp.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])

    expected_indices = jnp.arange(
        len(FIT_XGRID)
    )  # Should return indices for the entire FIT_XGRID
    result = fktable_xgrid_indices(fktable, FIT_XGRID, fill_fk_xgrid_with_zeros=True)

    assert jnp.array_equal(result, expected_indices)


def test_fktable_xgrid_indices_no_fill():
    # Case where fill_fk_xgrid_with_zeros is False
    fktable = FKTableDataMock(xgrid=jnp.array([0.1, 0.2, 0.3]))
    FIT_XGRID = jnp.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])

    expected_indices = jnp.array([1, 3, 5])  # Indices where fk_xgrid matches FIT_XGRID
    result = fktable_xgrid_indices(fktable, FIT_XGRID, fill_fk_xgrid_with_zeros=False)

    assert jnp.array_equal(result, expected_indices)


def test_fktable_xgrid_indices_with_tolerance():
    # Case where some points are close within tolerance
    fktable = FKTableDataMock(xgrid=jnp.array([0.10000001, 0.2, 0.30000001]))
    FIT_XGRID = jnp.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])

    # Due to tolerance, the indices should match as if they were the same
    expected_indices = jnp.array([1, 3, 5])
    result = fktable_xgrid_indices(fktable, FIT_XGRID, fill_fk_xgrid_with_zeros=False)

    assert jnp.array_equal(result, expected_indices)


def test_fktable_xgrid_indices_no_matches():
    # Case where no FK table xgrid matches FIT_XGRID
    fktable = FKTableDataMock(xgrid=jnp.array([0.6, 0.7, 0.8]))
    FIT_XGRID = jnp.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])

    expected_indices = jnp.array(
        []
    )  # No matching indices, closest_indices returns empty array
    result = fktable_xgrid_indices(fktable, FIT_XGRID, fill_fk_xgrid_with_zeros=False)
    assert jnp.array_equal(result, expected_indices)


def test_fast_kernel_arrays():
    """
    Test that fast_kernel_arrays correctly loads FK tables and handles different parameters.
    """
    # Load data
    dataset = colibriAPI.data(**TEST_DATASETS)
    ds = dataset.datasets[0]

    # Base test: Default behavior
    fk_arrays = colibriAPI.fast_kernel_arrays(**TEST_DATASETS)
    assert isinstance(fk_arrays, tuple)
    assert len(fk_arrays) == len(dataset.datasets)
    assert isinstance(fk_arrays[0], tuple)

    # Manually load expected FK table
    fk_arr_expected = jnp.array(
        load_fktable(ds.fkspecs[0]).with_cuts(ds.cuts).get_np_fktable()
    )
    assert_allclose(fk_arrays[0][0], fk_arr_expected)

    # Test with specific flavour indices
    flavour_indices = ["g", "V"]  # Example: selecting specific flavours
    fk_arrays_flav = colibriAPI.fast_kernel_arrays(
        **{**TEST_DATASETS, "flavour_mapping": flavour_indices}
    )
    assert fk_arrays_flav[0][0].shape[1] == len(
        flavour_indices
    )  # Ensure correct number of flavours

    # Test with fill_fk_xgrid_with_zeros=True
    fk_arrays_filled = colibriAPI.fast_kernel_arrays(
        **{**TEST_DATASETS, "fill_fk_xgrid_with_zeros": True}
    )
    FIT_XGRID = colibriAPI.FIT_XGRID(**TEST_DATASETS)
    assert fk_arrays_filled[0][0].shape[-1] == len(FIT_XGRID)  # Check x-grid size

    # Ensure non-zero indices are properly mapped
    from colibri.utils import closest_indices

    fk_xgrid = load_fktable(ds.fkspecs[0]).xgrid
    non_zero_indices = closest_indices(FIT_XGRID, fk_xgrid, atol=1e-8)
    assert jnp.any(fk_arrays_filled[0][0][:, :, non_zero_indices] != 0)


def test_make_dis_prediction():
    """
    Test make_dis_prediction function gives the same results
    when all luminosity indexes are used to when flavour_indices=None
    """
    ds = colibriAPI.dataset(**TEST_DATASET)
    pdf_grid = colibriAPI.closure_test_pdf_grid(
        **{**CLOSURE_TEST_PDFSET, **TEST_DATASETS}
    )

    fktable = load_fktable(ds.fkspecs[0])
    fk_arr = jnp.array(fktable.get_np_fktable())

    FIT_XGRID = colibriAPI.FIT_XGRID(**TEST_DATASETS)
    pred1 = make_dis_prediction(fktable, FIT_XGRID, flavour_indices=None)(
        pdf_grid[0], fk_arr
    )

    pred2 = make_dis_prediction(
        fktable, FIT_XGRID, flavour_indices=fktable.luminosity_mapping
    )(pdf_grid[0], fk_arr)

    func = make_dis_prediction(fktable, FIT_XGRID, flavour_indices=None)
    pred = func(pdf_grid[0], fk_arr)

    assert_allclose(pred1, pred2)
    assert callable(func)
    assert type(pred) == jaxlib.xla_extension.ArrayImpl


def test_make_had_prediction():
    """
    Test make_had_prediction function gives the same results
    when all luminosity indexes are used to when flavour_indices=None
    """
    ds = colibriAPI.dataset(**TEST_DATASET_HAD)
    pdf_grid = colibriAPI.closure_test_pdf_grid(
        **{**CLOSURE_TEST_PDFSET, **TEST_DATASETS_HAD}
    )

    fktable = load_fktable(ds.fkspecs[0])
    fk_arr = jnp.array(fktable.get_np_fktable())

    FIT_XGRID = colibriAPI.FIT_XGRID(**TEST_DATASETS_HAD)
    pred1 = make_had_prediction(fktable, FIT_XGRID, flavour_indices=None)(
        pdf_grid[0], fk_arr
    )

    pred2 = make_had_prediction(
        fktable, FIT_XGRID, flavour_indices=fktable.luminosity_mapping
    )(pdf_grid[0], fk_arr)

    assert_allclose(pred1, pred2)

    func = make_had_prediction(fktable, FIT_XGRID, flavour_indices=None)
    pred = func(pdf_grid[0], fk_arr)

    assert callable(func)
    assert type(pred) == jaxlib.xla_extension.ArrayImpl


def test_make_pred_data():
    """
    Tests that make_pred_data returns a function.
    """
    eval_preds = colibriAPI.make_pred_data(**{**TEST_DATASETS, **TEST_DATASET})

    fk_arrs = colibriAPI.fast_kernel_arrays(**TEST_DATASETS)
    pdf_grid = colibriAPI.closure_test_central_pdf_grid(
        **{**CLOSURE_TEST_PDFSET, **TEST_DATASETS}
    )

    pred_data = eval_preds(pdf_grid, fk_arrs)

    assert callable(eval_preds)
    assert pred_data.shape == (fk_arrs[0][0].shape[0],)
