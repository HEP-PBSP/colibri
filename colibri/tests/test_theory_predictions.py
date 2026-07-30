"""
colibri.tests.test_theory_predictions.py

Test module for theory_predictions.py
"""

import jax.numpy as jnp
import jaxlib
from numpy.testing import assert_allclose
from validphys.fkparser import load_fktable

import numpy as np

from validphys.api import API as vpAPI

from colibri.api import API as colibriAPI
from colibri.tests.conftest import (
    CLOSURE_TEST_PDFSET,
    TEST_DATASET,
    TEST_DATASET_HAD,
    TEST_DATASETS,
    TEST_DATASETS_DIS_HAD,
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


def test_fktable_xgrid_indices():
    fktable = FKTableDataMock(xgrid=jnp.array([0.1, 0.2, 0.3]))
    FIT_XGRID = jnp.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])

    expected_indices = jnp.array([1, 3, 5])  # Indices where fk_xgrid matches FIT_XGRID
    result = fktable_xgrid_indices(fktable, FIT_XGRID)

    assert jnp.array_equal(result, expected_indices)


def test_fktable_xgrid_indices_with_tolerance():
    # Case where some points are close within tolerance
    fktable = FKTableDataMock(xgrid=jnp.array([0.10000001, 0.2, 0.30000001]))
    FIT_XGRID = jnp.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])

    # Due to tolerance, the indices should match as if they were the same
    expected_indices = jnp.array([1, 3, 5])
    result = fktable_xgrid_indices(fktable, FIT_XGRID)

    assert jnp.array_equal(result, expected_indices)


def test_fktable_xgrid_indices_no_matches():
    # Case where no FK table xgrid matches FIT_XGRID
    fktable = FKTableDataMock(xgrid=jnp.array([0.6, 0.7, 0.8]))
    FIT_XGRID = jnp.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])

    expected_indices = jnp.array(
        []
    )  # No matching indices, closest_indices returns empty array
    result = fktable_xgrid_indices(fktable, FIT_XGRID)
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


def test_fast_kernel_arrays_hadronic_fill_with_zeros():
    """
    Test that fast_kernel_arrays correctly fills the x-grid with zeros for hadronic FK tables.
    This is a regression test for the bug where the 4D hadronic array was assigned
    into a 3D zeros array.
    """
    from colibri.utils import closest_indices
    from validphys.fkparser import load_fktable

    dataset = colibriAPI.data(**TEST_DATASETS_HAD)
    ds = dataset.datasets[0]
    FIT_XGRID = colibriAPI.FIT_XGRID(**TEST_DATASETS_HAD)

    # This should not raise an error (regression check)
    fk_arrays_filled = colibriAPI.fast_kernel_arrays(
        **{**TEST_DATASETS_HAD, "fill_fk_xgrid_with_zeros": True}
    )

    fk_arr = fk_arrays_filled[0][0]

    # Hadronic FK array should be 4D: (Ndat, Nfl, Nfit_x, Nfit_x)
    assert fk_arr.ndim == 4
    assert fk_arr.shape[2] == len(FIT_XGRID)
    assert fk_arr.shape[3] == len(FIT_XGRID)

    # Check that non-zero values are placed at the correct x-grid positions
    fk_xgrid = load_fktable(ds.fkspecs[0]).xgrid
    non_zero_indices = closest_indices(FIT_XGRID, fk_xgrid, atol=1e-8)
    non_zero_indices = np.array(non_zero_indices)

    # The non-zero block should contain non-zero values
    assert jnp.any(
        fk_arr[:, :, non_zero_indices[:, None], non_zero_indices[None, :]] != 0
    )

    # Entries outside the non-zero block should be zero
    all_indices = np.arange(len(FIT_XGRID))
    zero_indices = np.setdiff1d(all_indices, non_zero_indices)
    if len(zero_indices) > 0:
        assert jnp.all(fk_arr[:, :, zero_indices, :] == 0)
        assert jnp.all(fk_arr[:, :, :, zero_indices] == 0)


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
    assert isinstance(pred, jnp.ndarray)


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
    assert isinstance(pred, jnp.ndarray)


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


def test_predictions_independent_of_fill_fk_xgrid_with_zeros():
    """
    Regression test: ``fill_fk_xgrid_with_zeros`` is a memory/layout option and
    must not change the theory predictions.

    Previously the prediction closures unconditionally indexed the FK array with
    ``fktable_xgrid_indices``, which are indices into FIT_XGRID. That is only
    valid for zero-padded FK arrays; with ``fill_fk_xgrid_with_zeros=False`` the
    FK x-grid axis was silently re-shuffled and out-of-range indices were clamped
    by jax, giving wrong predictions.

    Uses a mixed DIS + hadronic setup so that both convolution paths are covered
    and FIT_XGRID is a strict superset of at least one FK x-grid (otherwise the
    index mapping is the identity and the bug is invisible).
    """
    data = colibriAPI.data(**TEST_DATASETS_DIS_HAD)
    FIT_XGRID = colibriAPI.FIT_XGRID(**TEST_DATASETS_DIS_HAD)

    # Guard: the setup must actually exercise a non-trivial x-grid mapping.
    mappings = [
        fktable_xgrid_indices(load_fktable(ds.fkspecs[0]).with_cuts(ds.cuts), FIT_XGRID)
        for ds in data.datasets
    ]
    assert any(
        not jnp.array_equal(idx, jnp.arange(len(idx))) for idx in mappings
    ), "test setup is degenerate: every FK x-grid maps onto FIT_XGRID as the identity"

    pdf_grid = colibriAPI.closure_test_central_pdf_grid(
        **{**CLOSURE_TEST_PDFSET, **TEST_DATASETS_DIS_HAD}
    )

    preds = {}
    for fill in (False, True):
        inp = {**TEST_DATASETS_DIS_HAD, "fill_fk_xgrid_with_zeros": fill}
        eval_preds = colibriAPI.make_pred_data(**inp)
        preds[fill] = eval_preds(pdf_grid, colibriAPI.fast_kernel_arrays(**inp))

    assert_allclose(preds[False], preds[True], rtol=1e-6)

    # Cross-check both against validphys, which is the ground truth here.
    vp_preds = np.concatenate(
        [
            np.array(
                vpAPI.dataset_inputs_results(
                    **{
                        **TEST_DATASETS_DIS_HAD,
                        "dataset_inputs": [ds_inp],
                        "pdf": CLOSURE_TEST_PDFSET["closure_test_pdf"],
                        "use_t0": False,
                    }
                )[1].central_value
            )
            for ds_inp in TEST_DATASETS_DIS_HAD["dataset_inputs"]
        ]
    )
    assert_allclose(preds[False], vp_preds, rtol=1e-6)
    assert_allclose(preds[True], vp_preds, rtol=1e-6)
