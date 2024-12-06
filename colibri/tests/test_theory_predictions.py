"""
colibri.tests.test_theory_predictions.py

Test module for theory_predictions.py
"""

from numpy.testing import assert_allclose
import jax.numpy as jnp
import jaxlib

from colibri.api import API as colibriAPI
from colibri.theory_predictions import make_dis_prediction, make_had_prediction

from colibri.tests.conftest import (
    TEST_DATASET,
    CLOSURE_TEST_PDFSET,
    TEST_DATASET_HAD,
    TEST_DATASETS,
    TEST_DATASETS_HAD,
)

from validphys.fkparser import load_fktable


def test_fast_kernel_arrays():
    """
    Test that the fast kernel arrays are correctly loaded
    """
    fk_arrays = colibriAPI.fast_kernel_arrays(**TEST_DATASETS)

    assert len(fk_arrays) == 1
    assert type(fk_arrays) == tuple
    assert type(fk_arrays[0]) == tuple

    data = colibriAPI.data(**TEST_DATASETS)
    ds = data.datasets[0]
    fk_arr = jnp.array(load_fktable(ds.fkspecs[0]).with_cuts(ds.cuts).get_np_fktable())

    assert_allclose(fk_arrays[0][0], fk_arr)


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
