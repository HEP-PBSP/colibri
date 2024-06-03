from numpy.testing import assert_allclose
import jax.numpy as jnp

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

    assert_allclose(pred1, pred2)


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
