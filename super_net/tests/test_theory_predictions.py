from numpy.testing import assert_allclose

from super_net.api import API as SuperNetAPI
from super_net.theory_predictions import make_dis_prediction, make_had_prediction

from super_net.tests.conftest import TEST_DATASET, CLOSURE_TEST_PDFSET, TEST_DATASET_HAD

from validphys.fkparser import load_fktable


def test_make_dis_prediction():
    """
    Test make_dis_prediction function gives the same results
    when all luminosity indexes are used to when flavour_indices=None
    """
    ds = SuperNetAPI.dataset(**TEST_DATASET)
    pdf_grid = SuperNetAPI.closure_test_pdf_grid(**CLOSURE_TEST_PDFSET)

    fktable = load_fktable(ds.fkspecs[0])

    pred1 = make_dis_prediction(fktable, vectorized=False, flavour_indices=None)(
        pdf_grid[0]
    )

    pred2 = make_dis_prediction(
        fktable, vectorized=False, flavour_indices=fktable.luminosity_mapping
    )(pdf_grid[0])

    assert_allclose(pred1, pred2)


def test_make_had_prediction():
    """
    Test make_had_prediction function gives the same results
    when all luminosity indexes are used to when flavour_indices=None
    """
    ds = SuperNetAPI.dataset(**TEST_DATASET_HAD)
    pdf_grid = SuperNetAPI.closure_test_pdf_grid(**CLOSURE_TEST_PDFSET)

    fktable = load_fktable(ds.fkspecs[0])

    pred1 = make_had_prediction(fktable, vectorized=False, flavour_indices=None)(
        pdf_grid[0]
    )

    pred2 = make_had_prediction(
        fktable, vectorized=False, flavour_indices=fktable.luminosity_mapping
    )(pdf_grid[0])

    assert_allclose(pred1, pred2)
