from colibri.api import API as colibriAPI

from colibri.tests.conftest import (
    T0_PDFSET,
    TRVAL_INDEX,
    REPLICA_INDEX,
    TestPDFModel,
    N_PARAMS,
    TEST_POS_DATASET,
    TEST_DATASETS_DIS_HAD,
)

import jaxlib


def test_make_chi2():
    """
    Tests the function in colibri.loss_functions.make_chi2 works as expected.
    """
    # loss function is a compiled function that takes in input predictions from forward map
    loss_function = colibriAPI.make_chi2(**{**TEST_DATASETS_DIS_HAD, **T0_PDFSET})

    # forward map is a compiled function that takes in input a model (eg pdf)
    forward_map = colibriAPI.make_pred_data(**{**TEST_DATASETS_DIS_HAD, **T0_PDFSET})

    pdf_model = TestPDFModel(n_parameters=N_PARAMS)

    FIT_XGRID = colibriAPI.FIT_XGRID(**TEST_DATASETS_DIS_HAD)

    pred_and_pdf = pdf_model.pred_and_pdf_func(FIT_XGRID, forward_map=forward_map)

    predictions, _ = pred_and_pdf(FIT_XGRID)

    assert loss_function(predictions).dtype == float
    assert loss_function(predictions) >= 0


def test_make_chi2_with_positivity():
    """
    Tests the function in colibri.loss_functions.make_chi2_with_positivity works as expected.
    """
    # loss function is a compiled function that takes in input predictions from forward map
    loss_function = colibriAPI.make_chi2_with_positivity(
        **{**TEST_DATASETS_DIS_HAD, **TEST_POS_DATASET, **T0_PDFSET}
    )

    # forward map is a compiled function that takes in input a model (eg pdf)
    forward_map = colibriAPI.make_pred_data(**{**TEST_DATASETS_DIS_HAD, **T0_PDFSET})

    pdf_model = TestPDFModel(n_parameters=N_PARAMS)

    FIT_XGRID = colibriAPI.FIT_XGRID(**{**TEST_DATASETS_DIS_HAD, **TEST_POS_DATASET})

    pred_and_pdf = pdf_model.pred_and_pdf_func(FIT_XGRID, forward_map=forward_map)

    predictions, pdf = pred_and_pdf(FIT_XGRID)

    assert loss_function(predictions, pdf).dtype == float
    assert loss_function(predictions, pdf) >= 0


def test_make_chi2_training_data():
    """
    Test the function produced by make_chi2_training_data works as expected. In particular tests:
    - make_chi2_training_data returns a jit-compiled function
    -
    """

    result = colibriAPI.make_chi2_training_data(
        **{**TEST_DATASETS_DIS_HAD, **T0_PDFSET, **TRVAL_INDEX, **REPLICA_INDEX}
    )

    # Check that make_chi2_training_data is a jit-compiled function
    assert isinstance(result, jaxlib.xla_extension.PjitFunction)
