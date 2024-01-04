from super_net.api import API as SuperNetAPI

from super_net.loss_functions import make_chi2_training_data

from super_net.tests.conftest import TEST_DATASETS, T0_PDFSET, TRVAL_INDEX

from numpy.testing import assert_allclose

import jaxlib

def test_make_chi2_training_data():
    """
    Test the function produced by make_chi2_training_data works as expected. In particular tests:
    - 
    """

    data = SuperNetAPI._data_values(**{**TEST_DATASETS, **T0_PDFSET, **TRVAL_INDEX})
    pred_data = SuperNetAPI._pred_data(**TEST_DATASETS)
    result = make_chi2_training_data(data, pred_data)

    # Check that make_chi2_training_data is a jit-compiled function
    assert isinstance(result, jaxlib.xla_extension.PjitFunction)
