from super_net.api import API as SuperNetAPI

from super_net.commondata_utils import experimental_commondata_tuple, pseudodata_commondata_tuple

from validphys.coredata import CommonData

from super_net.tests.conftest import TEST_DATASETS, CLOSURE_TEST_PDFSET
from validphys.covmats import dataset_t0_predictions

from numpy.testing import assert_allclose


def test_closuretest_commondata_tuple():
    """
    Test that the theory predictions defining the central values of the a
    closuretest commondata object are consistent with the ones computed
    using validphys.covmats.dataset_t0_predictions.
    """

    data = SuperNetAPI.data(**TEST_DATASETS)
    pdf = SuperNetAPI.closure_test_pdf(**CLOSURE_TEST_PDFSET)

    # TEST_DATASETS should only contain one dataset
    t0_pred = dataset_t0_predictions(data.datasets[0], pdf)

    ct_cd_tuple = SuperNetAPI.closuretest_commondata_tuple(
        **{**TEST_DATASETS, **CLOSURE_TEST_PDFSET}
    )

    assert_allclose(ct_cd_tuple[0].central_values, t0_pred, rtol=1e-6)


def test_experimental_commondata_tuple():
    """
    Test that experimental_commondata_tuple returns a tuple of CommonData objects. 
    """

    data = SuperNetAPI.data(**TEST_DATASETS)
    result = experimental_commondata_tuple(data)

    # Check that experimental_commondata_tuple produces a tuple
    assert isinstance(result, tuple)

    # Check that experimental_commondata_tuple produces a tuple of CommonData objects
    for commondata_instance in result:
        assert isinstance(commondata_instance, CommonData)

def test_pseudodata_commondata_tuple():
    """
    Test that pseudodata_commondata_tuple returns a tuple of CommonData objects.
    """

    data = SuperNetAPI.data(**TEST_DATASETS)
    exp_tuple = experimental_commondata_tuple(data)
    result = pseudodata_commondata_tuple(data, exp_tuple, 123456)

    # Check that pseudodata_commondata_tuple produces a tuple
    assert isinstance(result, tuple)

    # Check that pseudodata_commondata_tuple produces a tuple of CommonData objects
    for commondata_instance in result:
        assert isinstance(commondata_instance, CommonData) 
