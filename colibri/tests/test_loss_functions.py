from colibri.api import API as colibriAPI

from colibri.tests.conftest import (
    T0_PDFSET,
    TRVAL_INDEX,
    REPLICA_INDEX,
    TEST_DATASETS_DIS_HAD,
)


def test_make_chi2_training_data():
    """
    Test the function produced by make_chi2_training_data works as expected. In particular tests:
    - make_chi2_training_data returns a jit-compiled function
    -
    """

    result = colibriAPI.make_chi2_training_data(
        **{**TEST_DATASETS_DIS_HAD, **T0_PDFSET, **TRVAL_INDEX, **REPLICA_INDEX}
    )

    # Check that make_chi2_training_data is a callable
    assert callable(result)
