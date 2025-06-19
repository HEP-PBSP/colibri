"""
colibri.tests.test_loss_functions

Tests for the loss functions in the colibri package.
"""

from colibri.api import API as colibriAPI
from colibri.tests.conftest import (
    REPLICA_INDEX,
    T0_PDFSET,
    TEST_DATASETS_DIS_HAD,
    TEST_POS_DATASET,
    TRVAL_INDEX,
)


def test_make_chi2_training_data():
    """
    Tests that make_chi2_training_data returns a function.
    """

    result = colibriAPI.make_chi2_training_data(
        **{**TEST_DATASETS_DIS_HAD, **T0_PDFSET, **TRVAL_INDEX, **REPLICA_INDEX}
    )

    # Check that make_chi2_training_data is a callable
    assert callable(result)


def test_make_chi2_training_data_with_positivity():
    """
    Tests that make_chi2_training_data_with_positivity returns a function.
    """

    result = colibriAPI.make_chi2_training_data_with_positivity(
        **{
            **TEST_DATASETS_DIS_HAD,
            **TEST_POS_DATASET,
            **T0_PDFSET,
            **TRVAL_INDEX,
            **REPLICA_INDEX,
        }
    )

    # Check that make_chi2_training_data_with_positivity is a callable
    assert callable(result)


def test_make_chi2_validation_data_with_positivity():
    """
    Tests that make_chi2_validation_data_with_positivity returns a function.
    """

    chi2_func = colibriAPI.make_chi2_validation_data_with_positivity(
        **{
            **TEST_DATASETS_DIS_HAD,
            **TEST_POS_DATASET,
            **T0_PDFSET,
            **TRVAL_INDEX,
            **REPLICA_INDEX,
        }
    )

    # Check that make_chi2_validation_data_with_positivity is a callable
    assert callable(chi2_func)

    # check that if no validation data is provided, the function returns jnp.nan
    chi2_func_no_validation = colibriAPI.make_chi2_validation_data_with_positivity(
        **{
            **TEST_DATASETS_DIS_HAD,
            **TEST_POS_DATASET,
            **T0_PDFSET,
            **REPLICA_INDEX,
            "mc_validation_fraction": 0,
        }
    )
    assert callable(chi2_func_no_validation)
    assert type(chi2_func_no_validation(1, 1, 1, 1, 1)) == float
