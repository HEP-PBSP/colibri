"""
Main scope of this module is to test that the union of the xgrid of the fktables of the 
DIS and Hadronic datasets are the the same as the XGRID stored in the constants.py file.
"""

import numpy as np
from colibri.api import API as cAPI
from colibri.constants import XGRID
from colibri.tests.conftest import TEST_FULL_DIS_DATASET, TEST_FULL_HAD_DATASET
from numpy.testing import assert_allclose


def test_dis_xgrid():
    """
    Tests that the union of the xgrids of the DIS datasets in the theory specified
    in conftest is the same as the XGRID stored in constants.py
    """

    FIT_XGRID = cAPI.FIT_XGRID(**TEST_FULL_DIS_DATASET)

    assert_allclose(FIT_XGRID, np.array(XGRID))


def test_had_xgrid():
    """
    Tests that the union of the xgrids of the Hadronic datasets in the theory specified
    in conftest is the same as the XGRID stored in constants.py
    """

    FIT_XGRID = cAPI.FIT_XGRID(**TEST_FULL_HAD_DATASET)

    # note: the very low x values are not used for the hadronic datasets
    assert_allclose(FIT_XGRID, np.array(XGRID[6:]))
