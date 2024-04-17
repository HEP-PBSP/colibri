"""
Main scope of this module is to test that the union of the xgrid of the fktables of the 
DIS and Hadronic datasets are the the same as the XGRID stored in the constants.py file.
"""

import numpy as np
from numpy.testing import assert_allclose

from colibri.api import API as cAPI
from colibri.constants import XGRID
from colibri.tests.conftest import TEST_FULL_DIS_DATASET

from validphys.fkparser import load_fktable

def test_dis_xgrid():
    """
    Tests that the union of the xgrids of the DIS datasets in the theory specified
    in conftest is the same as the XGRID stored in constants.py
    """

    data = cAPI.data(**TEST_FULL_DIS_DATASET)

    xgrids = set()

    for ds in data.datasets:
        for fkspec in ds.fkspecs:
            fktable = load_fktable(fkspec)
            xgrids.update(fktable.xgrid)

    xgrid_union = np.array(sorted(xgrids))

    assert_allclose(xgrid_union, np.array(XGRID))


def test_had_xgrid():
    """
    Tests that the union of the xgrids of the Hadronic datasets in the theory specified
    in conftest is the same as the XGRID stored in constants.py
    """
    # TODO