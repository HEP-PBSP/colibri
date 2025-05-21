"""
colibri.tests.test_constants.py

Module for testing that the constants in the colibri module are
working as expected.
"""

import numpy as np
from colibri.constants import (
    FLAVOURS_ID_MAPPINGS,
    FLAVOUR_TO_ID_MAPPING,
    XGRID,
    LHAPDF_XGRID,
    evolution_to_flavour_matrix,
)
from colibri.tests.conftest import EXPECTED_XGRID, EXPECTED_LHAPDF_XGRID


def test_id_to_flavour_mappings():
    """
    Test that the id to flavour mapping is correct.
    """

    # Check that the type is dict
    assert isinstance(FLAVOURS_ID_MAPPINGS, dict)

    # Check that the mapping is correct
    assert FLAVOURS_ID_MAPPINGS[0] == "photon"
    assert FLAVOURS_ID_MAPPINGS[1] == r"\Sigma"
    assert FLAVOURS_ID_MAPPINGS[2] == "g"
    assert FLAVOURS_ID_MAPPINGS[3] == "V"
    assert FLAVOURS_ID_MAPPINGS[4] == "V3"
    assert FLAVOURS_ID_MAPPINGS[5] == "V8"
    assert FLAVOURS_ID_MAPPINGS[6] == "V15"
    assert FLAVOURS_ID_MAPPINGS[7] == "V24"
    assert FLAVOURS_ID_MAPPINGS[8] == "V35"
    assert FLAVOURS_ID_MAPPINGS[9] == "T3"
    assert FLAVOURS_ID_MAPPINGS[10] == "T8"
    assert FLAVOURS_ID_MAPPINGS[11] == "T15"
    assert FLAVOURS_ID_MAPPINGS[12] == "T24"
    assert FLAVOURS_ID_MAPPINGS[13] == "T35"


def test_flavour_to_id_mapping():
    """
    Test that the flavour to ID mapping is correct.
    """
    # Check that the type is dict
    assert isinstance(FLAVOUR_TO_ID_MAPPING, dict)

    # Check that the mapping is correct
    assert FLAVOUR_TO_ID_MAPPING["photon"] == 0
    assert FLAVOUR_TO_ID_MAPPING[r"\Sigma"] == 1
    assert FLAVOUR_TO_ID_MAPPING["g"] == 2
    assert FLAVOUR_TO_ID_MAPPING["V"] == 3
    assert FLAVOUR_TO_ID_MAPPING["V3"] == 4
    assert FLAVOUR_TO_ID_MAPPING["V8"] == 5
    assert FLAVOUR_TO_ID_MAPPING["V15"] == 6
    assert FLAVOUR_TO_ID_MAPPING["V24"] == 7
    assert FLAVOUR_TO_ID_MAPPING["V35"] == 8
    assert FLAVOUR_TO_ID_MAPPING["T3"] == 9
    assert FLAVOUR_TO_ID_MAPPING["T8"] == 10
    assert FLAVOUR_TO_ID_MAPPING["T15"] == 11
    assert FLAVOUR_TO_ID_MAPPING["T24"] == 12
    assert FLAVOUR_TO_ID_MAPPING["T35"] == 13


def test_XGRID():
    """
    Test that the XGRID is correct.
    """

    # Check the expected XGRID length
    assert len(XGRID) == 50
    assert XGRID == EXPECTED_XGRID


def test_LHAPDF_XGRID():
    """
    Test that the LHAPDFXGRID is correct.
    """

    # Check the expected XGRID length
    assert len(LHAPDF_XGRID) == 196
    assert LHAPDF_XGRID == EXPECTED_LHAPDF_XGRID


def test_evolution_to_flavour_matrix():
    """
    Tests that the evolution to flavour matrix rotation is correct.
    """

    # photon basis vector in the evolution basis
    photon_ev = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # photon basis vector in the flavour basis
    photon_fl = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    assert np.allclose(evolution_to_flavour_matrix @ photon_ev, photon_fl)

    # sigma basis vector in the evolution basis
    sigma_ev = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # sigma basis vector in the flavour basis
    sigma_fl = np.array(
        [
            1 / 12,
            1 / 12,
            1 / 12,
            1 / 12,
            1 / 12,
            1 / 12,
            0,
            1 / 12,
            1 / 12,
            1 / 12,
            1 / 12,
            1 / 12,
            1 / 12,
            0,
        ]
    )
    assert np.allclose(evolution_to_flavour_matrix @ sigma_ev, sigma_fl)
