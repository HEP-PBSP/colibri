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

    NOTE: testing assumes the following ordering of the flavours:

    Evolution basis:
    [photon, sigma, gluon, V, V3, V8, V15, V24, V35, T3, T8, T15, T24, T35]

    Flavour basis:
    [TBAR, BBAR, CBAR, SBAR, UBAR, DBAR, gluon, D, U, S, C, B, T, photon]

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

    # gluon basis vector in the evolution basis
    gluon_ev = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # gluon basis vector in the flavour basis
    gluon_fl = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

    assert np.allclose(evolution_to_flavour_matrix @ gluon_ev, gluon_fl)

    # V basis vector in the evolution basis
    V_ev = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # V basis vector in the flavour basis
    V_fl = np.array(
        [
            -1 / 12,
            -1 / 12,
            -1 / 12,
            -1 / 12,
            -1 / 12,
            -1 / 12,
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

    assert np.allclose(evolution_to_flavour_matrix @ V_ev, V_fl)

    # V3 basis vector in the evolution basis
    V3_ev = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # V3 basis vector in the flavour basis
    V3_fl = np.array([0, 0, 0, 0, -0.25, 0.25, 0, -0.25, 0.25, 0, 0, 0, 0, 0])

    assert np.allclose(evolution_to_flavour_matrix @ V3_ev, V3_fl)

    # V8 basis vector in the evolution basis
    V8_ev = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    # V8 basis vector in the flavour basis
    V8_fl = np.array(
        [0, 0, 0, 1 / 6, -1 / 12, -1 / 12, 0, 1 / 12, 1 / 12, -1 / 6, 0, 0, 0, 0]
    )

    assert np.allclose(evolution_to_flavour_matrix @ V8_ev, V8_fl)

    # V15 basis vector in the evolution basis
    V15_ev = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

    # V15 basis vector in the flavour basis
    V15_fl = np.array(
        [
            0,
            0,
            0.125,
            -1 / 24,
            -1 / 24,
            -1 / 24,
            0,
            1 / 24,
            1 / 24,
            1 / 24,
            -0.125,
            0,
            0,
            0,
        ]
    )
    assert np.allclose(evolution_to_flavour_matrix @ V15_ev, V15_fl)
