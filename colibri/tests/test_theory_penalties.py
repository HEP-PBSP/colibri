"""
colibri.tests.test_theory_penalties.py

Test module for theory_penalties.py
"""

from unittest.mock import MagicMock

import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose
from validphys.fkparser import load_fktable

from colibri.api import API as colibriAPI
from colibri.tests.conftest import (
    TEST_DATASETS,
    TEST_POS_DATASET,
    TEST_SINGLE_POS_DATASET,
    TEST_SINGLE_POS_DATASET_HAD,
)
from colibri.theory_penalties import integrability_penalty


def test_positivity_fast_kernel_arrays():
    """
    Test that the positivity fast kernel arrays are correctly loaded
    """
    fk_arrays = colibriAPI.positivity_fast_kernel_arrays(
        **{**TEST_POS_DATASET, **TEST_DATASETS}
    )

    assert len(fk_arrays) == 1
    assert type(fk_arrays) == tuple
    assert type(fk_arrays[0]) == tuple

    data = colibriAPI.posdatasets(**{**TEST_POS_DATASET, **TEST_DATASETS})

    ds = data.data[0]
    fk_arr = jnp.array(load_fktable(ds.fkspecs[0]).with_cuts(ds.cuts).get_np_fktable())

    assert_allclose(fk_arrays[0][0], fk_arr)


@pytest.mark.parametrize(
    "posdataset", [TEST_SINGLE_POS_DATASET, TEST_SINGLE_POS_DATASET_HAD]
)
def test_make_penalty_posdataset(posdataset):
    """
    Tests that make_penalty_posdataset returns a function.
    """
    penalty_posdata = colibriAPI.make_penalty_posdataset(
        **{**posdataset, **TEST_DATASETS}
    )

    assert callable(penalty_posdata)


def test_make_penalty_posdata():
    """
    Tests that make_penalty_posdata returns a function.
    """
    penalty_posdata = colibriAPI.make_penalty_posdata(
        **{**TEST_POS_DATASET, **TEST_DATASETS}
    )

    assert callable(penalty_posdata)


def test_integrability_penalty_no_integrability():
    """
    Test integrability penalty function when integrability is False
    """
    # Mock integrability settings
    integrability_settings = MagicMock()
    integrability_settings.integrability = False

    # Get the penalty function
    penalty_fn = integrability_penalty(integrability_settings)

    # Check that it returns 0 for any input
    pdf_dummy = jnp.ones((14, 50))
    assert penalty_fn(pdf_dummy) == 0


def test_integrability_penalty_integrability():
    """
    Test integrability penalty function when integrability is True
    """
    # Mock integrability settings
    integrability_settings = MagicMock()
    integrability_settings.integrability = True
    integrability_settings.integrability_specs = {
        "evolution_flavours": [
            1,
            2,
        ],
        "lambda_integrability": 2.0,
        "integrability_xgrid": [
            2.00000000e-07,
            3.03430477e-07,
        ],
    }

    # Get the penalty function
    penalty_fn = integrability_penalty(integrability_settings)

    pdf_dummy = jnp.ones((14, 50))  # assumed to be x * pdf
    penalty = penalty_fn(pdf_dummy)

    # expected penalty
    expected_penalty = 2 * (2 * 2)
    assert jnp.sum(penalty, axis=-1) == expected_penalty
