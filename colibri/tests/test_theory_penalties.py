"""
colibri.tests.test_theory_penalties.py

Test module for theory_penalties.py
"""

from numpy.testing import assert_allclose
import jax.numpy as jnp
import pytest

from colibri.api import API as colibriAPI

from colibri.tests.conftest import (
    TEST_DATASETS,
    TEST_POS_DATASET,
    TEST_SINGLE_POS_DATASET,
    TEST_SINGLE_POS_DATASET_HAD,
)

from validphys.fkparser import load_fktable


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


import jax.numpy as jnp
import pytest
from unittest.mock import MagicMock
from colibri.theory_penalties import (
    integrability_penalty,
)  # Replace `your_module` with the actual module name


def test_integrability_penalty_no_integrability():
    """
    Test integrability penalty function when integrability is False
    """
    # Mock integrability settings
    integrability_settings = MagicMock()
    integrability_settings.integrability = False

    # Define a dummy FIT_XGRID
    FIT_XGRID = jnp.array([0.01, 0.1, 0.5])

    # Get the penalty function
    penalty_fn = integrability_penalty(integrability_settings, FIT_XGRID)

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
        "evolution_flavours": [1, 2, 3],
        "lambda_integrability": 2.0,
    }

    # # mock the closest indices function
    # global closest_indices
    # mock_closest_indices = lambda XGRID, FIT_XGRID: jnp.array([0]) # assume it selects the first index
    # closest_indices = mock_closest_indices

    # Define a dummy FIT_XGRID
    FIT_XGRID = jnp.array([8.62783932e-01, 9.30944081e-01, 1.00000000e00])

    # define a dummy XGRID (global variable in the module)
    global XGRID
    XGRID = [0.01, 0.1, 0.5]

    # Get the penalty function
    penalty_fn = integrability_penalty(integrability_settings, FIT_XGRID)

    pdf_dummy = jnp.ones((14, 50))  # assumed to be x * pdf
    penalty = penalty_fn(pdf_dummy)

    # expected penalty
    expected_penalty = 2.0 * 3
    assert jnp.sum(penalty, axis=-1) == expected_penalty
