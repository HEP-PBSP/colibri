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
