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
from colibri.constants import XGRID
from colibri.tests.conftest import (
    TEST_DATASETS,
    TEST_POS_DATASET,
    TEST_SINGLE_POS_DATASET,
    TEST_SINGLE_POS_DATASET_HAD,
    TEST_THEORYID,
    TEST_USECUTS,
    TEST_XGRID,
)
from colibri.theory_penalties import (
    integrability_penalty,
    make_penalty_posdataset,
    positivity_fast_kernel_arrays,
)


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
    penalty_fn = integrability_penalty(integrability_settings, TEST_XGRID)

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
            0.1,
            0.2,
        ],
    }

    # Get the penalty function
    penalty_fn = integrability_penalty(integrability_settings, TEST_XGRID)

    pdf_dummy = jnp.ones((14, 50))  # assumed to be x * pdf
    penalty = penalty_fn(pdf_dummy)

    # expected penalty
    expected_penalty = 2 * (2 * 2)
    assert jnp.sum(penalty, axis=-1) == expected_penalty


def test_integrability_penalty_raises_error():
    """
    Tests that the error is raised as should in the integrability penalty function.
    """

    integrability_settings = MagicMock()
    integrability_settings.integrability = True
    integrability_settings.integrability_specs = {
        "evolution_flavours": [0, 1, 2],
        "integrability_xgrid": [0.05, 0.15, 0.35],  # Out of FIT_XGRID range
        "lambda_integrability": 1.0,
    }
    FIT_XGRID = jnp.array([0.1, 0.2, 0.3])

    with pytest.raises(
        ValueError,
        match="Integrability xgrid points are not included in the range of the fit xgrid",
    ):
        integrability_penalty(integrability_settings, FIT_XGRID)


def test_make_penalty_posdataset_pos_penalty():
    """
    Tests the callable that the make_penalty_posdataset function returns.
    Tests that penalty is small negative number when PDF is positive.
    """
    # Mock inputs
    posdatasets = colibriAPI.posdatasets(
        **{**TEST_POS_DATASET, "theoryid": TEST_THEORYID, "use_cuts": TEST_USECUTS}
    )
    posdataset = posdatasets[0]

    FIT_XGRID = XGRID
    flavour_indices = None

    penalty_func = make_penalty_posdataset(posdataset, FIT_XGRID, flavour_indices)

    # Define positive mock PDF that should not yield a penalty.
    pdf = jnp.ones((14, 50))
    alpha = 1e-7
    lambda_positivity = 2.0
    fk_tuple = positivity_fast_kernel_arrays(posdatasets, flavour_indices=None)[0]

    # Compute result
    result = penalty_func(pdf, alpha, lambda_positivity, fk_tuple).sum()

    # Expected result: small negative number because of elu function
    assert result < 0, "Penalty function computation seems incorrect."
    assert abs(result) < 1e-2


def test_make_penalty_posdata_pos_penalties():
    """
    Tests the callable that the make_penalty_posdata function returns.
    Tests that result is smaller than zero
    """
    test_inp = {**TEST_POS_DATASET, **TEST_DATASETS, "fill_fk_xgrid_with_zeros": True}
    penalty_posdata_func = colibriAPI.make_penalty_posdata(**test_inp)

    fk_arrays = colibriAPI.fast_kernel_arrays(**test_inp)

    pdf = jnp.ones((14, 50))
    alpha = 1e-7
    lambda_positivity = 2.0

    result = penalty_posdata_func(pdf, alpha, lambda_positivity, fk_arrays).sum()

    assert result < 0, "Penalty function computation seems incorrect."
    assert abs(result) < 1e-2
