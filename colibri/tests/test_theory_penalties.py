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
    TEST_XGRID,
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


import numpy as np
import jax
import jax.numpy as jnp
import pytest
from unittest.mock import MagicMock, patch

# Mock dependencies
from validphys.core import PositivitySetSpec
from colibri.theory_predictions import pred_funcs_from_dataset  # Adjust import as necessary
from colibri.theory_penalties import make_penalty_posdataset  # Adjust import as necessary

def test_make_penalty_posdataset():
    """
    Test that penalty is zero when PDF are POS
    """
    # Mock inputs
    posdataset = MagicMock(spec=PositivitySetSpec)
    posdataset.fkspecs = MagicMock()
    FIT_XGRID = np.array([0.1, 0.2, 0.3])
    flavour_indices = None

    # Mock pred_funcs_from_dataset
    def mock_pred_func(pdf, fk_arr):
        return pdf * fk_arr
    
    pred_funcs_from_dataset.return_value = [mock_pred_func, mock_pred_func]
    
    with patch("colibri.theory_penalties.OP", new_callable=lambda: {"NULL": lambda *args: sum(args)}):
        posdataset.op = "NULL"
        penalty_func = make_penalty_posdataset(posdataset, FIT_XGRID, flavour_indices)
        
        # Define test inputs
        pdf = jnp.array([1.0, 2.0, 3.0])
        alpha = 1.0
        lambda_positivity = 2.0
        fk_dataset = [jnp.array([0.5, 0.5, 0.5]), jnp.array([0.5, 0.5, 0.5])]
        
        # Compute result
        result = penalty_func(pdf, alpha, lambda_positivity, fk_dataset)
        
        # Expected result (manually computed)
        expected = lambda_positivity * jax.nn.elu(-((pdf * 0.5) + (pdf * 0.5)), alpha)
        import IPython; IPython.embed()
        
        assert jnp.allclose(result, expected), "Penalty function computation is incorrect"
