"""
colibri.tests.test_pdf_model

Tests for the PDFModel class.
"""

import jax.numpy as jnp
from numpy.testing import assert_array_equal

from colibri.tests.conftest import (
    TEST_FK_ARRAYS,
    TEST_FORWARD_MAP_DIS,
    TEST_PDF_GRID,
    TEST_XGRID,
    TestPDFModel,
)

model = TestPDFModel(n_parameters=2)


def test_param_names():
    """
    Tests that the param_names property returns the correct names.
    """
    assert model.param_names == ["w_1", "w_2"]


def test_param_names_order():
    """
    Tests that param_names returns names in the correct order.

    This covers the docstring requirement:
    "The order of the names is important as it will be assumed to be the order
    of the parameters fed to the model."
    """
    names = model.param_names

    # Check that names follow the expected pattern
    assert isinstance(names, list), "param_names should return a list"
    assert len(names) > 0, "param_names should not be empty"

    # For TestPDFModel with 2 parameters, check order
    assert names[0] == "w_1", "First parameter should be 'w_1'"
    assert names[1] == "w_2", "Second parameter should be 'w_2'"


def test_param_names_different_sizes():
    """
    Tests that param_names works correctly with different numbers of parameters.

    This ensures the property dynamically generates names based on n_parameters.
    """
    # Test with 1 parameter
    model_1 = TestPDFModel(n_parameters=1)
    assert model_1.param_names == ["w_1"]
    assert len(model_1.param_names) == 1

    # Test with 3 parameters
    model_3 = TestPDFModel(n_parameters=3)
    assert model_3.param_names == ["w_1", "w_2", "w_3"]
    assert len(model_3.param_names) == 3

    # Test with 5 parameters
    model_5 = TestPDFModel(n_parameters=5)
    assert len(model_5.param_names) == 5
    assert model_5.param_names == ["w_1", "w_2", "w_3", "w_4", "w_5"]


def test_n_parameters_from_explicit_setting():
    """
    Tests that n_parameters returns the value set explicitly in __init__.

    This covers the case: if self._n_parameters is not None, return that value
    """
    # When TestPDFModel sets n_parameters in __init__
    model_test = TestPDFModel(n_parameters=4)

    # n_parameters should return the explicitly set value
    assert model_test.n_parameters == 4
    assert isinstance(model_test.n_parameters, int)


def test_n_parameters_derived_from_param_names():
    """
    Tests that n_parameters returns len(param_names) when derived.

    This covers the case: return len(self.param_names)
    """
    # Create a model and verify n_parameters matches len(param_names)
    model_test = TestPDFModel(n_parameters=3)

    # n_parameters should equal length of param_names
    assert model_test.n_parameters == len(model_test.param_names)

    # Both should be 3
    assert model_test.n_parameters == 3
    assert len(model_test.param_names) == 3


def test_n_parameters_consistency():
    """
    Tests that n_parameters is consistent across multiple accesses.

    Ensures that both paths (explicit or derived) give consistent results.
    """
    # Access n_parameters multiple times
    n1 = model.n_parameters
    n2 = model.n_parameters
    n3 = model.n_parameters

    # All accesses should return the same value
    assert n1 == n2 == n3 == 2


def test_n_parameters_matches_param_names_length():
    """
    Tests that n_parameters always matches len(param_names).

    This is critical because param_names order matters for parameter feeding.
    """
    for n in [1, 2, 3, 5, 10]:
        model_n = TestPDFModel(n_parameters=n)
        assert model_n.n_parameters == len(
            model_n.param_names
        ), f"n_parameters ({model_n.n_parameters}) should match len(param_names) ({len(model_n.param_names)})"


def test_param_names_type():
    """
    Tests that param_names returns a list.

    Validates the return type specified in the docstring.
    """
    names = model.param_names
    assert isinstance(names, list), "param_names must return a list"
    assert all(
        isinstance(name, str) for name in names
    ), "All param names must be strings"


def test_grid_values_func():
    """
    Tests that the grid_values_func returns the correct values.
    """
    func = model.grid_values_func(TEST_XGRID)
    params = jnp.array([2, 3])

    expected_output = sum([param * TEST_PDF_GRID for param in params])

    assert_array_equal(func(params), expected_output)


def test_pred_and_pdf_func():
    """
    Tests that the pred_and_pdf_func returns the correct values.
    """
    pred_and_pdf = model.pred_and_pdf_func(TEST_XGRID, TEST_FORWARD_MAP_DIS)

    params = jnp.array([2, 3])
    predictions, pdf = pred_and_pdf(params, TEST_FK_ARRAYS)

    expected_predictions = jnp.einsum("ijk,jk->i", TEST_FK_ARRAYS[0], pdf)

    assert jnp.allclose(predictions, expected_predictions)
