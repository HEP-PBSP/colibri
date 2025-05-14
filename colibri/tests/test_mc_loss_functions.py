"""
colibri.tests.test_mc_loss_functions

Tests for the Monte Carlo loss functions in the colibri package.
"""

import jax.numpy as jnp

from colibri.mc_loss_functions import (
    make_chi2_training_data,
    make_chi2_training_data_with_positivity,
    make_chi2_validation_data,
    make_chi2_validation_data_with_positivity,
)
from colibri.mc_utils import MCPseudodata
from colibri.training_validation import PosdataTrainValidationSplit
from colibri.tests.conftest import TEST_POS_FK_ARRAYS

# Mock data
training_indices = jnp.array([0, 1, 2])
validation_indices = jnp.array([0, 1])
pseudodata = jnp.array([10.0, 20.0, 30.0, 40.0])

fit_covariance_matrix = jnp.array(
    [
        [1.0, 0.1, 0.2, 0.3],
        [0.1, 1.0, 0.4, 0.5],
        [0.2, 0.4, 1.0, 0.6],
        [0.3, 0.5, 0.6, 1.0],
    ]
)

# Expected values
predictions = jnp.array([12.0, 18.0, 31.0, 42.0])
pdf = jnp.array([1.0, 1.0, 1.0, 1.0])

alpha = 0.5
lambda_positivity = 1.0


def test_chi2_training_data():

    mc_pseudodata = MCPseudodata(
        pseudodata=pseudodata,
        training_indices=training_indices,
        validation_indices=None,
    )

    batch_idx = jnp.array([0, 1])

    # Computing chi2 value
    chi2_func = make_chi2_training_data(mc_pseudodata, fit_covariance_matrix)
    chi2_value = chi2_func(predictions, batch_idx)

    # Calculate expected chi2 manually for verification
    expected_chi2 = 8.888889

    # Assertion
    assert jnp.isclose(
        chi2_value, expected_chi2
    ), f"Expected {expected_chi2}, got {chi2_value}"


def test_chi2_validation_data():

    mc_pseudodata = MCPseudodata(
        pseudodata=pseudodata,
        training_indices=None,
        validation_indices=validation_indices,
        trval_split=True,
    )

    # Computing chi2 value
    chi2_func = make_chi2_validation_data(mc_pseudodata, fit_covariance_matrix)
    chi2_value = chi2_func(predictions)

    # Calculate expected chi2 manually for verification
    expected_chi2 = 8.88888931274414

    # Assertion
    assert jnp.isclose(
        chi2_value, expected_chi2
    ), f"Expected {expected_chi2}, got {chi2_value}"


def test_chi2_training_data_with_positivity():

    mc_pseudodata = MCPseudodata(
        pseudodata=pseudodata,
        training_indices=training_indices,
        validation_indices=None,
    )

    # Expected values
    predictions = jnp.array([12.0, 18.0, 31.0, 42.0])
    batch_idx = jnp.array([0, 1, 2])

    mc_posdata_split = PosdataTrainValidationSplit(
        training=training_indices, validation=None, n_training=3, n_validation=0
    )

    mock_penalty_posdata = (
        lambda pdf, alpha, lambda_positivity, TEST_POS_FK_ARRAYS: jnp.array(
            [1.0, 2.0, 3.0, 4.0]
        )
    )

    # Computing chi2 value
    chi2_func = make_chi2_training_data_with_positivity(
        mc_pseudodata, mc_posdata_split, fit_covariance_matrix, mock_penalty_posdata
    )
    chi2_value = chi2_func(
        predictions,
        pdf,
        batch_idx,
        alpha,
        lambda_positivity,
        TEST_POS_FK_ARRAYS,
    )

    # Calculate expected chi2 manually for verification
    expected_chi2 = 17.451614

    # Assertion
    assert jnp.isclose(
        chi2_value, expected_chi2
    ), f"Expected {expected_chi2}, got {chi2_value}"


def test_chi2_validation_data_with_positivity():
    mc_pseudodata = MCPseudodata(
        pseudodata=pseudodata,
        training_indices=None,
        validation_indices=validation_indices,
        trval_split=True,
    )

    mc_posdata_split = PosdataTrainValidationSplit(
        training=None, validation=validation_indices, n_training=0, n_validation=2
    )

    mock_penalty_posdata = (
        lambda pdf, alpha, lambda_positivity, TEST_POS_FK_ARRAYS: jnp.array(
            [1.0, 2.0, 3.0, 4.0]
        )
    )

    # Computing chi2 value
    chi2_func = make_chi2_validation_data_with_positivity(
        mc_pseudodata, mc_posdata_split, fit_covariance_matrix, mock_penalty_posdata
    )
    chi2_value = chi2_func(
        predictions,
        pdf,
        alpha,
        lambda_positivity,
        TEST_POS_FK_ARRAYS,
    )

    # Calculate expected chi2 manually for verification
    expected_chi2 = 11.888889

    # Assertion
    assert jnp.isclose(
        chi2_value, expected_chi2
    ), f"Expected {expected_chi2}, got {chi2_value}"
