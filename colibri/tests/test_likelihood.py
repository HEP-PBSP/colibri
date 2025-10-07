"""
colibri.tests.test_likelihood.py

Tests for the likelihood module.
"""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose
import colibri

from colibri.likelihood import LogLikelihood, log_likelihood, mc_log_likelihood
from colibri.mc_utils import MCPseudodata
from colibri.tests.conftest import (
    MOCK_CENTRAL_COVMAT_INDEX,
    MOCK_CHI2,
    MOCK_PDF_MODEL,
    MOCK_PENALTY_POSDATA,
    TEST_FK_ARRAYS,
    TEST_FORWARD_MAP_DIS,
    TEST_POS_FK_ARRAYS,
    TEST_XGRID,
)

# Monkey patch chi2 imported in likelihood
colibri.likelihood.chi2 = MOCK_CHI2

jax.config.update("jax_enable_x64", True)

# Define mock input parameters
bayesian_prior = lambda x: x

integrability_penalty = lambda pdf: jnp.array([0.0])


@pytest.mark.parametrize("pos_penalty", [True, False])
def test_LogLikelihood_class(pos_penalty):
    """
    Tests the LogLikelihood class.
    """
    log_likelihood_class = LogLikelihood(
        central_covmat_index=MOCK_CENTRAL_COVMAT_INDEX,
        pdf_model=MOCK_PDF_MODEL,
        fit_xgrid=TEST_XGRID,
        forward_map=TEST_FORWARD_MAP_DIS,
        fast_kernel_arrays=TEST_FK_ARRAYS,
        positivity_fast_kernel_arrays=TEST_POS_FK_ARRAYS,
        penalty_posdata=MOCK_PENALTY_POSDATA,
        positivity_penalty_settings={
            "positivity_penalty": pos_penalty,
            "alpha": 1e-7,
            "lambda_positivity": 1000,
        },
        integrability_penalty=integrability_penalty,
    )

    assert_allclose(
        MOCK_CENTRAL_COVMAT_INDEX.central_values,
        log_likelihood_class.central_values,
    )
    assert_allclose(MOCK_CENTRAL_COVMAT_INDEX.covmat, log_likelihood_class.covmat)
    assert MOCK_PDF_MODEL == log_likelihood_class.pdf_model
    assert MOCK_PENALTY_POSDATA == log_likelihood_class.penalty_posdata

    # Test the __call__ method
    params = jnp.array(
        [
            2.0,
        ]
    )
    if pos_penalty:
        # -0.5 * (10.0 + 5.0) = -7.5
        assert log_likelihood_class(params) == jnp.array(
            [
                -7.5,
            ]
        )
    else:
        # -0.5 * (10.0) = -5.0
        assert log_likelihood_class(params) == jnp.array(
            [
                -5.0,
            ]
        )


@pytest.mark.parametrize("pos_penalty", [True, False])
def test_log_likelihood(pos_penalty):
    """
    Tests that the log_likeliihodd function just returns an
    LogLikelihood instance.
    """
    POS_PENALTY_SETTINGS = (
        {"positivity_penalty": pos_penalty, "alpha": 1e-7, "lambda_positivity": 1000},
    )
    log_likelihood_class = LogLikelihood(
        central_covmat_index=MOCK_CENTRAL_COVMAT_INDEX,
        pdf_model=MOCK_PDF_MODEL,
        fit_xgrid=TEST_XGRID,
        forward_map=TEST_FORWARD_MAP_DIS,
        fast_kernel_arrays=TEST_FK_ARRAYS,
        positivity_fast_kernel_arrays=TEST_POS_FK_ARRAYS,
        penalty_posdata=MOCK_PENALTY_POSDATA,
        positivity_penalty_settings=POS_PENALTY_SETTINGS,
        integrability_penalty=integrability_penalty,
    )
    log_like = log_likelihood(
        MOCK_CENTRAL_COVMAT_INDEX,
        MOCK_PDF_MODEL,
        TEST_XGRID,
        TEST_FORWARD_MAP_DIS,
        TEST_FK_ARRAYS,
        TEST_POS_FK_ARRAYS,
        MOCK_PENALTY_POSDATA,
        positivity_penalty_settings=POS_PENALTY_SETTINGS,
        integrability_penalty=integrability_penalty,
    )

    assert type(log_likelihood_class) == type(log_like)


def test_log_likelihood_with_and_without_pos_penalty():
    """
    Tests the log_likelihood function with and without positivity penalty.
    """

    # Test with positivity_penalty enabled
    positivity_penalty_settings = {
        "positivity_penalty": True,
        "alpha": 0.1,
        "lambda_positivity": 0.5,
    }

    # Instantiate the class
    log_likelihood_class = LogLikelihood(
        MOCK_CENTRAL_COVMAT_INDEX,
        MOCK_PDF_MODEL,
        TEST_XGRID,
        TEST_FORWARD_MAP_DIS,
        TEST_FK_ARRAYS,
        TEST_POS_FK_ARRAYS,
        MOCK_PENALTY_POSDATA,
        positivity_penalty_settings,
        integrability_penalty=integrability_penalty,
    )

    # Mock the params
    params = jnp.array([0.3, 0.4])

    # Call log_likelihood with positivity penalty enabled
    ll_value_with_penalty = log_likelihood_class.log_likelihood(
        params,
        log_likelihood_class.central_values,
        log_likelihood_class.inv_covmat,
        log_likelihood_class.fast_kernel_arrays,
        log_likelihood_class.positivity_fast_kernel_arrays,
    )

    # Expectation: chi2 value + penalty (5.0) => -0.5 * (10.0 + 5.0)
    assert ll_value_with_penalty == pytest.approx(-7.5)

    # Test with positivity_penalty disabled
    positivity_penalty_settings = {
        "positivity_penalty": False,
        "alpha": 0.1,
        "lambda_positivity": 0.5,
    }

    # Instantiate the class
    log_likelihood_class = LogLikelihood(
        MOCK_CENTRAL_COVMAT_INDEX,
        MOCK_PDF_MODEL,
        TEST_XGRID,
        TEST_FORWARD_MAP_DIS,
        TEST_FK_ARRAYS,
        TEST_POS_FK_ARRAYS,
        MOCK_PENALTY_POSDATA,
        positivity_penalty_settings,
        integrability_penalty=integrability_penalty,
    )

    ll_value_without_penalty = log_likelihood_class.log_likelihood(
        params,
        log_likelihood_class.central_values,
        log_likelihood_class.inv_covmat,
        log_likelihood_class.fast_kernel_arrays,
        log_likelihood_class.positivity_fast_kernel_arrays,
    )

    # Expectation: Only chi2 value, no penalty => -0.5 * (10.0)
    assert ll_value_without_penalty == pytest.approx(-5.0)


@pytest.mark.parametrize("pos_penalty", [True, False])
def test_mc_log_likelihood_with_split(pos_penalty):
    """
    Tests mc_log_likelihood returns two LogLikelihood instances when a
    train/validation split exists, and that they produce expected values.
    """

    # Create a tiny pseudodata setup consistent with TEST_N_DATA = 2
    pseudodata = jnp.array([1.0, 2.0])
    fit_covariance_matrix = jnp.eye(2)
    training_indices = jnp.array([0])
    validation_indices = jnp.array([1])

    mc_pd = MCPseudodata(
        pseudodata=pseudodata,
        training_indices=training_indices,
        validation_indices=validation_indices,
        trval_split=True,
    )

    positivity_penalty_settings = {
        "positivity_penalty": pos_penalty,
        "alpha": 1e-7,
        "lambda_positivity": 1000,
    }

    train_loglike, val_loglike = mc_log_likelihood(
        mc_pd,
        fit_covariance_matrix,
        MOCK_PDF_MODEL,
        TEST_XGRID,
        TEST_FORWARD_MAP_DIS,
        TEST_FK_ARRAYS,
        TEST_POS_FK_ARRAYS,
        MOCK_PENALTY_POSDATA,
        positivity_penalty_settings,
        integrability_penalty,
    )

    # Both should be instances of LogLikelihood
    assert isinstance(train_loglike, LogLikelihood)
    assert isinstance(val_loglike, LogLikelihood)

    params = jnp.array([0.3, 0.4])

    train_val = train_loglike(params)
    val_val = val_loglike(params)

    expected = -7.5 if pos_penalty else -5.0
    assert_allclose(train_val, jnp.array([expected]))
    assert_allclose(val_val, jnp.array([expected]))


@pytest.mark.parametrize("pos_penalty", [True, False])
def test_mc_log_likelihood_without_split_returns_nan_for_validation(pos_penalty):
    """
    Tests mc_log_likelihood when no train/validation split is requested: the
    validation log-likelihood should return NaN.
    """

    # Pseudodata across both points; training uses all when no split
    pseudodata = jnp.array([1.0, 2.0])
    fit_covariance_matrix = jnp.eye(2)
    training_indices = jnp.array([0, 1])
    validation_indices = jnp.array([])

    mc_pd = MCPseudodata(
        pseudodata=pseudodata,
        training_indices=training_indices,
        validation_indices=validation_indices,
        trval_split=False,
    )

    positivity_penalty_settings = {
        "positivity_penalty": pos_penalty,
        "alpha": 1e-7,
        "lambda_positivity": 1000,
    }

    train_loglike, val_loglike = mc_log_likelihood(
        mc_pd,
        fit_covariance_matrix,
        MOCK_PDF_MODEL,
        TEST_XGRID,
        TEST_FORWARD_MAP_DIS,
        TEST_FK_ARRAYS,
        TEST_POS_FK_ARRAYS,
        MOCK_PENALTY_POSDATA,
        positivity_penalty_settings,
        integrability_penalty,
    )

    # Train should be a LogLikelihood, validation is a callable returning NaN
    assert isinstance(train_loglike, LogLikelihood)

    params = jnp.array([0.3, 0.4])
    train_val = train_loglike(params)
    expected = -7.5 if pos_penalty else -5.0
    assert_allclose(train_val, jnp.array([expected]))

    val_val = val_loglike(params)
    assert jnp.isnan(val_val)


@pytest.mark.parametrize("pos_penalty", [True, False])
def test_LogLikelihood_call_with_batch_idx(pos_penalty):
    """
    Tests that calling LogLikelihood with a batch_idx computes the
    log-likelihood on the selected subset without errors and returns
    the expected value given our mocks.
    """

    positivity_penalty_settings = {
        "positivity_penalty": pos_penalty,
        "alpha": 1e-7,
        "lambda_positivity": 1000,
    }

    log_likelihood_class = LogLikelihood(
        central_covmat_index=MOCK_CENTRAL_COVMAT_INDEX,
        pdf_model=MOCK_PDF_MODEL,
        fit_xgrid=TEST_XGRID,
        forward_map=TEST_FORWARD_MAP_DIS,
        fast_kernel_arrays=TEST_FK_ARRAYS,
        positivity_fast_kernel_arrays=TEST_POS_FK_ARRAYS,
        penalty_posdata=MOCK_PENALTY_POSDATA,
        positivity_penalty_settings=positivity_penalty_settings,
        integrability_penalty=integrability_penalty,
    )

    params = jnp.array([0.3, 0.4])

    # Select only the first data point
    batch_idx = jnp.array([0])

    ll_value_batched = log_likelihood_class(params, batch_idx=batch_idx)

    expected = -7.5 if pos_penalty else -5.0
    assert_allclose(ll_value_batched, jnp.array([expected]))
