"""
colibri.tests.test_likelihood.py

Tests for the likelihood module.
"""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from colibri.likelihood import LogLikelihood, log_likelihood, mc_log_likelihood
from colibri.mc_utils import MCPseudodata
from colibri.tests.conftest import (
    MOCK_CENTRAL_SQRT_COVMAT_INDEX,
    MOCK_PDF_MODEL,
    MOCK_PENALTY_POSDATA,
    TEST_FK_ARRAYS,
    TEST_FORWARD_MAP_DIS,
    TEST_POS_FK_ARRAYS,
    TEST_XGRID,
)
from colibri.data_batch import BatchSpec

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
        central_sqrt_covmat_index=MOCK_CENTRAL_SQRT_COVMAT_INDEX,
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
        MOCK_CENTRAL_SQRT_COVMAT_INDEX.central_values,
        log_likelihood_class.central_values,
    )
    assert_allclose(
        MOCK_CENTRAL_SQRT_COVMAT_INDEX.sqrt_covmat, log_likelihood_class.sqrt_covmat
    )
    assert MOCK_PDF_MODEL == log_likelihood_class.pdf_model
    assert MOCK_PENALTY_POSDATA == log_likelihood_class.penalty_posdata

    # Test the __call__ method
    params = jnp.array(
        [
            2.0,
        ]
    )
    # Compute expected value using actual prediction and covariance
    predictions, pdf = log_likelihood_class.pred_and_pdf(
        params, log_likelihood_class.fast_kernel_arrays
    )
    predictions = predictions[log_likelihood_class.central_values_idx]
    diff = predictions - log_likelihood_class.central_values
    z = jnp.einsum("ij, j", log_likelihood_class.inv_sqrt_covmat, diff)
    chi2_val = jnp.dot(z, z)

    pos_pen = (
        jnp.sum(
            log_likelihood_class.penalty_posdata(
                pdf,
                log_likelihood_class.positivity_penalty_settings["alpha"],
                log_likelihood_class.positivity_penalty_settings["lambda_positivity"],
                log_likelihood_class.positivity_fast_kernel_arrays,
            ),
            axis=-1,
        )
        if pos_penalty
        else 0.0
    )
    integ_pen = jnp.sum(integrability_penalty(pdf), axis=-1)
    expected = -0.5 * (chi2_val + pos_pen + integ_pen)

    assert_allclose(float(log_likelihood_class(params)), float(expected))


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
        central_sqrt_covmat_index=MOCK_CENTRAL_SQRT_COVMAT_INDEX,
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
        MOCK_CENTRAL_SQRT_COVMAT_INDEX,
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
        MOCK_CENTRAL_SQRT_COVMAT_INDEX,
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
        log_likelihood_class.inv_sqrt_covmat,
        log_likelihood_class.fast_kernel_arrays,
        log_likelihood_class.positivity_fast_kernel_arrays,
    )

    # Compute expectation directly: -0.5 * (chi2 + pos_pen + integ_pen)
    predictions, pdf = log_likelihood_class.pred_and_pdf(
        params, log_likelihood_class.fast_kernel_arrays
    )
    predictions = predictions[log_likelihood_class.central_values_idx]
    diff = predictions - log_likelihood_class.central_values
    z = jnp.einsum("ij, j", log_likelihood_class.inv_sqrt_covmat, diff)

    chi2_val = jnp.dot(z, z)
    pos_pen = jnp.sum(
        log_likelihood_class.penalty_posdata(
            pdf,
            positivity_penalty_settings["alpha"],
            positivity_penalty_settings["lambda_positivity"],
            log_likelihood_class.positivity_fast_kernel_arrays,
        ),
        axis=-1,
    )
    integ_pen = jnp.sum(integrability_penalty(pdf), axis=-1)
    expected_with_penalty = -0.5 * (chi2_val + pos_pen + integ_pen)
    assert float(ll_value_with_penalty) == pytest.approx(float(expected_with_penalty))

    # Test with positivity_penalty disabled
    positivity_penalty_settings = {
        "positivity_penalty": False,
        "alpha": 0.1,
        "lambda_positivity": 0.5,
    }

    # Instantiate the class
    log_likelihood_class = LogLikelihood(
        MOCK_CENTRAL_SQRT_COVMAT_INDEX,
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
        log_likelihood_class.inv_sqrt_covmat,
        log_likelihood_class.fast_kernel_arrays,
        log_likelihood_class.positivity_fast_kernel_arrays,
    )

    # Expectation: Only chi2 value (penalties zeroed)
    predictions, pdf = log_likelihood_class.pred_and_pdf(
        params, log_likelihood_class.fast_kernel_arrays
    )
    predictions = predictions[log_likelihood_class.central_values_idx]
    diff = predictions - log_likelihood_class.central_values
    z = jnp.einsum("ij, j", log_likelihood_class.inv_sqrt_covmat, diff)
    chi2_val = jnp.dot(z, z)
    expected_without_penalty = -0.5 * chi2_val
    assert float(ll_value_without_penalty) == pytest.approx(
        float(expected_without_penalty)
    )


@pytest.mark.parametrize("pos_penalty", [True, False])
def test_mc_log_likelihood_with_split(pos_penalty):
    """
    Tests mc_log_likelihood returns two LogLikelihood instances when a
    train/validation split exists, and that they produce expected values.
    """

    # Create a tiny pseudodata setup consistent with TEST_N_DATA = 2
    pseudodata = jnp.array([1.0, 2.0])
    general_covariance_matrix = jnp.eye(2)
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
        general_covariance_matrix,
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

    # Compute expected for train and validation independently
    def compute_expected(ll_obj):
        preds, pdf = ll_obj.pred_and_pdf(params, ll_obj.fast_kernel_arrays)
        preds = preds[ll_obj.central_values_idx]
        diff = preds - ll_obj.central_values
        inv = ll_obj.inv_sqrt_covmat.T @ ll_obj.inv_sqrt_covmat
        chi2_val = jnp.einsum("i,ij,j", diff, inv, diff)
        pos_pen = (
            jnp.sum(
                ll_obj.penalty_posdata(
                    pdf,
                    ll_obj.positivity_penalty_settings["alpha"],
                    ll_obj.positivity_penalty_settings["lambda_positivity"],
                    ll_obj.positivity_fast_kernel_arrays,
                ),
                axis=-1,
            )
            if pos_penalty
            else 0.0
        )
        integ_pen = jnp.sum(integrability_penalty(pdf), axis=-1)
        return -0.5 * (chi2_val + pos_pen + integ_pen)

    expected_train = compute_expected(train_loglike)
    expected_val = compute_expected(val_loglike)

    assert_allclose(float(train_val), float(expected_train))
    assert_allclose(float(val_val), float(expected_val))


@pytest.mark.parametrize("pos_penalty", [True, False])
def test_mc_log_likelihood_without_split_returns_nan_for_validation(pos_penalty):
    """
    Tests mc_log_likelihood when no train/validation split is requested: the
    validation log-likelihood should return NaN.
    """

    # Pseudodata across both points; training uses all when no split
    pseudodata = jnp.array([1.0, 2.0])
    general_covariance_matrix = jnp.eye(2)
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
        general_covariance_matrix,
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
    # Compute expected train value
    predictions, pdf = train_loglike.pred_and_pdf(
        params, train_loglike.fast_kernel_arrays
    )
    predictions = predictions[train_loglike.central_values_idx]
    diff = predictions - train_loglike.central_values
    z = jnp.einsum("ij, j", train_loglike.inv_sqrt_covmat, diff)
    chi2_val = jnp.dot(z, z)
    pos_pen = (
        jnp.sum(
            train_loglike.penalty_posdata(
                pdf,
                train_loglike.positivity_penalty_settings["alpha"],
                train_loglike.positivity_penalty_settings["lambda_positivity"],
                train_loglike.positivity_fast_kernel_arrays,
            ),
            axis=-1,
        )
        if pos_penalty
        else 0.0
    )
    integ_pen = jnp.sum(integrability_penalty(pdf), axis=-1)
    expected = -0.5 * (chi2_val + pos_pen + integ_pen)
    assert_allclose(float(train_val), float(expected))

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
        central_sqrt_covmat_index=MOCK_CENTRAL_SQRT_COVMAT_INDEX,
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
    batch = BatchSpec(idx=jnp.array([0]))

    ll_value_batched = log_likelihood_class(params, batch=batch)

    # Compute expected on the batch index: recompute inv_covmat on the sub-covmat
    predictions, pdf = log_likelihood_class.pred_and_pdf(
        params, log_likelihood_class.fast_kernel_arrays
    )
    predictions = predictions[log_likelihood_class.central_values_idx]
    predictions_b = predictions[batch.idx]
    central_b = log_likelihood_class.central_values[batch.idx]
    cov_b = log_likelihood_class.sqrt_covmat[batch.idx][:, batch.idx]
    inv_b = jnp.linalg.inv(cov_b)
    diff_b = predictions_b - central_b
    chi2_b = jnp.einsum("i,ij,j", diff_b, inv_b, diff_b)
    pos_pen = (
        jnp.sum(
            log_likelihood_class.penalty_posdata(
                pdf,
                log_likelihood_class.positivity_penalty_settings["alpha"],
                log_likelihood_class.positivity_penalty_settings["lambda_positivity"],
                log_likelihood_class.positivity_fast_kernel_arrays,
            ),
            axis=-1,
        )
        if pos_penalty
        else 0.0
    )
    integ_pen = jnp.sum(integrability_penalty(pdf), axis=-1)
    expected = -0.5 * (chi2_b + pos_pen + integ_pen)
    assert_allclose(float(ll_value_batched), float(expected))


@pytest.mark.parametrize("pos_penalty", [True, False])
def test_LogLikelihood_call_with_batch_with_inv_cov(pos_penalty):
    """
    Tests that calling LogLikelihood with a BatchSpec that already has a
    precomputed `inv_cov` uses the provided inverse covariance instead of
    recomputing it.
    """

    positivity_penalty_settings = {
        "positivity_penalty": pos_penalty,
        "alpha": 1e-7,
        "lambda_positivity": 1000,
    }

    log_likelihood_class = LogLikelihood(
        central_sqrt_covmat_index=MOCK_CENTRAL_SQRT_COVMAT_INDEX,
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

    # Select first two data points and precompute their inverse covariance
    batch_idx = jnp.array([0, 1])
    cov_b = log_likelihood_class.sqrt_covmat[batch_idx][:, batch_idx]
    inv_b = jnp.linalg.inv(cov_b)

    # Provide the precomputed inverse covariance in the BatchSpec
    batch = BatchSpec(idx=batch_idx, inv_cov=inv_b)

    ll_value_batched = log_likelihood_class(params, batch=batch)

    # Compute expected value using the provided inv_b (should be identical)
    predictions, pdf = log_likelihood_class.pred_and_pdf(
        params, log_likelihood_class.fast_kernel_arrays
    )
    predictions = predictions[log_likelihood_class.central_values_idx]
    predictions_b = predictions[batch.idx]
    central_b = log_likelihood_class.central_values[batch.idx]
    diff_b = predictions_b - central_b
    chi2_b = jnp.einsum("i,ij,j", diff_b, inv_b, diff_b)
    pos_pen = (
        jnp.sum(
            log_likelihood_class.penalty_posdata(
                pdf,
                log_likelihood_class.positivity_penalty_settings["alpha"],
                log_likelihood_class.positivity_penalty_settings["lambda_positivity"],
                log_likelihood_class.positivity_fast_kernel_arrays,
            ),
            axis=-1,
        )
        if pos_penalty
        else 0.0
    )
    integ_pen = jnp.sum(integrability_penalty(pdf), axis=-1)
    expected = -0.5 * (chi2_b + pos_pen + integ_pen)

    assert_allclose(float(ll_value_batched), float(expected))
