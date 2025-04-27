"""
colibri.tests.test_likelihood.py

Tests for the likelihood module.
"""

import copy
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from colibri.likelihood import LogLikelihood, log_likelihood
from colibri.tests.conftest import (
    MOCK_CENTRAL_INV_COVMAT_INDEX,
    MOCK_PDF_MODEL,
    TEST_FK_ARRAYS,
    TEST_FORWARD_MAP,
    TEST_POS_FK_ARRAYS,
)

jax.config.update("jax_enable_x64", True)

# Define mock input parameters
bayesian_prior = lambda x: x
FIT_XGRID = jnp.logspace(-7, 0, 50)
fast_kernel_arrays = [jnp.eye(2)]
positivity_fast_kernel_arrays = [jnp.eye(2)]
mock_chi2 = lambda central_values, predictions, inv_covmat: 0.0

_penalty_posdata = (
    lambda pdf, alpha, lambda_positivity, positivity_fast_kernel_arrays: jnp.array(
        [1.0, 1.0]
    )
)

integrability_penalty = lambda pdf: jnp.array([0.0])

ns_settings = {
    "ultranest_seed": 42,
    "ReactiveNS_settings": {"vectorized": False},
    "SliceSampler_settings": None,
    "Run_settings": {"frac_remain": 0.5, "min_num_live_points": 5},
    "n_posterior_samples": 10,
    "posterior_resampling_seed": 123,
    "sampler_plot": False,
}

vect_ns_settings = copy.deepcopy(ns_settings)
vect_ns_settings["ReactiveNS_settings"]["vectorized"] = True


@pytest.mark.parametrize("pos_penalty", [True, False])
def test_LogLikelihood_class(pos_penalty):
    """
    Tests the LogLikelihood class.
    """
    ultranest_loglike = LogLikelihood(
        central_inv_covmat_index=MOCK_CENTRAL_INV_COVMAT_INDEX,
        pdf_model=MOCK_PDF_MODEL,
        fit_xgrid=FIT_XGRID,
        forward_map=TEST_FORWARD_MAP,
        fast_kernel_arrays=TEST_FK_ARRAYS,
        positivity_fast_kernel_arrays=TEST_POS_FK_ARRAYS,
        ns_settings=ns_settings,
        chi2=mock_chi2,
        penalty_posdata=_penalty_posdata,
        positivity_penalty_settings={
            "positivity_penalty": pos_penalty,
            "alpha": 1e-7,
            "lambda_positivity": 1000,
        },
        integrability_penalty=integrability_penalty,
    )

    assert_allclose(
        MOCK_CENTRAL_INV_COVMAT_INDEX.central_values, ultranest_loglike.central_values
    )
    assert_allclose(
        MOCK_CENTRAL_INV_COVMAT_INDEX.inv_covmat, ultranest_loglike.inv_covmat
    )
    assert MOCK_PDF_MODEL == ultranest_loglike.pdf_model
    assert _penalty_posdata == ultranest_loglike.penalty_posdata


@pytest.mark.parametrize("pos_penalty", [True, False])
@patch("colibri.ultranest_fit.jax.vmap")
def test_LogLikelihood_vect_class(mock_jax_vmap, pos_penalty):
    """
    Tests the LogLikelihood class with vectorized ReactiveNS settings.
    """

    ultranest_loglike = LogLikelihood(
        central_inv_covmat_index=MOCK_CENTRAL_INV_COVMAT_INDEX,
        pdf_model=MOCK_PDF_MODEL,
        fit_xgrid=FIT_XGRID,
        forward_map=TEST_FORWARD_MAP,
        fast_kernel_arrays=TEST_FK_ARRAYS,
        positivity_fast_kernel_arrays=TEST_POS_FK_ARRAYS,
        ns_settings=vect_ns_settings,
        chi2=mock_chi2,
        penalty_posdata=_penalty_posdata,
        positivity_penalty_settings={
            "positivity_penalty": pos_penalty,
            "alpha": 1e-7,
            "lambda_positivity": 1000,
        },
        integrability_penalty=integrability_penalty,
    )

    assert_allclose(
        MOCK_CENTRAL_INV_COVMAT_INDEX.central_values, ultranest_loglike.central_values
    )
    assert_allclose(
        MOCK_CENTRAL_INV_COVMAT_INDEX.inv_covmat, ultranest_loglike.inv_covmat
    )
    assert MOCK_PDF_MODEL == ultranest_loglike.pdf_model

    assert mock_jax_vmap.call_count == 4


@pytest.mark.parametrize("pos_penalty", [True, False])
def test_log_likelihood(pos_penalty):
    """
    Tests that the log_likeliihodd function just returns an
    LogLikelihood instance.
    """
    POS_PENALTY_SETTINGS = (
        {"positivity_penalty": pos_penalty, "alpha": 1e-7, "lambda_positivity": 1000},
    )
    ultranest_loglike = LogLikelihood(
        central_inv_covmat_index=MOCK_CENTRAL_INV_COVMAT_INDEX,
        pdf_model=MOCK_PDF_MODEL,
        fit_xgrid=FIT_XGRID,
        forward_map=TEST_FORWARD_MAP,
        fast_kernel_arrays=TEST_FK_ARRAYS,
        positivity_fast_kernel_arrays=TEST_POS_FK_ARRAYS,
        ns_settings=ns_settings,
        chi2=mock_chi2,
        penalty_posdata=_penalty_posdata,
        positivity_penalty_settings=POS_PENALTY_SETTINGS,
        integrability_penalty=integrability_penalty,
    )
    log_like = log_likelihood(
        MOCK_CENTRAL_INV_COVMAT_INDEX,
        MOCK_PDF_MODEL,
        FIT_XGRID,
        TEST_FORWARD_MAP,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        ns_settings,
        _penalty_posdata,
        positivity_penalty_settings=POS_PENALTY_SETTINGS,
        integrability_penalty=integrability_penalty,
    )

    assert type(ultranest_loglike) == type(log_like)


def test_log_likelihood_with_and_without_pos_penalty():
    # Mocking inputs
    central_inv_covmat_index = MagicMock()
    central_inv_covmat_index.central_values = jnp.array([1.0, 2.0])
    central_inv_covmat_index.inv_covmat = jnp.eye(2)

    pdf_model = MagicMock()
    pdf_model.pred_and_pdf_func = MagicMock(
        return_value=lambda params, fast_kernel_arrays: (params, params)
    )

    # Simple forward map placeholder
    forward_map = lambda x: x

    # Dummy FK tables
    fast_kernel_arrays = (jnp.array([1.0]),)
    positivity_fast_kernel_arrays = (jnp.array([1.0]),)

    ns_settings = {"ReactiveNS_settings": {"vectorized": False}}

    # Mocking chi2 and penalty_posdata
    chi2_mock = MagicMock(return_value=10.0)
    penalty_posdata_mock = MagicMock(return_value=jnp.array([5.0]))

    # Test with positivity_penalty enabled
    positivity_penalty_settings = {
        "positivity_penalty": True,
        "alpha": 0.1,
        "lambda_positivity": 0.5,
    }

    # Instantiate the class
    log_likelihood_class = LogLikelihood(
        central_inv_covmat_index,
        pdf_model,
        jnp.array([0.1, 0.2]),  # dummy fit_xgrid
        forward_map,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        ns_settings,
        chi2_mock,
        penalty_posdata_mock,
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
        central_inv_covmat_index,
        pdf_model,
        jnp.array([0.1, 0.2]),  # dummy fit_xgrid
        forward_map,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        ns_settings,
        chi2_mock,
        penalty_posdata_mock,
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
