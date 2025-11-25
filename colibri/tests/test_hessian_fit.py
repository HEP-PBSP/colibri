"""
colibri.tests.test_hessian_fit

Unit tests for the Hessian-based fit and its export routine.
"""

from unittest.mock import patch

import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from colibri.hessian_fit import HessianFit, hessian_fit, run_hessian_fit
from colibri.tests.conftest import MOCK_PDF_MODEL


N_PARAMS = len(MOCK_PDF_MODEL.full_param_names)


class MockOptimizerProvider:
    @staticmethod
    def init(parameters):
        # optax-like API: return an optimizer state; not used in mock
        return jnp.zeros_like(parameters)

    @staticmethod
    def update(grads, opt_state, params):
        # Return zero updates so params stay constant; maintain API shape
        return jnp.zeros_like(params), opt_state


class MockEarlyStopper:
    def __init__(self):
        self.should_stop = False

    def update(self, epoch_val_loss):
        # Never early stop (validation loss is NaN in Hessian fit loop)
        self.should_stop = False
        return self


# Simple concave log-likelihood around 0 so chi2(p) = ||p||^2
log_likelihood = lambda p: -0.5 * jnp.sum(p**2)


def test_hessian_fit_runs_and_shapes():
    hessian_settings = {
        "iter_init": 2,
        "tolerance": 1.0,
        "n_eigvec": 20,
        "rng_seed": 0,
    }

    param_initialiser_settings = {"type": "zeros"}

    result = hessian_fit(
        pdf_model=MOCK_PDF_MODEL,
        log_likelihood=log_likelihood,
        optimizer_provider=MockOptimizerProvider(),
        max_epochs=5,
        hessian_settings=hessian_settings,
        param_initialiser_settings=param_initialiser_settings,
    )

    assert isinstance(result, HessianFit)
    assert result.full_param_names == MOCK_PDF_MODEL.full_param_names

    # Hessian of ||p||^2 with the internal 0.5 factor should be identity
    assert_allclose(result.hessian, jnp.eye(N_PARAMS), rtol=1e-12, atol=1e-12)

    # Covariance should be tolerance^2 * I with tol=1.0
    assert_allclose(result.cov_params, jnp.eye(N_PARAMS), rtol=1e-12, atol=1e-12)

    # Expect 2 eigen-variations per parameter
    assert result.resampled_posterior.shape == (2 * N_PARAMS, N_PARAMS)
    # Pairs are +/- symmetric
    plus = result.resampled_posterior[0::2]
    minus = result.resampled_posterior[1::2]
    assert_allclose(plus, -minus, rtol=1e-12, atol=1e-12)


@patch("colibri.export_results.write_exportgrid")
def test_run_hessian_fit_exports(mock_write_exportgrid, tmp_path):

    pdf_replicas = 4  # these will be 2 times n_eigvec
    hessian_settings = {
        "iter_init": 1,
        "tolerance": 1.0,
        "n_eigvec": 2,
        "rng_seed": 123,
    }

    param_initialiser_settings = {"type": "zeros"}

    fit_result = hessian_fit(
        pdf_model=MOCK_PDF_MODEL,
        log_likelihood=log_likelihood,
        optimizer_provider=MockOptimizerProvider(),
        max_epochs=1,
        hessian_settings=hessian_settings,
        param_initialiser_settings=param_initialiser_settings,
    )

    # Run export routine
    run_hessian_fit(fit_result, tmp_path, MOCK_PDF_MODEL, Q0=1.65)

    # One exportgrid per resampled set
    assert mock_write_exportgrid.call_count == pdf_replicas

    # Files from export_hessian_results
    assert (tmp_path / "hessian_result.csv").exists()
    assert (tmp_path / "hessian_fit_summary.json").exists()


def test_hessian_fit_raises_when_require_local_min_fails():
    # Start away from the optimum so gradient != 0, triggering local-minimum failure
    hessian_settings = {
        "iter_init": 1,
        "tolerance": 1.0,
        "rng_seed": 0,
        "n_eigvec": 2,
        "require_local_min": True,
        "grad_tol": 1e-12,  # very small to ensure failure
    }

    # Force non-zero initial params via Normal with mean=1, std=0
    param_initialiser_settings = {"type": "normal", "means": 1.0, "stds": 0.0}

    with pytest.raises(ValueError):
        hessian_fit(
            pdf_model=MOCK_PDF_MODEL,
            log_likelihood=log_likelihood,
            optimizer_provider=MockOptimizerProvider(),
            max_epochs=1,
            hessian_settings=hessian_settings,
            param_initialiser_settings=param_initialiser_settings,
        )


def test_hessian_fit_logs_warning_on_local_min_check_failed(caplog):
    hessian_settings = {
        "iter_init": 1,
        "tolerance": 1.0,
        "rng_seed": 0,
        "n_eigvec": 2,
        "require_local_min": False,
        "grad_tol": 1e-12,  # ensure gradient not small
    }
    param_initialiser_settings = {"type": "normal", "means": 1.0, "stds": 0.0}

    with caplog.at_level("WARNING"):
        res = hessian_fit(
            pdf_model=MOCK_PDF_MODEL,
            log_likelihood=log_likelihood,
            optimizer_provider=MockOptimizerProvider(),
            max_epochs=1,
            hessian_settings=hessian_settings,
            param_initialiser_settings=param_initialiser_settings,
        )

    # Expect a warning about local minimum check failed
    assert any(
        (rec.levelname == "WARNING" and "Local minimum check failed" in rec.message)
        for rec in caplog.records
    )
    assert isinstance(res, HessianFit)


def test_hessian_fit_logs_critical_on_non_pd_hessian(caplog):
    # Define a convex log-likelihood so chi2 is negative definite -> Hessian not PD
    bad_loglike = lambda p: 0.5 * jnp.sum(p**2)

    hessian_settings = {
        "iter_init": 1,
        "tolerance": 1.0,
        "n_eigvec": 2,
        "rng_seed": 0,
    }
    param_initialiser_settings = {"type": "zeros"}

    with caplog.at_level("CRITICAL"):
        hessian_fit(
            pdf_model=MOCK_PDF_MODEL,
            log_likelihood=bad_loglike,
            optimizer_provider=MockOptimizerProvider(),
            max_epochs=1,
            hessian_settings=hessian_settings,
            param_initialiser_settings=param_initialiser_settings,
        )

    assert any(
        (
            rec.levelname == "CRITICAL"
            and "Hessian matrix is not positive definite" in rec.message
        )
        for rec in caplog.records
    )


def test_hessian_fit_raises_on_nonfinite_min_chi2():
    # Define a log-likelihood that always returns NaN, making chi2 non-finite
    nan_loglike = lambda p: jnp.nan

    hessian_settings = {
        "iter_init": 1,
        "tolerance": 1.0,
        "n_eigvec": 2,
        "rng_seed": 0,
    }
    param_initialiser_settings = {"type": "zeros"}

    with pytest.raises(ValueError):
        hessian_fit(
            pdf_model=MOCK_PDF_MODEL,
            log_likelihood=nan_loglike,
            optimizer_provider=MockOptimizerProvider(),
            max_epochs=1,
            hessian_settings=hessian_settings,
            param_initialiser_settings=param_initialiser_settings,
        )
