"""
colibri.tests.test_hessian_fit

Unit tests for the Hessian-based fit and its export routine.
"""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from colibri.hessian_fit import HessianFit, hessian_fit, run_hessian_fit
from colibri.tests.conftest import MOCK_PDF_MODEL


N_PARAMS = len(MOCK_PDF_MODEL.param_names)


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


@pytest.mark.parametrize("error_type", ["replicas", "hessian"])
def test_hessian_fit_runs_and_shapes(error_type):
    hessian_settings = {
        "iter_init": 2,
        "tolerance": 1.0,
        "ErrorType": error_type,
        # Replicas settings (ignored for ErrorType=="hessian")
        "n_samples": 7,
        "rng_key": jax.random.PRNGKey(0),
    }

    param_initialiser_settings = {"type": "zeros"}

    result = hessian_fit(
        pdf_model=MOCK_PDF_MODEL,
        log_likelihood=log_likelihood,
        optimizer_provider=MockOptimizerProvider(),
        early_stopper=MockEarlyStopper(),
        max_epochs=5,
        hessian_settings=hessian_settings,
        param_initialiser_settings=param_initialiser_settings,
    )

    assert isinstance(result, HessianFit)
    assert result.param_names == MOCK_PDF_MODEL.param_names

    # Hessian of ||p||^2 with the internal 0.5 factor should be identity
    assert_allclose(result.hessian, jnp.eye(N_PARAMS), rtol=1e-12, atol=1e-12)

    # Covariance should be tolerance^2 * I with tol=1.0
    assert_allclose(result.cov_params, jnp.eye(N_PARAMS), rtol=1e-12, atol=1e-12)

    if error_type == "replicas":
        assert result.resampled_posterior.shape == (
            hessian_settings["n_samples"],
            N_PARAMS,
        )
    else:
        # Expect 2 eigen-variations per parameter
        assert result.resampled_posterior.shape == (2 * N_PARAMS, N_PARAMS)
        # Pairs are +/- symmetric
        plus = result.resampled_posterior[0::2]
        minus = result.resampled_posterior[1::2]
        assert_allclose(plus, -minus, rtol=1e-12, atol=1e-12)


@patch("colibri.export_results.write_exportgrid")
def test_run_hessian_fit_exports(mock_write_exportgrid, tmp_path):
    # Build a small Hessian fit result using replicas mode for deterministic count
    n_samples = 5
    hessian_settings = {
        "iter_init": 1,
        "tolerance": 1.0,
        "ErrorType": "replicas",
        "n_samples": n_samples,
        "rng_key": jax.random.PRNGKey(123),
    }

    param_initialiser_settings = {"type": "zeros"}

    fit_result = hessian_fit(
        pdf_model=MOCK_PDF_MODEL,
        log_likelihood=log_likelihood,
        optimizer_provider=MockOptimizerProvider(),
        early_stopper=MockEarlyStopper(),
        max_epochs=1,
        hessian_settings=hessian_settings,
        param_initialiser_settings=param_initialiser_settings,
    )

    # Run export routine
    run_hessian_fit(fit_result, tmp_path, MOCK_PDF_MODEL)

    # One exportgrid per resampled set
    assert mock_write_exportgrid.call_count == n_samples

    # Files from export_hessian_results
    assert (tmp_path / "hessian_result.csv").exists()
    assert (tmp_path / "hessian_fit_summary.json").exists()


def test_hessian_fit_raises_when_require_local_min_fails():
    # Start away from the optimum so gradient != 0, triggering local-minimum failure
    hessian_settings = {
        "iter_init": 1,
        "tolerance": 1.0,
        "ErrorType": "replicas",
        "n_samples": 3,
        "rng_key": jax.random.PRNGKey(0),
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
            early_stopper=MockEarlyStopper(),
            max_epochs=1,
            hessian_settings=hessian_settings,
            param_initialiser_settings=param_initialiser_settings,
        )


def test_hessian_fit_logs_warning_on_local_min_check_failed(caplog):
    hessian_settings = {
        "iter_init": 1,
        "tolerance": 1.0,
        "ErrorType": "replicas",
        "n_samples": 2,
        "rng_key": jax.random.PRNGKey(0),
        "require_local_min": False,
        "grad_tol": 1e-12,  # ensure gradient not small
    }
    param_initialiser_settings = {"type": "normal", "means": 1.0, "stds": 0.0}

    with caplog.at_level("WARNING"):
        res = hessian_fit(
            pdf_model=MOCK_PDF_MODEL,
            log_likelihood=log_likelihood,
            optimizer_provider=MockOptimizerProvider(),
            early_stopper=MockEarlyStopper(),
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
        "ErrorType": "replicas",
        "n_samples": 2,
        "rng_key": jax.random.PRNGKey(0),
    }
    param_initialiser_settings = {"type": "zeros"}

    with caplog.at_level("CRITICAL"):
        hessian_fit(
            pdf_model=MOCK_PDF_MODEL,
            log_likelihood=bad_loglike,
            optimizer_provider=MockOptimizerProvider(),
            early_stopper=MockEarlyStopper(),
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


def test_hessian_fit_unknown_error_type_raises():
    hessian_settings = {
        "iter_init": 1,
        "tolerance": 1.0,
        "ErrorType": "unknown_mode",
    }
    param_initialiser_settings = {"type": "zeros"}

    with pytest.raises(ValueError):
        hessian_fit(
            pdf_model=MOCK_PDF_MODEL,
            log_likelihood=log_likelihood,
            optimizer_provider=MockOptimizerProvider(),
            early_stopper=MockEarlyStopper(),
            max_epochs=1,
            hessian_settings=hessian_settings,
            param_initialiser_settings=param_initialiser_settings,
        )
