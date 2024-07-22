from unittest.mock import Mock, patch

import copy
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose
from colibri.ultranest_fit import (
    UltranestFit,
    run_ultranest_fit,
    ultranest_fit,
    UltraNestLogLikelihood,
)
from colibri.loss_functions import chi2
from colibri.tests.conftest import (
    MOCK_CENTRAL_INV_COVMAT_INDEX,
    MOCK_PDF_MODEL,
    TEST_FORWARD_MAP,
    TEST_FK_ARRAYS,
    TEST_POS_FK_ARRAYS,
    UltraNestLogLikelihoodMock,
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


def test_UltraNestLogLikelihood_class():
    """
    Tests the UltraNestLogLikelihood class.
    """
    ultranest_loglike = UltraNestLogLikelihood(
        central_inv_covmat_index=MOCK_CENTRAL_INV_COVMAT_INDEX,
        pdf_model=MOCK_PDF_MODEL,
        fit_xgrid=FIT_XGRID,
        forward_map=TEST_FORWARD_MAP,
        fast_kernel_arrays=TEST_FK_ARRAYS,
        positivity_fast_kernel_arrays=TEST_POS_FK_ARRAYS,
        ns_settings=ns_settings,
        chi2=mock_chi2,
        penalty_posdata=_penalty_posdata,
        alpha=1e-7,
        lambda_positivity=1000,
    )

    assert_allclose(
        MOCK_CENTRAL_INV_COVMAT_INDEX.central_values, ultranest_loglike.central_values
    )
    assert_allclose(
        MOCK_CENTRAL_INV_COVMAT_INDEX.inv_covmat, ultranest_loglike.inv_covmat
    )
    assert MOCK_PDF_MODEL == ultranest_loglike.pdf_model
    assert _penalty_posdata == ultranest_loglike.penalty_posdata


@patch("colibri.ultranest_fit.jax.vmap")
def test_UltraNestLogLikelihood_vect_class(mock_jax_vmap):
    """
    Tests the UltraNestLogLikelihood class with vectorized ReactiveNS settings.
    """

    ultranest_loglike = UltraNestLogLikelihood(
        central_inv_covmat_index=MOCK_CENTRAL_INV_COVMAT_INDEX,
        pdf_model=MOCK_PDF_MODEL,
        fit_xgrid=FIT_XGRID,
        forward_map=TEST_FORWARD_MAP,
        fast_kernel_arrays=TEST_FK_ARRAYS,
        positivity_fast_kernel_arrays=TEST_POS_FK_ARRAYS,
        ns_settings=vect_ns_settings,
        chi2=mock_chi2,
        penalty_posdata=_penalty_posdata,
        alpha=1e-7,
        lambda_positivity=1000,
    )

    assert_allclose(
        MOCK_CENTRAL_INV_COVMAT_INDEX.central_values, ultranest_loglike.central_values
    )
    assert_allclose(
        MOCK_CENTRAL_INV_COVMAT_INDEX.inv_covmat, ultranest_loglike.inv_covmat
    )
    assert MOCK_PDF_MODEL == ultranest_loglike.pdf_model

    assert mock_jax_vmap.call_count == 3


def test_ultranest_fit():
    # Create mock pdf model
    mock_pdf_model = Mock()
    mock_pdf_model.param_names = ["param1", "param2"]
    mock_pdf_model.grid_values_func = lambda xgrid: lambda params: jnp.ones(
        (14, len(xgrid))
    )
    mock_pdf_model.pred_and_pdf_func = lambda xgrid, forward_map: (
        lambda params, fast_kernel_arrays: (params, jnp.ones((14, len(xgrid))))
    )
    _pred_data = None
    mock_log_likelihood = UltraNestLogLikelihoodMock(
        MOCK_CENTRAL_INV_COVMAT_INDEX,
        mock_pdf_model,
        FIT_XGRID,
        _pred_data,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        ns_settings,
        chi2,
        _penalty_posdata,
        alpha=1e-7,
        lambda_positivity=0,
    )

    fit_result = ultranest_fit(
        mock_pdf_model,
        bayesian_prior,
        ns_settings,
        mock_log_likelihood,
    )

    assert isinstance(fit_result, UltranestFit)
    assert fit_result.resampled_posterior.shape == (
        ns_settings["n_posterior_samples"],
        len(mock_pdf_model.param_names),
    )
    assert fit_result.param_names == ["param1", "param2"]
    assert fit_result.ultranest_specs == ns_settings
    assert isinstance(fit_result.ultranest_result, dict)


def test_ultranest_fit_vectorized():
    # Create mock pdf model
    mock_pdf_model = Mock()
    mock_pdf_model.param_names = ["param1", "param2"]
    mock_pdf_model.grid_values_func = lambda xgrid: lambda params: jnp.ones(
        (14, len(xgrid))
    )
    mock_pdf_model.pred_and_pdf_func = lambda xgrid, forward_map: (
        lambda params, fast_kernel_arrays: (params, jnp.ones((14, len(xgrid))))
    )
    _pred_data = None
    ns_settings["ReactiveNS_settings"]["vectorized"] = True

    mock_log_likelihood = UltraNestLogLikelihoodMock(
        MOCK_CENTRAL_INV_COVMAT_INDEX,
        mock_pdf_model,
        FIT_XGRID,
        _pred_data,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        ns_settings,
        chi2,
        _penalty_posdata,
        alpha=1e-7,
        lambda_positivity=0,
    )

    fit_result = ultranest_fit(
        mock_pdf_model,
        bayesian_prior,
        ns_settings,
        mock_log_likelihood,
    )

    assert isinstance(fit_result, UltranestFit)
    assert fit_result.resampled_posterior.shape == (
        ns_settings["n_posterior_samples"],
        len(mock_pdf_model.param_names),
    )
    assert fit_result.param_names == ["param1", "param2"]
    assert fit_result.ultranest_specs == ns_settings
    assert isinstance(fit_result.ultranest_result, dict)


def test_ultranest_fit_with_SliceSampler():
    ns_settings = {
        "ultranest_seed": 42,
        "ReactiveNS_settings": {"vectorized": False},
        "SliceSampler_settings": {"nsteps": 10},
        "Run_settings": {"frac_remain": 0.5, "min_num_live_points": 5},
        "n_posterior_samples": 10,
        "posterior_resampling_seed": 123,
        "sampler_plot": False,
        "popstepsampler": False,
    }
    # Create mock pdf model
    mock_pdf_model = Mock()
    mock_pdf_model.param_names = ["param1", "param2"]
    mock_pdf_model.grid_values_func = lambda xgrid: lambda params: jnp.ones(
        (14, len(xgrid))
    )
    mock_pdf_model.pred_and_pdf_func = lambda xgrid, forward_map: (
        lambda params, fast_kernel_arrays: (params, jnp.ones((14, len(xgrid))))
    )
    _pred_data = None
    
    mock_log_likelihood = UltraNestLogLikelihoodMock(
        MOCK_CENTRAL_INV_COVMAT_INDEX,
        mock_pdf_model,
        FIT_XGRID,
        _pred_data,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        ns_settings,
        chi2,
        _penalty_posdata,
        alpha=1e-7,
        lambda_positivity=0,
    )

    fit_result = ultranest_fit(
        mock_pdf_model,
        bayesian_prior,
        ns_settings,
        mock_log_likelihood,
    )

    assert isinstance(fit_result, UltranestFit)
    assert fit_result.resampled_posterior.shape == (
        ns_settings["n_posterior_samples"],
        len(mock_pdf_model.param_names),
    )
    assert fit_result.param_names == ["param1", "param2"]
    assert fit_result.ultranest_specs == ns_settings
    assert isinstance(fit_result.ultranest_result, dict)


def test_ultranest_fit_with_popSliceSampler():
    ns_settings = {
        "ultranest_seed": 42,
        "ReactiveNS_settings": {"vectorized": False},
        "SliceSampler_settings": {"nsteps": 10, "popsize": 10},
        "Run_settings": {"frac_remain": 0.5, "min_num_live_points": 5},
        "n_posterior_samples": 10,
        "posterior_resampling_seed": 123,
        "sampler_plot": False,
        "popstepsampler": True,
    }
    # Create mock pdf model
    mock_pdf_model = Mock()
    mock_pdf_model.param_names = ["param1", "param2"]
    mock_pdf_model.grid_values_func = lambda xgrid: lambda params: jnp.ones(
        (14, len(xgrid))
    )
    mock_pdf_model.pred_and_pdf_func = lambda xgrid, forward_map: (
        lambda params, fast_kernel_arrays: (params, jnp.ones((14, len(xgrid))))
    )
    _pred_data = None

    fit_result = ultranest_fit(
        MOCK_CENTRAL_INV_COVMAT_INDEX,
        _pred_data,
        _penalty_posdata,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        mock_pdf_model,
        bayesian_prior,
        ns_settings,
        FIT_XGRID,
    )

    assert isinstance(fit_result, UltranestFit)
    assert fit_result.resampled_posterior.shape == (
        ns_settings["n_posterior_samples"],
        len(mock_pdf_model.param_names),
    )
    assert fit_result.param_names == ["param1", "param2"]
    assert fit_result.ultranest_specs == ns_settings
    assert isinstance(fit_result.ultranest_result, dict)


@patch("colibri.export_results.write_exportgrid")
def test_run_ultranest_fit(mock_write_exportgrid, tmp_path):

    # Define mock ultranest fit
    mock_ultranest_fit = Mock()
    mock_ultranest_fit.resampled_posterior = jax.random.normal(
        jax.random.PRNGKey(0), (10, 2)
    )
    mock_ultranest_fit.param_names = ["param1", "param2"]
    mock_ultranest_fit.full_posterior_samples = jax.random.normal(
        jax.random.PRNGKey(0), (100, 2)
    )
    mock_ultranest_fit.bayes_complexity = 2.0
    mock_ultranest_fit.avg_chi2 = 0.3
    mock_ultranest_fit.min_chi2 = 0.1
    mock_ultranest_fit.logz = 7.0

    # Create mock pdf model
    mock_pdf_model = Mock()
    mock_pdf_model.param_names = ["param1", "param2"]

    # Run the run_ultranest_fit function
    output_path = str(tmp_path)
    run_ultranest_fit(mock_ultranest_fit, output_path, mock_pdf_model)

    # Check if the write_exportgrid function was called for each sample
    assert (
        mock_write_exportgrid.call_count
        == mock_ultranest_fit.resampled_posterior.shape[0]
    )

    # Assertions - check if files are created in the output path
    assert (tmp_path / "ns_result.csv").exists()
    assert (tmp_path / "bayes_metrics.csv").exists()
    assert (tmp_path / "full_posterior_sample.csv").exists()
