"""
Module for testing that the time per evaluation of the likelihood is below a certain threshold.
"""

import jax
import time
import pytest
from colibri.api import API as colibriAPI
from colibri.bayes_prior import bayesian_prior
from colibri.tests.conftest import (
    T0_PDFSET,
    TEST_FULL_DIS_DATASET,
    TEST_FULL_POS_DATASET,
    TEST_FULL_HAD_DATASET,
    TEST_FULL_GLOBAL_DATASET,
)
from wmin.api import API as wminAPI
from wmin.tests.wmin_conftest import (
    TEST_PRIOR_SETTINGS_WMIN,
    TEST_WMIN_SETTINGS_NBASIS_10,
    TEST_WMIN_SETTINGS_NBASIS_100,
)

N_LOOP_ITERATIONS = 100
THRESHOLD_TIME_DIS = 1e-2
THRESHOLD_TIME_DIS_POS = 3e-2
THRESHOLD_TIME_HAD = 3e-1
THRESHOLD_TIME_HAD_POS = 3e-1
THRESHOLD_TIME_GLOBAL = 3e-1
THRESHOLD_TIME_GLOBAL_POS = 3e-1


@pytest.mark.parametrize(
    "wmin_model_settings", [TEST_WMIN_SETTINGS_NBASIS_10, TEST_WMIN_SETTINGS_NBASIS_100]
)
def test_likelihood_dis_wmin(wmin_model_settings):
    """
    Tests that the time per evaluation of the likelihood is below a certain threshold
    for DIS data with wmin model.
    """
    # get chi2 with no positivity
    loss_function = colibriAPI.make_chi2(**{**TEST_FULL_DIS_DATASET, **T0_PDFSET})

    # get forward map
    forward_map = colibriAPI.make_pred_data(**TEST_FULL_DIS_DATASET)

    # get FIT_XGRID
    FIT_XGRID = colibriAPI.FIT_XGRID(**TEST_FULL_DIS_DATASET)

    # get pdf_model
    pdf_model = wminAPI.pdf_model(
        **{**wmin_model_settings, "output_path": None, "dump_model": False}
    )

    # get bayesian prior
    prior = bayesian_prior(**TEST_PRIOR_SETTINGS_WMIN)

    pred_and_pdf = pdf_model.pred_and_pdf_func(FIT_XGRID, forward_map=forward_map)

    @jax.jit
    def log_likelihood(params):
        predictions, _ = pred_and_pdf(params)
        return -0.5 * loss_function(predictions)

    # Sample params from the prior and compute the log likelihood
    rng = jax.random.PRNGKey(0)
    start_time = time.perf_counter()
    for _ in range(N_LOOP_ITERATIONS):
        rng, key = jax.random.split(rng)
        hypercube = jax.random.uniform(
            key,
            shape=(wmin_model_settings["wmin_settings"]["n_basis"],),
        )
        params = prior(hypercube)

        log_likelihood(params)
    end_time = time.perf_counter()
    time_per_eval = (end_time - start_time) / N_LOOP_ITERATIONS
    print(f"Time per evaluation: {time_per_eval}")

    assert time_per_eval < THRESHOLD_TIME_DIS


@pytest.mark.parametrize(
    "wmin_model_settings", [TEST_WMIN_SETTINGS_NBASIS_10, TEST_WMIN_SETTINGS_NBASIS_100]
)
def test_likelihood_dis_wmin_with_pos(wmin_model_settings):
    """
    Tests that the time per evaluation of the likelihood is below a certain threshold
    for DIS data with wmin model and positivity.
    """
    # get chi2 with no positivity
    loss_function = colibriAPI.make_chi2_with_positivity(
        **{**TEST_FULL_DIS_DATASET, **TEST_FULL_POS_DATASET, **T0_PDFSET}
    )

    # get forward map
    forward_map = colibriAPI.make_pred_data(**TEST_FULL_DIS_DATASET)

    # get FIT_XGRID
    FIT_XGRID = colibriAPI.FIT_XGRID(
        **{**TEST_FULL_DIS_DATASET, **TEST_FULL_POS_DATASET}
    )

    # get pdf_model
    pdf_model = wminAPI.pdf_model(
        **{**wmin_model_settings, "output_path": None, "dump_model": False}
    )

    # get bayesian prior
    prior = bayesian_prior(**TEST_PRIOR_SETTINGS_WMIN)

    pred_and_pdf = pdf_model.pred_and_pdf_func(FIT_XGRID, forward_map=forward_map)

    @jax.jit
    def log_likelihood(params):
        predictions, pdf = pred_and_pdf(params)
        return -0.5 * loss_function(predictions, pdf)

    # Sample params from the prior and compute the log likelihood
    rng = jax.random.PRNGKey(0)
    start_time = time.perf_counter()
    for _ in range(N_LOOP_ITERATIONS):
        rng, key = jax.random.split(rng)
        hypercube = jax.random.uniform(
            key,
            shape=(wmin_model_settings["wmin_settings"]["n_basis"],),
        )
        params = prior(hypercube)

        log_likelihood(params)
    end_time = time.perf_counter()
    time_per_eval = (end_time - start_time) / N_LOOP_ITERATIONS
    print(f"Time per evaluation: {time_per_eval}")

    assert time_per_eval < THRESHOLD_TIME_DIS_POS


@pytest.mark.parametrize(
    "wmin_model_settings", [TEST_WMIN_SETTINGS_NBASIS_10, TEST_WMIN_SETTINGS_NBASIS_100]
)
def test_likelihood_had_wmin(wmin_model_settings):
    """
    Tests that the time per evaluation of the likelihood is below a certain threshold
    for HAD data with wmin model.
    """
    # get chi2 with no positivity
    loss_function = colibriAPI.make_chi2(**{**TEST_FULL_HAD_DATASET, **T0_PDFSET})

    # get forward map
    forward_map = colibriAPI.make_pred_data(**TEST_FULL_HAD_DATASET)

    # get FIT_XGRID
    FIT_XGRID = colibriAPI.FIT_XGRID(**TEST_FULL_HAD_DATASET)

    # get pdf_model
    pdf_model = wminAPI.pdf_model(
        **{**wmin_model_settings, "output_path": None, "dump_model": False}
    )

    # get bayesian prior
    prior = bayesian_prior(**TEST_PRIOR_SETTINGS_WMIN)

    pred_and_pdf = pdf_model.pred_and_pdf_func(FIT_XGRID, forward_map=forward_map)

    @jax.jit
    def log_likelihood(params):
        predictions, _ = pred_and_pdf(params)
        return -0.5 * loss_function(predictions)

    # Sample params from the prior and compute the log likelihood
    rng = jax.random.PRNGKey(0)
    start_time = time.perf_counter()
    for _ in range(N_LOOP_ITERATIONS):
        rng, key = jax.random.split(rng)
        hypercube = jax.random.uniform(
            key,
            shape=(wmin_model_settings["wmin_settings"]["n_basis"],),
        )
        params = prior(hypercube)

        log_likelihood(params)
    end_time = time.perf_counter()
    time_per_eval = (end_time - start_time) / N_LOOP_ITERATIONS
    print(f"Time per evaluation: {time_per_eval}")

    assert time_per_eval < THRESHOLD_TIME_HAD


@pytest.mark.parametrize(
    "wmin_model_settings", [TEST_WMIN_SETTINGS_NBASIS_10, TEST_WMIN_SETTINGS_NBASIS_100]
)
def test_likelihood_had_wmin_with_pos(wmin_model_settings):
    """
    Tests that the time per evaluation of the likelihood is below a certain threshold
    for HAD data with wmin model and positivity.
    """
    # get chi2 with no positivity
    loss_function = colibriAPI.make_chi2_with_positivity(
        **{**TEST_FULL_HAD_DATASET, **TEST_FULL_POS_DATASET, **T0_PDFSET}
    )

    # get forward map
    forward_map = colibriAPI.make_pred_data(
        **{**TEST_FULL_HAD_DATASET, **TEST_FULL_POS_DATASET}
    )

    # get FIT_XGRID
    FIT_XGRID = colibriAPI.FIT_XGRID(
        **{**TEST_FULL_HAD_DATASET, **TEST_FULL_POS_DATASET}
    )

    # get pdf_model
    pdf_model = wminAPI.pdf_model(
        **{**wmin_model_settings, "output_path": None, "dump_model": False}
    )

    # get bayesian prior
    prior = bayesian_prior(**TEST_PRIOR_SETTINGS_WMIN)

    pred_and_pdf = pdf_model.pred_and_pdf_func(FIT_XGRID, forward_map=forward_map)

    @jax.jit
    def log_likelihood(params):
        predictions, pdf = pred_and_pdf(params)
        return -0.5 * loss_function(predictions, pdf)

    # Sample params from the prior and compute the log likelihood
    rng = jax.random.PRNGKey(0)
    start_time = time.perf_counter()
    for _ in range(N_LOOP_ITERATIONS):
        rng, key = jax.random.split(rng)
        hypercube = jax.random.uniform(
            key,
            shape=(wmin_model_settings["wmin_settings"]["n_basis"],),
        )
        params = prior(hypercube)

        log_likelihood(params)
    end_time = time.perf_counter()
    time_per_eval = (end_time - start_time) / N_LOOP_ITERATIONS
    print(f"Time per evaluation: {time_per_eval}")

    assert time_per_eval < THRESHOLD_TIME_HAD_POS


@pytest.mark.parametrize(
    "wmin_model_settings", [TEST_WMIN_SETTINGS_NBASIS_10, TEST_WMIN_SETTINGS_NBASIS_100]
)
def test_likelihood_global_wmin(wmin_model_settings):
    """
    Tests that the time per evaluation of the likelihood is below a certain threshold
    for global data with wmin model.
    """
    # get chi2 with no positivity
    loss_function = colibriAPI.make_chi2(**{**TEST_FULL_GLOBAL_DATASET, **T0_PDFSET})

    # get forward map
    forward_map = colibriAPI.make_pred_data(**TEST_FULL_GLOBAL_DATASET)

    # get FIT_XGRID
    FIT_XGRID = colibriAPI.FIT_XGRID(**TEST_FULL_GLOBAL_DATASET)

    # get pdf_model
    pdf_model = wminAPI.pdf_model(
        **{**wmin_model_settings, "output_path": None, "dump_model": False}
    )

    # get bayesian prior
    prior = bayesian_prior(**TEST_PRIOR_SETTINGS_WMIN)

    pred_and_pdf = pdf_model.pred_and_pdf_func(FIT_XGRID, forward_map=forward_map)

    @jax.jit
    def log_likelihood(params):
        predictions, _ = pred_and_pdf(params)
        return -0.5 * loss_function(predictions)

    # Sample params from the prior and compute the log likelihood
    rng = jax.random.PRNGKey(0)
    start_time = time.perf_counter()
    for _ in range(N_LOOP_ITERATIONS):
        rng, key = jax.random.split(rng)
        hypercube = jax.random.uniform(
            key,
            shape=(wmin_model_settings["wmin_settings"]["n_basis"],),
        )
        params = prior(hypercube)

        log_likelihood(params)
    end_time = time.perf_counter()
    time_per_eval = (end_time - start_time) / N_LOOP_ITERATIONS
    print(f"Time per evaluation: {time_per_eval}")

    assert time_per_eval < THRESHOLD_TIME_GLOBAL


@pytest.mark.parametrize(
    "wmin_model_settings", [TEST_WMIN_SETTINGS_NBASIS_10, TEST_WMIN_SETTINGS_NBASIS_100]
)
def test_likelihood_global_wmin_with_pos(wmin_model_settings):
    """
    Tests that the time per evaluation of the likelihood is below a certain threshold
    for global data with wmin model and positivity.
    """
    # get chi2 with no positivity
    loss_function = colibriAPI.make_chi2_with_positivity(
        **{**TEST_FULL_GLOBAL_DATASET, **TEST_FULL_POS_DATASET, **T0_PDFSET}
    )

    # get forward map
    forward_map = colibriAPI.make_pred_data(
        **{**TEST_FULL_GLOBAL_DATASET, **TEST_FULL_POS_DATASET}
    )

    # get FIT_XGRID
    FIT_XGRID = colibriAPI.FIT_XGRID(
        **{**TEST_FULL_GLOBAL_DATASET, **TEST_FULL_POS_DATASET}
    )

    # get pdf_model
    pdf_model = wminAPI.pdf_model(
        **{**wmin_model_settings, "output_path": None, "dump_model": False}
    )

    # get bayesian prior
    prior = bayesian_prior(**TEST_PRIOR_SETTINGS_WMIN)

    pred_and_pdf = pdf_model.pred_and_pdf_func(FIT_XGRID, forward_map=forward_map)

    @jax.jit
    def log_likelihood(params):
        predictions, pdf = pred_and_pdf(params)
        return -0.5 * loss_function(predictions, pdf)

    # Sample params from the prior and compute the log likelihood
    rng = jax.random.PRNGKey(0)
    start_time = time.perf_counter()
    for _ in range(N_LOOP_ITERATIONS):
        rng, key = jax.random.split(rng)
        hypercube = jax.random.uniform(
            key,
            shape=(wmin_model_settings["wmin_settings"]["n_basis"],),
        )
        params = prior(hypercube)

        log_likelihood(params)
    end_time = time.perf_counter()
    time_per_eval = (end_time - start_time) / N_LOOP_ITERATIONS
    print(f"Time per evaluation: {time_per_eval}")

    assert time_per_eval < THRESHOLD_TIME_GLOBAL_POS
