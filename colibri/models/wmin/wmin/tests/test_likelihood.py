"""
Module for testing that the time per evaluation of the likelihood is below a certain threshold.
"""

import time

import jax
import pytest
from colibri.api import API as colibriAPI
from colibri.bayes_prior import bayesian_prior
from colibri.tests.conftest import (
    T0_PDFSET,
    TEST_DATASETS,
    TEST_DATASETS_DIS_HAD,
    TEST_DATASETS_HAD,
    TEST_POS_DATASET,
)
from wmin.api import API as wminAPI
from wmin.tests.wmin_conftest import (
    TEST_PRIOR_SETTINGS_WMIN,
    TEST_WMIN_SETTINGS_NBASIS_10,
    TEST_WMIN_SETTINGS_NBASIS_100,
)

N_LOOP_ITERATIONS = 100

"""
Threshold times are chosen > 1 order of magnitude higher than the expected time since 
the time can vary depending on the machine and github actions runners are usually slower 
than local machines.
"""
THRESHOLD_TIME_DIS = 5e-4
THRESHOLD_TIME_DIS_POS = 7e-4
THRESHOLD_TIME_HAD = 7e-3
THRESHOLD_TIME_HAD_POS = 1e-2
THRESHOLD_TIME_GLOBAL = 5e-2
THRESHOLD_TIME_GLOBAL_POS = 6e-2

RNG_KEY = 0


def prior_samples(prior, wmin_model_settings):
    # Sample params from the prior
    prior_samples = []
    rng = jax.random.PRNGKey(RNG_KEY)
    hypercube_samples = jax.random.uniform(
        rng, shape=(N_LOOP_ITERATIONS, wmin_model_settings["wmin_settings"]["n_basis"])
    )

    for i in range(N_LOOP_ITERATIONS):
        prior_samples.append(prior(hypercube_samples[i]))

    return prior_samples


@pytest.mark.parametrize(
    "wmin_model_settings", [TEST_WMIN_SETTINGS_NBASIS_10, TEST_WMIN_SETTINGS_NBASIS_100]
)
def test_likelihood_dis_wmin(wmin_model_settings):
    """
    Tests that the time per evaluation of the likelihood is below a certain threshold
    for DIS data with wmin model.
    """
    # get chi2 with no positivity
    loss_function = colibriAPI.make_chi2(**{**TEST_DATASETS, **T0_PDFSET})
    # get forward map
    forward_map = colibriAPI.make_pred_data(**TEST_DATASETS)
    # get FIT_XGRID
    FIT_XGRID = colibriAPI.FIT_XGRID(**TEST_DATASETS)
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

    # Sample params from the prior
    prior_list = prior_samples(prior, wmin_model_settings)

    # compile likelihood
    log_likelihood(prior_list[0])

    # evaluate likelihood time
    start_time = time.perf_counter()
    for i in range(N_LOOP_ITERATIONS):
        log_likelihood(prior_list[i])
    end_time = time.perf_counter()

    time_per_eval = (end_time - start_time) / N_LOOP_ITERATIONS
    print(f"Likelihood time per evaluation for DIS: {time_per_eval}")

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
        **{**TEST_DATASETS, **TEST_POS_DATASET, **T0_PDFSET}
    )

    # get forward map
    forward_map = colibriAPI.make_pred_data(**{**TEST_DATASETS, **TEST_POS_DATASET})

    # get FIT_XGRID
    FIT_XGRID = colibriAPI.FIT_XGRID(**{**TEST_DATASETS, **TEST_POS_DATASET})

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

    # Sample params from the prior
    prior_list = prior_samples(prior, wmin_model_settings)

    # compile likelihood
    log_likelihood(prior_list[0])

    # evaluate likelihood time
    start_time = time.perf_counter()
    for i in range(N_LOOP_ITERATIONS):
        log_likelihood(prior_list[i])
    end_time = time.perf_counter()

    time_per_eval = (end_time - start_time) / N_LOOP_ITERATIONS
    print(f"Likelihood time per evaluation for DIS w/ POS: {time_per_eval}")

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
    loss_function = colibriAPI.make_chi2(**{**TEST_DATASETS_HAD, **T0_PDFSET})

    # get forward map
    forward_map = colibriAPI.make_pred_data(**TEST_DATASETS_HAD)

    # get FIT_XGRID
    FIT_XGRID = colibriAPI.FIT_XGRID(**TEST_DATASETS_HAD)

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

    # Sample params from the prior
    prior_list = prior_samples(prior, wmin_model_settings)

    # compile likelihood
    log_likelihood(prior_list[0])

    # evaluate likelihood time
    start_time = time.perf_counter()
    for i in range(N_LOOP_ITERATIONS):
        log_likelihood(prior_list[i])
    end_time = time.perf_counter()

    time_per_eval = (end_time - start_time) / N_LOOP_ITERATIONS
    print(f"Likelihood time per evaluation for HAD: {time_per_eval}")

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
        **{**TEST_DATASETS_HAD, **TEST_POS_DATASET, **T0_PDFSET}
    )

    # get forward map
    forward_map = colibriAPI.make_pred_data(**{**TEST_DATASETS_HAD, **TEST_POS_DATASET})

    # get FIT_XGRID
    FIT_XGRID = colibriAPI.FIT_XGRID(**{**TEST_DATASETS_HAD, **TEST_POS_DATASET})

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

    # Sample params from the prior
    prior_list = prior_samples(prior, wmin_model_settings)

    # compile likelihood
    log_likelihood(prior_list[0])

    # evaluate likelihood time
    start_time = time.perf_counter()
    for i in range(N_LOOP_ITERATIONS):
        log_likelihood(prior_list[i])
    end_time = time.perf_counter()

    time_per_eval = (end_time - start_time) / N_LOOP_ITERATIONS
    print(f"Likelihood time per evaluation for HAD w/ POS: {time_per_eval}")

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
    loss_function = colibriAPI.make_chi2(**{**TEST_DATASETS_DIS_HAD, **T0_PDFSET})

    # get forward map
    forward_map = colibriAPI.make_pred_data(**TEST_DATASETS_DIS_HAD)

    # get FIT_XGRID
    FIT_XGRID = colibriAPI.FIT_XGRID(**TEST_DATASETS_DIS_HAD)

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

    # Sample params from the prior
    prior_list = prior_samples(prior, wmin_model_settings)

    # compile likelihood
    log_likelihood(prior_list[0])

    # evaluate likelihood time
    start_time = time.perf_counter()
    for i in range(N_LOOP_ITERATIONS):
        log_likelihood(prior_list[i])
    end_time = time.perf_counter()

    time_per_eval = (end_time - start_time) / N_LOOP_ITERATIONS
    print(f"Likelihood time per evaluation for HAD and DIS: {time_per_eval}")

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
        **{**TEST_DATASETS_DIS_HAD, **TEST_POS_DATASET, **T0_PDFSET}
    )

    # get forward map
    forward_map = colibriAPI.make_pred_data(
        **{**TEST_DATASETS_DIS_HAD, **TEST_POS_DATASET}
    )

    # get FIT_XGRID
    FIT_XGRID = colibriAPI.FIT_XGRID(**{**TEST_DATASETS_DIS_HAD, **TEST_POS_DATASET})

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

    # Sample params from the prior
    prior_list = prior_samples(prior, wmin_model_settings)

    # compile likelihood
    log_likelihood(prior_list[0])

    # evaluate likelihood time
    start_time = time.perf_counter()
    for i in range(N_LOOP_ITERATIONS):
        log_likelihood(prior_list[i])
    end_time = time.perf_counter()

    time_per_eval = (end_time - start_time) / N_LOOP_ITERATIONS
    print(f"Likelihood time per evaluation for HAD and DIS w/ POS: {time_per_eval}")

    assert time_per_eval < THRESHOLD_TIME_GLOBAL_POS
