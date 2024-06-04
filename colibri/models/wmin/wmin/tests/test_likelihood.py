"""
Module for testing that the time per evaluation of the likelihood is below a certain threshold.
"""

import pathlib
import subprocess as sp
import time

import jax
import jax.scipy.linalg as jla
import pytest
from colibri.api import API as colibriAPI
from colibri.loss_functions import chi2
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
    EXE,
    RUNCARD_WMIN_LIKELIHOOD_TYPE,
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
    loss_function = chi2
    # get forward map
    forward_map = colibriAPI.make_pred_data(**TEST_DATASETS)
    # get FIT_XGRID
    FIT_XGRID = colibriAPI.FIT_XGRID(**TEST_DATASETS)
    # get fast_kernel_arrays
    fast_kernel_arrays = colibriAPI.fast_kernel_arrays(**TEST_DATASETS)
    # get centralconvat_index
    central_covmat_index = colibriAPI.central_covmat_index(
        **{**TEST_DATASETS, **T0_PDFSET}
    )
    central_values = central_covmat_index.central_values
    inv_covmat = jla.inv(central_covmat_index.covmat)
    # get pdf_model
    pdf_model = wminAPI.pdf_model(
        **{**wmin_model_settings, "output_path": None, "dump_model": False}
    )
    # get bayesian prior
    prior = bayesian_prior(**TEST_PRIOR_SETTINGS_WMIN)

    pred_and_pdf = pdf_model.pred_and_pdf_func(FIT_XGRID, forward_map=forward_map)

    @jax.jit
    def log_likelihood(params, central_values, inv_covmat, fast_kernel_arrays):
        predictions, _ = pred_and_pdf(params, fast_kernel_arrays)
        return -0.5 * loss_function(central_values, predictions, inv_covmat)

    # Sample params from the prior
    prior_list = prior_samples(prior, wmin_model_settings)

    # compile likelihood
    log_likelihood(prior_list[0], central_values, inv_covmat, fast_kernel_arrays)

    # evaluate likelihood time
    start_time = time.perf_counter()
    for i in range(N_LOOP_ITERATIONS):
        log_likelihood(prior_list[i], central_values, inv_covmat, fast_kernel_arrays)
    end_time = time.perf_counter()

    time_per_eval = (end_time - start_time) / N_LOOP_ITERATIONS
    print(f"Likelihood time per evaluation for DIS: {time_per_eval}")

    assert time_per_eval < THRESHOLD_TIME_DIS


@pytest.mark.parametrize(
    "wmin_model_settings", [TEST_WMIN_SETTINGS_NBASIS_10, TEST_WMIN_SETTINGS_NBASIS_100]
)
def test_likelihood_had_wmin(wmin_model_settings):
    """
    Tests that the time per evaluation of the likelihood is below a certain threshold
    for HAD data with wmin model.
    """
    # get chi2 with no positivity
    loss_function = chi2

    # get forward map
    forward_map = colibriAPI.make_pred_data(**TEST_DATASETS_HAD)

    # get FIT_XGRID
    FIT_XGRID = colibriAPI.FIT_XGRID(**TEST_DATASETS_HAD)

    # get fast_kernel_arrays
    fast_kernel_arrays = colibriAPI.fast_kernel_arrays(**TEST_DATASETS_HAD)
    # get centralconvat_index
    central_covmat_index = colibriAPI.central_covmat_index(
        **{**TEST_DATASETS_HAD, **T0_PDFSET}
    )
    central_values = central_covmat_index.central_values
    inv_covmat = jla.inv(central_covmat_index.covmat)

    # get pdf_model
    pdf_model = wminAPI.pdf_model(
        **{**wmin_model_settings, "output_path": None, "dump_model": False}
    )

    # get bayesian prior
    prior = bayesian_prior(**TEST_PRIOR_SETTINGS_WMIN)

    pred_and_pdf = pdf_model.pred_and_pdf_func(FIT_XGRID, forward_map=forward_map)

    @jax.jit
    def log_likelihood(params, central_values, inv_covmat, fast_kernel_arrays):
        predictions, _ = pred_and_pdf(params, fast_kernel_arrays)
        return -0.5 * loss_function(central_values, predictions, inv_covmat)

    # Sample params from the prior
    prior_list = prior_samples(prior, wmin_model_settings)

    # compile likelihood
    log_likelihood(prior_list[0], central_values, inv_covmat, fast_kernel_arrays)

    # evaluate likelihood time
    start_time = time.perf_counter()
    for i in range(N_LOOP_ITERATIONS):
        log_likelihood(prior_list[i], central_values, inv_covmat, fast_kernel_arrays)
    end_time = time.perf_counter()

    time_per_eval = (end_time - start_time) / N_LOOP_ITERATIONS
    print(f"Likelihood time per evaluation for HAD: {time_per_eval}")

    assert time_per_eval < THRESHOLD_TIME_HAD


@pytest.mark.parametrize(
    "wmin_model_settings", [TEST_WMIN_SETTINGS_NBASIS_10, TEST_WMIN_SETTINGS_NBASIS_100]
)
def test_likelihood_global_wmin(wmin_model_settings):
    """
    Tests that the time per evaluation of the likelihood is below a certain threshold
    for global data with wmin model.
    """
    # get chi2 with no positivity
    loss_function = chi2

    # get forward map
    forward_map = colibriAPI.make_pred_data(**TEST_DATASETS_DIS_HAD)

    # get FIT_XGRID
    FIT_XGRID = colibriAPI.FIT_XGRID(**TEST_DATASETS_DIS_HAD)

    # get fast_kernel_arrays
    fast_kernel_arrays = colibriAPI.fast_kernel_arrays(**TEST_DATASETS_DIS_HAD)
    # get centralconvat_index
    central_covmat_index = colibriAPI.central_covmat_index(
        **{**TEST_DATASETS_DIS_HAD, **T0_PDFSET}
    )
    central_values = central_covmat_index.central_values
    inv_covmat = jla.inv(central_covmat_index.covmat)

    # get pdf_model
    pdf_model = wminAPI.pdf_model(
        **{**wmin_model_settings, "output_path": None, "dump_model": False}
    )

    # get bayesian prior
    prior = bayesian_prior(**TEST_PRIOR_SETTINGS_WMIN)

    pred_and_pdf = pdf_model.pred_and_pdf_func(FIT_XGRID, forward_map=forward_map)

    @jax.jit
    def log_likelihood(params, central_values, inv_covmat, fast_kernel_arrays):
        predictions, _ = pred_and_pdf(params, fast_kernel_arrays)
        return -0.5 * loss_function(central_values, predictions, inv_covmat)

    # Sample params from the prior
    prior_list = prior_samples(prior, wmin_model_settings)

    # compile likelihood
    log_likelihood(prior_list[0], central_values, inv_covmat, fast_kernel_arrays)

    # evaluate likelihood time
    start_time = time.perf_counter()
    for i in range(N_LOOP_ITERATIONS):
        log_likelihood(prior_list[i], central_values, inv_covmat, fast_kernel_arrays)
    end_time = time.perf_counter()

    time_per_eval = (end_time - start_time) / N_LOOP_ITERATIONS
    print(f"Likelihood time per evaluation for HAD and DIS: {time_per_eval}")

    assert time_per_eval < THRESHOLD_TIME_GLOBAL


@pytest.mark.parametrize("float_type", [64, 32])
def test_likelihood_is_correct_type(float_type):
    """
    Tests that the likelihood is compiled with the correct float type.
    """

    regression_path = pathlib.Path("colibri/models/wmin/wmin/tests/regression_runcards")
    dir_path = (
        pathlib.Path("colibri/models/wmin/wmin/tests/regression_runcards")
        / RUNCARD_WMIN_LIKELIHOOD_TYPE.split(".")[0]
    )

    if float_type == 64:
        sp.run(
            f"{EXE} {RUNCARD_WMIN_LIKELIHOOD_TYPE}".split(),
            cwd=regression_path,
            check=True,
        )
    elif float_type == 32:
        sp.run(
            f"{EXE} {RUNCARD_WMIN_LIKELIHOOD_TYPE} --float32".split(),
            cwd=regression_path,
            check=True,
        )

    # read dtype from file and assert it is the corret one
    with open(dir_path / "dtype.txt", "r") as f:
        dtype = f.read().strip()
        assert dtype == f"float{float_type}"

    # remove directory with results
    sp.run(f"rm -r {dir_path}".split(), check=True)
