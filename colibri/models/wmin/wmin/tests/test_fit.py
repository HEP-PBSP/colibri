"""
Module containing regression tests for wmin.
"""

import subprocess as sp

import pytest
from wmin.tests.wmin_conftest import (
    EXE,
    POSTFIT_EXE,
    RUNCARD_TEST_FIT_WMIN_BAYES_DIS,
    RUNCARD_TEST_FIT_WMIN_BAYES_HAD,
    RUNCARD_TEST_FIT_WMIN_MC_DIS,
    RUNCARD_TEST_FIT_WMIN_MC_HAD,
)


@pytest.mark.parametrize(
    "runcard", [RUNCARD_TEST_FIT_WMIN_BAYES_DIS, RUNCARD_TEST_FIT_WMIN_BAYES_HAD]
)
def test_bayesian_fits(runcard, tmp_path):
    """
    Test the Bayesian fits using the regression runcards.
    """
    regression_path = "colibri/models/wmin/wmin/tests/regression_runcards"

    tmp_fit_folder = tmp_path / "tmp_bayes_fit"

    # run bayesian fit at tmp_path
    sp.run(
        f"{EXE} {runcard} -o {tmp_fit_folder}".split(), cwd=regression_path, check=True
    )

    expected_results = f"colibri/models/wmin/wmin/tests/regression_results/{runcard.split('.')[0]}/ns_result.csv"
    actual_results = f"{tmp_fit_folder}/ns_result.csv"

    # check that the results are as expected using the diff command
    result = sp.run(
        f"diff {expected_results} {actual_results}".split(),
        stdout=sp.PIPE,
        stderr=sp.PIPE,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "runcard",
    [RUNCARD_TEST_FIT_WMIN_MC_DIS, RUNCARD_TEST_FIT_WMIN_MC_HAD],
)
def test_monte_carlo_fits(runcard, tmp_path):
    """
    Test the Monte Carlo fits using the regression runcards.
    """
    regression_path = "colibri/models/wmin/wmin/tests/regression_runcards"
    TMP_NAME = "tmp_mc_fit"
    tmp_fit_folder = tmp_path / TMP_NAME

    # run monte carlo fit once
    command = f"{EXE} {runcard} -rep 1 -o {tmp_fit_folder}"
    result = sp.run(command.split(), cwd=regression_path, check=True)

    # run postfit
    sp.run(
        f"{POSTFIT_EXE} {TMP_NAME} -t 1 -c 100",
        shell=True,
        cwd=tmp_path,
        check=True,
    )

    expected_results = f"colibri/models/wmin/wmin/tests/regression_results/{runcard.split('.')[0]}/mc_result.csv"
    actual_results = f"{tmp_fit_folder}/mc_result.csv"

    # check that the results are as expected using the diff command
    result = sp.run(
        f"diff {expected_results} {actual_results}".split(),
        stdout=sp.PIPE,
        stderr=sp.PIPE,
    )
    assert result.returncode == 0, result.stderr
