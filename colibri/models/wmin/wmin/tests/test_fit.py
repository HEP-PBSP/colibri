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
def test_bayesian_fits(runcard):
    """
    Test the Bayesian fits using the regression runcards.
    """

    regression_path = "colibri/models/wmin/wmin/tests/regression_runcards"

    sp.run(f"{EXE} {runcard}".split(), cwd=regression_path, check=True)

    expected_results = f"colibri/models/wmin/wmin/tests/regression_results/{runcard.split('.')[0]}/ns_result.csv"
    actual_results = f"{regression_path}/{runcard.split('.')[0]}/ns_result.csv"

    # check that the results are as expected using the diff command
    result = sp.run(
        f"diff {expected_results} {actual_results}".split(),
        stdout=sp.PIPE,
        stderr=sp.PIPE,
    )
    assert result.returncode == 0, result.stderr

    # remove the results file after the test
    sp.run(f"rm -r {runcard.split('.')[0]}".split(), cwd=regression_path, check=True)


@pytest.mark.parametrize(
    "runcard",
    [RUNCARD_TEST_FIT_WMIN_MC_DIS, RUNCARD_TEST_FIT_WMIN_MC_HAD],
)
def test_monte_carlo_fits(runcard):
    """
    Test the Monte Carlo fits using the regression runcards.
    """

    regression_path = "colibri/models/wmin/wmin/tests/regression_runcards"

    # run monte carlo fit once
    command = f"{EXE} {runcard} -rep 1"
    result = sp.run(command.split(), cwd=regression_path, check=True)

    # run postfit
    sp.run(
        f"{POSTFIT_EXE} {runcard.split('.')[0]} -t 1 -c 100",
        shell=True,
        cwd=regression_path,
        check=True,
    )

    expected_results = f"colibri/models/wmin/wmin/tests/regression_results/{runcard.split('.')[0]}/mc_result.csv"
    actual_results = f"{regression_path}/{runcard.split('.')[0]}/mc_result.csv"

    # check that the results are as expected using the diff command
    result = sp.run(
        f"diff {expected_results} {actual_results}".split(),
        stdout=sp.PIPE,
        stderr=sp.PIPE,
    )
    assert result.returncode == 0, result.stderr

    # remove the results file after the test
    sp.run(f"rm -r {runcard.split('.')[0]}".split(), cwd=regression_path, check=True)
