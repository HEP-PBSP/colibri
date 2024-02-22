import pandas as pd
import pathlib
from numpy.testing import assert_allclose

from conftest import (
    CLOSURE_TEST_PDFSET,
    PSEUDODATA_SEED,
    TRVAL_INDEX,
    REPLICA_INDEX,
    TEST_DATASETS,
)
from colibri.api import API as colibriAPI

MC_PSEUDODATA = {
    "level_1_seed": PSEUDODATA_SEED,
    **CLOSURE_TEST_PDFSET,
    **TRVAL_INDEX,
    **REPLICA_INDEX,
    **TEST_DATASETS,
}

TEST_COMMONDATA_FOLDER = pathlib.Path(__file__).with_name("test_commondata")


def test_mc_pseudodata():
    """
    Regression test, testing that currently generated pseudodata is consistent
    with the reference one.
    """
    reference_pseudodata = pd.read_csv(
        TEST_COMMONDATA_FOLDER / "NMC_level2_central_values.csv"
    )

    current_pseudodata = colibriAPI.mc_pseudodata(**MC_PSEUDODATA)

    assert_allclose(reference_pseudodata["cv"].values, current_pseudodata.pseudodata)
