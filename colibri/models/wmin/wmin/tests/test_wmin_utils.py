import pandas as pd
import pytest
from wmin.utils import likelihood_time

N_MOCK_DATA = 100
MOCK_NAME_THEORY = "test_theory"


# Mocking the necessary objects and methods
class MockData:
    class MockDataset:
        @staticmethod
        def load_commondata():
            class MockCommonData:
                ndata = N_MOCK_DATA

            return MockCommonData()

    datasets = [MockDataset()]


class MockPDFModel:
    n_basis = 5

    @staticmethod
    def pred_and_pdf_func(FIT_XGRID, forward_map):
        def func(params):
            predictions = params * 2
            pdf = params * 3
            return predictions, pdf

        return func


def mock_chi2_with_positivity(predictions, pdf):
    return sum((predictions - pdf) ** 2)


def mock_pred_data(x):
    return x * 2


def mock_bayesian_prior(rng):
    return rng * 0.5


@pytest.fixture
def setup():
    return {
        "_chi2_with_positivity": mock_chi2_with_positivity,
        "_pred_data": mock_pred_data,
        "FIT_XGRID": [1, 2, 3],
        "pdf_model": MockPDFModel(),
        "bayesian_prior": mock_bayesian_prior,
        "data": MockData(),
        "theoryid": MOCK_NAME_THEORY,
        "n_prior_samples": 10,
    }


def test_likelihood_time_structure(setup):
    """
    test the structure of the output of likelihood_time
    """

    result = likelihood_time(**setup)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["Ndata", "Theory", "Likelihood eval time (s)"]
    assert result.index[0] == "wmin"


def test_likelihood_time_values(setup):
    """
    test the values of the output of likelihood_time
    """

    result = likelihood_time(**setup)
    assert result.loc["wmin", "Ndata"] == N_MOCK_DATA
    assert result.loc["wmin", "Theory"] == MOCK_NAME_THEORY
    assert isinstance(result.loc["wmin", "Likelihood eval time (s)"], float)


def test_likelihood_time_samples():
    """
    test the number of samples in the output of likelihood_time
    """
    setupp = setup()
    setupp["n_prior_samples"] = 5
    result = likelihood_time(**setupp)

    assert len(result) == 1
    assert result.loc["wmin", "Ndata"] == 100
    assert result.loc["wmin", "Theory"] == "test_theory"
    assert isinstance(result.loc["wmin", "Likelihood eval time (s)"], float)
    assert result.loc["wmin", "Likelihood eval time (s)"] > 0
