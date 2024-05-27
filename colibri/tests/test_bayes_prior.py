import jax
import jax.numpy as jnp
import jax.scipy.stats
from jax import random
from colibri.bayes_prior import bayesian_prior
import numpy as np
import pytest
from unittest.mock import patch
import pandas as pd


def test_uniform_prior():
    prior_settings = {
        "type": "uniform_parameter_prior",
        "min_val": -1.0,
        "max_val": 1.0,
    }
    prior_transform = bayesian_prior(prior_settings)

    key = random.PRNGKey(0)
    cube = random.uniform(key, shape=(10,))

    transformed = prior_transform(cube)
    expected = (
        cube * (prior_settings["max_val"] - prior_settings["min_val"])
        + prior_settings["min_val"]
    )

    assert np.allclose(transformed, expected), "Uniform prior transformation failed."


@patch("colibri.bayes_prior.get_full_posterior")
def test_gaussian_prior(mock_get_full_posterior):
    # Create a mock posterior dataframe
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])

    class MockDataFrame:
        def mean(self):
            return pd.Series(mean)  # Convert mean to a Pandas Series

        def cov(self):
            return pd.DataFrame(cov)  # Convert cov to a Pandas DataFrame

    mock_get_full_posterior.return_value = MockDataFrame()

    prior_settings = {
        "type": "prior_from_gauss_posterior",
        "prior_fit": "mock_prior_fit",
    }

    prior_transform = bayesian_prior(prior_settings)

    key = random.PRNGKey(0)
    cube = random.uniform(key, shape=(10, 2))

    transformed = prior_transform(cube)
    independent_gaussian = jax.scipy.stats.norm.ppf(cube)
    expected = mean + jnp.dot(independent_gaussian, jnp.linalg.cholesky(cov).T)

    assert np.allclose(transformed, expected), "Gaussian prior transformation failed."


def test_invalid_prior_type():
    prior_settings = {"type": "invalid_type"}

    with pytest.raises(ValueError) as e:
        bayesian_prior(prior_settings)
