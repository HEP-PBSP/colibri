import jax
import jax.numpy as jnp
import jax.scipy.stats
from jax import random
from colibri.bayes_prior import bayesian_prior
from colibri.core import PriorSettings
import numpy as np
import pytest
from unittest.mock import patch
import pandas as pd


@pytest.mark.parametrize(
    "description, prior_distribution_specs, cube_shape, expected_fn",
    [
        (
            "global bounds",
            {"min_val": -1.0, "max_val": 1.0},
            (3,),
            lambda cube, specs: cube * (specs["max_val"] - specs["min_val"])
            + specs["min_val"],
        ),
        (
            "per-parameter bounds",
            {"bounds": [[-1.0, 1.0], [0.0, 2.0], [10.0, 20.0]]},  # Shape: (3, 2)
            (3,),
            lambda cube, specs: cube
            * (np.array(specs["bounds"])[:, 1] - np.array(specs["bounds"])[:, 0])
            + np.array(specs["bounds"])[:, 0],
        ),
    ],
)
def test_uniform_prior(description, prior_distribution_specs, cube_shape, expected_fn):
    prior_settings = PriorSettings(
        **{
            "prior_distribution": "uniform_parameter_prior",
            "prior_distribution_specs": prior_distribution_specs,
        }
    )
    prior_transform = bayesian_prior(prior_settings)

    key = random.PRNGKey(0)
    cube = random.uniform(key, shape=cube_shape)

    transformed = prior_transform(cube)
    expected = expected_fn(cube, prior_distribution_specs)

    assert np.allclose(
        transformed, expected
    ), f"Uniform prior transformation failed for {description}"


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

    prior_settings = PriorSettings(
        **{
            "prior_distribution": "prior_from_gauss_posterior",
            "prior_distribution_specs": {"prior_fit": "mock_prior_fit"},
        }
    )

    prior_transform = bayesian_prior(prior_settings)

    key = random.PRNGKey(0)
    cube = random.uniform(key, shape=(10, 2))

    transformed = prior_transform(cube)
    independent_gaussian = jax.scipy.stats.norm.ppf(cube)
    expected = mean + jnp.dot(independent_gaussian, jnp.linalg.cholesky(cov).T)

    assert np.allclose(transformed, expected), "Gaussian prior transformation failed."


def test_invalid_prior_type():
    prior_settings = PriorSettings(
        **{"prior_distribution": "invalid_type", "prior_distribution_specs": {}}
    )

    with pytest.raises(ValueError) as e:
        bayesian_prior(prior_settings)
