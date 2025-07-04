"""
colibri.tests.test_bayes_prior

Module to test the bayesian_prior function and its associated classes.
"""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import jax.scipy.stats
import numpy as np
import pandas as pd
import pytest
from jax import random

from colibri.bayes_prior import bayesian_prior
from colibri.core import PriorSettings
from colibri.tests.conftest import MOCK_PDF_MODEL, TEST_PRIOR_SETTINGS_UNIFORM


def test_uniform_prior():
    """
    Test the transformation of a uniform prior distribution.
    """
    prior_transform = bayesian_prior(TEST_PRIOR_SETTINGS_UNIFORM, MOCK_PDF_MODEL)

    key = random.PRNGKey(0)
    cube = random.uniform(key, shape=(10,))

    transformed = prior_transform(cube)
    expected = (
        cube
        * (
            TEST_PRIOR_SETTINGS_UNIFORM.prior_distribution_specs["max_val"]
            - TEST_PRIOR_SETTINGS_UNIFORM.prior_distribution_specs["min_val"]
        )
        + TEST_PRIOR_SETTINGS_UNIFORM.prior_distribution_specs["min_val"]
    )

    assert np.allclose(transformed, expected), "Uniform prior transformation failed."

    # ---- Test per-parameter bounds case ----

    bounds = {
        "param1": (-1.0, 1.0),
        "param2": (0.0, 2.0),
    }

    prior_settings_bounds = PriorSettings(
        **{
            "prior_distribution": "uniform_parameter_prior",
            "prior_distribution_specs": {"bounds": bounds},
        }
    )

    prior_transform_bounds = bayesian_prior(prior_settings_bounds, MOCK_PDF_MODEL)

    cube_bounds = random.uniform(key, shape=(2,))
    expected_bounds = jnp.array(
        [
            cube_bounds[0] * (1.0 - (-1.0)) + (-1.0),
            cube_bounds[1] * (2.0 - 0.0) + 0.0,
        ]
    )

    transformed_bounds = prior_transform_bounds(cube_bounds)

    assert jnp.allclose(
        transformed_bounds, expected_bounds
    ), "Uniform prior transformation (per-parameter bounds) failed."

    # ---- Test missing parameter in bounds ----
    incomplete_bounds = {
        "param0": (-1.0, 1.0),
        # "param1" is missing on purpose
    }

    prior_settings_missing_bounds = PriorSettings(
        **{
            "prior_distribution": "uniform_parameter_prior",
            "prior_distribution_specs": {"bounds": incomplete_bounds},
        }
    )

    with pytest.raises(ValueError, match="Missing bounds for parameters"):
        bayesian_prior(prior_settings_missing_bounds, MOCK_PDF_MODEL)

    # ---- Test missing min_val/max_val and bounds ----
    prior_settings_invalid = PriorSettings(
        **{
            "prior_distribution": "uniform_parameter_prior",
            "prior_distribution_specs": {},  # neither "bounds" nor min/max
        }
    )

    with pytest.raises(ValueError, match="prior_distribution_specs must define either"):
        bayesian_prior(prior_settings_invalid, MOCK_PDF_MODEL)


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

    prior_transform = bayesian_prior(prior_settings, MOCK_PDF_MODEL)

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
        bayesian_prior(prior_settings, MOCK_PDF_MODEL)
