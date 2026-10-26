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
from unittest.mock import Mock

# Create a mock forward_map that exposes param_names matching MOCK_PDF_MODEL
MOCK_FORWARD_MAP = Mock()
MOCK_FORWARD_MAP.param_names = MOCK_PDF_MODEL.param_names
MOCK_FORWARD_MAP.pdf_param_names = MOCK_PDF_MODEL.param_names
MOCK_FORWARD_MAP.extra_param_names = []


def test_uniform_prior():
    """
    Test the transformation of a uniform prior distribution.
    """
    prior_transform = bayesian_prior(TEST_PRIOR_SETTINGS_UNIFORM, MOCK_FORWARD_MAP)
    MOCK_PDF_MODEL.n_parameters = 2

    key = random.PRNGKey(0)
    n_params = MOCK_PDF_MODEL.n_parameters
    cube = random.uniform(key, shape=(n_params,))

    # ---- Test sample() ----
    samples = prior_transform.sample(key, 5)

    assert samples.shape == (5, n_params)

    # ---- Test log_prob() ----
    x = jnp.array(samples)
    logp = prior_transform.log_prob(x).sum(axis=-1)

    assert logp.shape == ()
    assert jnp.isfinite(logp).all()

    transformed = prior_transform.prior_transform(cube)
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

    prior_transform_bounds = bayesian_prior(prior_settings_bounds, MOCK_FORWARD_MAP)

    cube_bounds = random.uniform(key, shape=(2,))
    expected_bounds = jnp.array(
        [
            cube_bounds[0] * (1.0 - (-1.0)) + (-1.0),
            cube_bounds[1] * (2.0 - 0.0) + 0.0,
        ]
    )

    transformed_bounds = prior_transform_bounds.prior_transform(cube_bounds)

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
        bayesian_prior(prior_settings_missing_bounds, MOCK_FORWARD_MAP)

    # ---- Test missing min_val/max_val and bounds ----
    prior_settings_invalid = PriorSettings(
        **{
            "prior_distribution": "uniform_parameter_prior",
            "prior_distribution_specs": {},  # neither "bounds" nor min/max
        }
    )

    with pytest.raises(ValueError, match="prior_distribution_specs must define either"):
        bayesian_prior(prior_settings_invalid, MOCK_FORWARD_MAP)


def _posterior_with(mean, cov, columns):
    """DataFrame of samples whose sample mean and covariance are mean and cov."""
    n = len(mean)
    # +-1 combinations along each eigen-direction give zero mean, unit covariance
    base = np.concatenate([np.eye(n), -np.eye(n)]) * np.sqrt(n)
    base = base * np.sqrt((2 * n - 1) / (2 * n))
    samples = mean + base @ np.linalg.cholesky(cov).T
    return pd.DataFrame(samples, columns=list(columns))


@patch("colibri.bayes_prior.get_full_posterior")
def test_gaussian_prior(mock_get_full_posterior):

    # Create a mock posterior dataframe
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])

    # Posterior samples with exactly this mean and covariance
    mock_get_full_posterior.return_value = _posterior_with(
        mean, cov, MOCK_FORWARD_MAP.pdf_param_names
    )

    prior_settings = PriorSettings(
        **{
            "prior_distribution": "prior_from_gauss_posterior",
            "prior_distribution_specs": {"prior_fit": "mock_prior_fit"},
        }
    )

    prior_transform = bayesian_prior(prior_settings, MOCK_FORWARD_MAP)

    key = random.PRNGKey(0)
    cube = random.uniform(key, shape=(10, 2))

    transformed = prior_transform.prior_transform(cube)
    independent_gaussian = jax.scipy.stats.norm.ppf(cube)
    expected = mean + jnp.dot(independent_gaussian, jnp.linalg.cholesky(cov).T)

    assert np.allclose(transformed, expected), "Gaussian prior transformation failed."

    # ---- Cover sample() ----
    with pytest.raises(NotImplementedError, match="sample not implemented"):
        prior_transform.sample(key, 10)

    # ---- Cover log_prob() ----
    with pytest.raises(NotImplementedError, match="log_prob not implemented"):
        prior_transform.log_prob(jnp.zeros((10, 2)))


def test_invalid_prior_type():

    prior_settings = PriorSettings(
        **{"prior_distribution": "invalid_type", "prior_distribution_specs": {}}
    )

    with pytest.raises(ValueError) as e:
        bayesian_prior(prior_settings, MOCK_FORWARD_MAP)


# ---------------------------------------------------------------------------
# Extra parameters of the forward map
# ---------------------------------------------------------------------------

from colibri.bayes_prior import extra_parameters_prior, fit_prior

UNIFORM_PDF_PRIOR = PriorSettings(
    prior_distribution="uniform_parameter_prior",
    prior_distribution_specs={"min_val": -1.0, "max_val": 1.0},
)
EXTRA_SETTINGS = PriorSettings(
    prior_distribution="uniform_parameter_prior",
    prior_distribution_specs={"bounds": {"c1": [10.0, 12.0], "c2": [-4.0, -2.0]}},
)


def _fm_with_extras(pdf_names=("param1", "param2"), extra_names=("c1", "c2")):
    fm = Mock()
    fm.pdf_param_names = list(pdf_names)
    fm.extra_param_names = list(extra_names)
    fm.param_names = list(pdf_names) + list(extra_names)
    fm.n_pdf_params = len(pdf_names)
    return fm


def test_bayesian_prior_covers_pdf_parameters_only():
    prior = bayesian_prior(UNIFORM_PDF_PRIOR, _fm_with_extras())
    assert prior.prior_transform(jnp.full(2, 0.5)).shape == (2,)


def test_bayesian_prior_rejects_extra_bounds_in_prior_settings():
    settings = PriorSettings(
        prior_distribution="uniform_parameter_prior",
        prior_distribution_specs={
            "bounds": {"param1": [0, 1], "param2": [0, 1], "c1": [0, 1]}
        },
    )
    with pytest.raises(ValueError, match="extra_prior_settings"):
        bayesian_prior(settings, _fm_with_extras())


@patch("colibri.bayes_prior.get_full_posterior")
def test_gaussian_prior_uses_pdf_columns_only(mock_get_full_posterior):
    """A prior fit with extra parameters only gives the prior of the PDF ones."""
    mean = np.array([1.0, -1.0, 5.0])
    cov = np.diag([1.0, 2.0, 3.0])
    mock_get_full_posterior.return_value = _posterior_with(
        mean, cov, ["param1", "param2", "c1"]
    )
    settings = PriorSettings(
        prior_distribution="prior_from_gauss_posterior",
        prior_distribution_specs={"prior_fit": "fit"},
    )
    prior = bayesian_prior(settings, _fm_with_extras(extra_names=("c1",)))
    np.testing.assert_allclose(prior.prior_transform(jnp.full(2, 0.5)), mean[:2])


def test_extra_parameters_prior_none_without_extras():
    assert extra_parameters_prior(_fm_with_extras(extra_names=()), None) is None


def test_extra_parameters_prior_requires_settings():
    with pytest.raises(ValueError, match=r"\['c1', 'c2'\]"):
        extra_parameters_prior(_fm_with_extras(), None)


def test_extra_parameters_prior_global_bounds():
    prior = extra_parameters_prior(
        _fm_with_extras(),
        PriorSettings(
            prior_distribution="uniform_parameter_prior",
            prior_distribution_specs={"min_val": -5, "max_val": 5},
        ),
    )
    np.testing.assert_allclose(prior.prior_transform(jnp.array([0.0, 1.0])), [-5, 5])


def test_fit_prior_without_extras_is_pdf_prior():
    fm = _fm_with_extras(extra_names=())
    pdf_prior = bayesian_prior(UNIFORM_PDF_PRIOR, fm)
    assert fit_prior(fm, pdf_prior, None) is pdf_prior


def test_fit_prior_joins_pdf_and_extra_priors():
    fm = _fm_with_extras()
    prior = fit_prior(
        fm,
        bayesian_prior(UNIFORM_PDF_PRIOR, fm),
        extra_parameters_prior(fm, EXTRA_SETTINGS),
    )

    # prior_transform: one point and a batch
    np.testing.assert_allclose(
        prior.prior_transform(jnp.array([0.0, 1.0, 0.5, 0.5])), [-1, 1, 11, -3]
    )
    cubes = random.uniform(random.PRNGKey(0), (7, 4))
    points = prior.prior_transform(cubes)
    assert points.shape == (7, 4)
    assert np.all((points[:, 2] >= 10) & (points[:, 2] <= 12))

    # log_prob: sum of the two independent priors
    x = jnp.array([0.0, 0.0, 11.0, -3.0])
    np.testing.assert_allclose(prior.log_prob(x), np.log(1 / 4) + np.log(1 / 4))
    assert prior.log_prob(jnp.array([0.0, 0.0, 13.0, -3.0])) == -np.inf

    # sample: all parameters, each in its own range
    samples = prior.sample(random.PRNGKey(1), 50)
    assert samples.shape == (50, 4)
    assert np.all(np.abs(samples[:, :2]) <= 1)
    assert np.all((samples[:, 3] >= -4) & (samples[:, 3] <= -2))


def test_fit_prior_rejects_pdf_prior_of_wrong_size():
    """A model prior covering all the parameters is caught with a clear error."""
    fm = _fm_with_extras()
    all_params_prior = bayesian_prior(
        UNIFORM_PDF_PRIOR,
        _fm_with_extras(pdf_names=("param1", "param2", "c1", "c2"), extra_names=()),
    )
    wrong = Mock()
    wrong.prior_transform = lambda cube: all_params_prior.prior_transform(
        jnp.full(4, 0.5)
    )
    with pytest.raises(ValueError, match="PDF parameters only"):
        fit_prior(fm, wrong, extra_parameters_prior(fm, EXTRA_SETTINGS))
