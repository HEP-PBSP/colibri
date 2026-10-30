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


def test_uniform_prior():
    """
    Test the transformation of a uniform prior distribution.
    """
    prior_transform = bayesian_prior(TEST_PRIOR_SETTINGS_UNIFORM, MOCK_PDF_MODEL)
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

    prior_transform_bounds = bayesian_prior(prior_settings_bounds, MOCK_PDF_MODEL)

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
        mean, cov, MOCK_PDF_MODEL.param_names
    )

    prior_settings = PriorSettings(
        **{
            "prior_distribution": "prior_from_gauss_posterior",
            "prior_distribution_specs": {"prior_fit": "mock_prior_fit"},
        }
    )

    prior_transform = bayesian_prior(prior_settings, MOCK_PDF_MODEL)

    key = random.PRNGKey(0)
    cube = random.uniform(key, shape=(10, 2))

    transformed = prior_transform.prior_transform(cube)
    independent_gaussian = jax.scipy.stats.norm.ppf(cube)
    expected = mean + jnp.dot(independent_gaussian, jnp.linalg.cholesky(cov).T)

    assert np.allclose(transformed, expected), "Gaussian prior transformation failed."

    # ---- log_prob(): multivariate normal, one point and a batch ----
    points = jnp.array([[0.0, 0.0], [1.0, -0.5]])
    expected_log_prob = jax.scipy.stats.multivariate_normal.logpdf(points, mean, cov)
    np.testing.assert_allclose(prior_transform.log_prob(points), expected_log_prob)
    np.testing.assert_allclose(
        prior_transform.log_prob(points[1]), expected_log_prob[1]
    )

    # ---- sample(): mean and covariance of the Gaussian ----
    samples = prior_transform.sample(key, 20000)
    assert samples.shape == (20000, 2)
    np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.03)
    np.testing.assert_allclose(np.cov(samples.T), cov, atol=0.03)

    # ---- unconstrained parametrisation (NUTS): identity ----
    np.testing.assert_allclose(prior_transform.from_unconstrained(points), points)
    np.testing.assert_allclose(
        prior_transform.log_prob_unconstrained(points), expected_log_prob
    )
    assert prior_transform.sample_unconstrained(key, 5).shape == (5, 2)


def test_invalid_prior_type():

    prior_settings = PriorSettings(
        **{"prior_distribution": "invalid_type", "prior_distribution_specs": {}}
    )

    with pytest.raises(ValueError) as e:
        bayesian_prior(prior_settings, MOCK_PDF_MODEL)


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


def _pdf_model(names=("param1", "param2")):
    pdf_model = Mock()
    pdf_model.param_names = list(names)
    return pdf_model


def test_bayesian_prior_covers_pdf_parameters_only():
    prior = bayesian_prior(UNIFORM_PDF_PRIOR, _pdf_model())
    assert prior.prior_transform(jnp.full(2, 0.5)).shape == (2,)


def test_bayesian_prior_without_pdf_model_is_empty():
    prior = bayesian_prior(UNIFORM_PDF_PRIOR, None)
    assert prior.prior_transform(jnp.zeros(0)).shape == (0,)


def test_bayesian_prior_warns_about_bounds_of_non_pdf_parameters(caplog):
    settings = PriorSettings(
        prior_distribution="uniform_parameter_prior",
        prior_distribution_specs={
            "bounds": {"param1": [0, 1], "param2": [0, 1], "c1": [0, 1]}
        },
    )
    with caplog.at_level("WARNING", logger="colibri.bayes_prior"):
        prior = bayesian_prior(settings, _pdf_model())
    assert "['c1']" in caplog.text
    assert "extra_prior_settings" in caplog.text
    assert prior.prior_transform(jnp.full(2, 0.5)).shape == (2,)


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
    prior = bayesian_prior(settings, _pdf_model())
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
    pdf_prior = bayesian_prior(UNIFORM_PDF_PRIOR, _pdf_model())
    assert fit_prior(fm, pdf_prior, None) is pdf_prior


def test_fit_prior_joins_pdf_and_extra_priors():
    fm = _fm_with_extras()
    prior = fit_prior(
        fm,
        bayesian_prior(UNIFORM_PDF_PRIOR, _pdf_model()),
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


def test_fit_prior_joins_unconstrained_functions():
    """The joint prior keeps the unconstrained parametrisation used by NUTS."""
    fm = _fm_with_extras()
    pdf_prior = bayesian_prior(UNIFORM_PDF_PRIOR, _pdf_model())
    extra_prior = extra_parameters_prior(fm, EXTRA_SETTINGS)
    prior = fit_prior(fm, pdf_prior, extra_prior)

    # z = 0 maps to the middle of each interval
    np.testing.assert_allclose(prior.from_unconstrained(jnp.zeros(4)), [0, 0, 11, -3])

    position = jnp.array([0.3, -1.2, 2.0, -0.5])
    np.testing.assert_allclose(
        prior.log_prob_unconstrained(position),
        pdf_prior.log_prob_unconstrained(position[:2])
        + extra_prior.log_prob_unconstrained(position[2:]),
    )
    # Jacobian is included: log p(x(z)) + log|dx/dz|
    jacobian = jax.jacfwd(prior.from_unconstrained)(position)
    np.testing.assert_allclose(
        prior.log_prob_unconstrained(position),
        prior.log_prob(prior.from_unconstrained(position))
        + jnp.linalg.slogdet(jacobian)[1],
        rtol=1e-6,
    )

    starts = prior.sample_unconstrained(random.PRNGKey(0), 5)
    assert starts.shape == (5, 4)
    points = jax.vmap(prior.from_unconstrained)(starts)
    assert np.all((points[:, 2] >= 10) & (points[:, 2] <= 12))
    assert np.all((points[:, 3] >= -4) & (points[:, 3] <= -2))


@patch("colibri.bayes_prior.get_full_posterior")
def test_fit_prior_gaussian_pdf_and_uniform_extras(mock_get_full_posterior):
    """A Gaussian PDF prior and uniform extras give a joint prior usable by NUTS."""
    mock_get_full_posterior.return_value = _posterior_with(
        np.array([1.0, -1.0]), np.eye(2), ["param1", "param2"]
    )
    gauss = PriorSettings(
        prior_distribution="prior_from_gauss_posterior",
        prior_distribution_specs={"prior_fit": "fit"},
    )
    fm = _fm_with_extras()
    prior = fit_prior(
        fm,
        bayesian_prior(gauss, _pdf_model()),
        extra_parameters_prior(fm, EXTRA_SETTINGS),
    )
    # PDF part: identity; extra part: sigmoid onto the bounds
    np.testing.assert_allclose(
        prior.from_unconstrained(jnp.array([0.3, -0.2, 0.0, 0.0])),
        [0.3, -0.2, 11, -3],
    )
    position = jnp.array([0.3, -0.2, 1.0, -2.0])
    jacobian = jax.jacfwd(prior.from_unconstrained)(position)
    np.testing.assert_allclose(
        prior.log_prob_unconstrained(position),
        prior.log_prob(prior.from_unconstrained(position))
        + jnp.linalg.slogdet(jacobian)[1],
        rtol=1e-6,
    )
    assert prior.sample_unconstrained(random.PRNGKey(0), 3).shape == (3, 4)


def test_fit_prior_without_unconstrained_part():
    """If one part has no unconstrained parametrisation, neither has the joint prior."""
    from colibri.core import BayesianPrior

    custom = BayesianPrior(
        prior_transform=lambda cube: cube,
        log_prob=lambda x: jnp.zeros(x.shape[:-1]),
        sample=lambda key, n: random.uniform(key, (n, 2)),
    )
    fm = _fm_with_extras()
    prior = fit_prior(fm, bayesian_prior(UNIFORM_PDF_PRIOR, _pdf_model()), custom)
    assert prior.from_unconstrained is None
    assert prior.log_prob_unconstrained is None
    assert prior.sample_unconstrained is None


@pytest.mark.parametrize("missing", ["pdf", "extra", "both"])
def test_fit_prior_logs_part_without_unconstrained(caplog, missing):
    """fit_prior says which part of the prior cannot be used with NUTS."""
    from colibri.core import BayesianPrior

    custom = BayesianPrior(
        prior_transform=lambda cube: cube,
        log_prob=lambda x: jnp.zeros(x.shape[:-1]),
        sample=lambda key, n: random.uniform(key, (n, 2)),
    )
    fm = _fm_with_extras()
    uniform_pdf = bayesian_prior(UNIFORM_PDF_PRIOR, _pdf_model())
    uniform_extra = extra_parameters_prior(fm, EXTRA_SETTINGS)

    with caplog.at_level("INFO", logger="colibri.bayes_prior"):
        fit_prior(
            fm,
            uniform_pdf if missing == "extra" else custom,
            uniform_extra if missing == "pdf" else custom,
        )

    assert ("(bayesian_prior)" in caplog.text) == (missing != "extra")
    assert ("(extra_parameters_prior)" in caplog.text) == (missing != "pdf")
    assert "NUTS" in caplog.text


def test_fit_prior_silent_when_nuts_compatible(caplog):
    fm = _fm_with_extras()
    with caplog.at_level("INFO", logger="colibri.bayes_prior"):
        fit_prior(
            fm,
            bayesian_prior(UNIFORM_PDF_PRIOR, _pdf_model()),
            extra_parameters_prior(fm, EXTRA_SETTINGS),
        )
    assert "NUTS" not in caplog.text


@pytest.mark.parametrize(
    "specs",
    [{"bounds": {"a": [0.0, 1.0]}}, {"min_val": -1.0, "max_val": 1.0}],
)
def test_uniform_prior_bounds_empty_parameter_list(specs):
    from colibri.bayes_prior import uniform_prior_bounds

    mins, maxs = uniform_prior_bounds([], specs)
    assert mins.shape == (0,)
    assert maxs.shape == (0,)
