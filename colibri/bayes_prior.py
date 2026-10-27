"""
colibri.bayes_prior.py

Priors of the Bayesian fits.

The fit parameters are the PDF parameters followed by the extra parameters of
the forward map (see ``forward_map.ForwardMap``). The two parts have
independent priors, each given by its own provider:

- ``bayesian_prior``: prior of the PDF parameters of ``pdf_model``. By
  default, built from ``prior_settings``.
- ``extra_parameters_prior``: prior of the extra parameters of the forward
  map. By default, uniform with bounds from ``extra_prior_settings``.

Apps can override either provider with any ``BayesianPrior``, as long as it
covers only its own part of the parameters. ``fit_prior`` joins them, and is
what the samplers use.
"""

import logging

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from colibri.core import BayesianPrior
from colibri.utils import cast_to_numpy, get_full_posterior

log = logging.getLogger(__name__)

tfd = tfp.distributions


def uniform_prior_bounds(param_names, specs, settings_name="prior_distribution_specs"):
    """
    Lower and upper bounds of a uniform prior on ``param_names``.

    ``specs`` either has per-parameter ``bounds: {name: [min, max]}`` or global
    ``min_val`` and ``max_val``.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Lower and upper bounds, in the order of ``param_names``.
    """
    param_names = list(param_names)

    if "bounds" in specs:
        bounds_dict = specs["bounds"]
        missing = [p for p in param_names if p not in bounds_dict]
        if missing:
            raise ValueError(f"Missing bounds for parameters: {missing}")

        bounds = jnp.array([bounds_dict[param] for param in param_names], dtype=float)
        return bounds[:, 0], bounds[:, 1]

    if "min_val" in specs and "max_val" in specs:
        n_params = len(param_names)
        mins = jnp.array([float(specs["min_val"])] * n_params)
        maxs = jnp.array([float(specs["max_val"])] * n_params)
        return mins, maxs

    raise ValueError(
        f"{settings_name} must define either 'bounds' or 'min_val' and 'max_val'"
    )


def uniform_prior(mins, maxs):
    """:class:`BayesianPrior` uniform between ``mins`` and ``maxs``."""
    prior = tfd.Uniform(low=mins, high=maxs)

    def sample(rng_key, n_samples):
        return prior.sample(seed=rng_key, sample_shape=(n_samples,))

    @jax.jit
    def log_prob(x):
        return prior.log_prob(x).sum(axis=-1)

    @jax.jit
    def prior_transform(cube):
        return cube * (maxs - mins) + mins

    return BayesianPrior(
        prior_transform=prior_transform,
        log_prob=log_prob,
        sample=sample,
    )


def bayesian_prior(prior_settings, pdf_model):
    """
    Prior of the PDF parameters (``pdf_model.param_names``).

    Apps can override this provider; the extra parameters of the forward map
    have their own prior (see ``extra_parameters_prior``).

    Parameters
    ----------
    prior_settings: PriorSettings
        The settings for the prior.

    pdf_model: pdf_model.PDFModel or None
        The PDF model, used to determine parameter names and ordering. None
        means there are no PDF parameters.

    Returns
    -------
    BayesianPrior
    """
    pdf_param_names = list(pdf_model.param_names) if pdf_model is not None else []

    if prior_settings.prior_distribution == "uniform_parameter_prior":
        prior_specs = prior_settings.prior_distribution_specs
        unused = sorted(set(prior_specs.get("bounds", {})) - set(pdf_param_names))
        if unused:
            log.warning(
                f"prior_settings has bounds for {unused}, which are not "
                "parameters of the PDF model: they are ignored. The bounds of "
                "the extra parameters of the forward map go in "
                "extra_prior_settings."
            )
        mins, maxs = uniform_prior_bounds(pdf_param_names, prior_specs)
        return uniform_prior(mins, maxs)

    elif prior_settings.prior_distribution == "prior_from_gauss_posterior":
        prior_fit = prior_settings.prior_distribution_specs["prior_fit"]

        # Only the PDF parameters of the prior fit
        df_fit = get_full_posterior(prior_fit)[pdf_param_names]

        # Compute mean and covariance matrix of the posterior
        mean_posterior = jnp.array(df_fit.mean().values)
        cov_posterior = jnp.array(df_fit.cov().values)

        sqrt_cov_posterior = jnp.linalg.cholesky(cov_posterior)

        # Define dummy log_prob and sample for now
        @jax.jit
        def log_prob(x):
            raise NotImplementedError("log_prob not implemented for Gaussian prior")

        def sample(rng_key, n_samples):
            raise NotImplementedError("sample not implemented for Gaussian prior")

        @cast_to_numpy
        @jax.jit
        def prior_transform(cube):
            # generate independent gaussian with mean 0 and std 1
            independent_gaussian = jax.scipy.stats.norm.ppf(cube)
            return mean_posterior + jnp.einsum(
                "ij,...j->...i", sqrt_cov_posterior, independent_gaussian
            )

        return BayesianPrior(
            prior_transform=prior_transform,
            log_prob=log_prob,
            sample=sample,
        )

    raise ValueError("Invalid prior type.")


def extra_parameters_bounds(forward_map, extra_prior_settings=None):
    """
    Lower and upper bounds of the uniform prior of the extra parameters of the
    forward map, from ``extra_prior_settings``.
    """
    extra_param_names = list(forward_map.extra_param_names)
    if extra_prior_settings is None:
        raise ValueError(
            f"The forward map has extra parameters {extra_param_names}: set their "
            "prior with extra_prior_settings (prior_distribution_specs with "
            "'bounds' or 'min_val' and 'max_val')."
        )
    return uniform_prior_bounds(
        extra_param_names,
        extra_prior_settings.prior_distribution_specs,
        settings_name="extra_prior_settings prior_distribution_specs",
    )


def extra_parameters_prior(forward_map, extra_prior_settings=None):
    """
    Prior of the extra parameters of the forward map
    (``forward_map.extra_param_names``), or None if there are none.

    The default is a uniform prior with bounds from ``extra_prior_settings``.
    Apps can override this provider with any prior on the extra parameters;
    the PDF parameters have their own prior (see ``bayesian_prior``).

    Parameters
    ----------
    forward_map: ForwardMap

    extra_prior_settings: PriorSettings, optional
        Parsed ``extra_prior_settings`` (see colibriConfig).

    Returns
    -------
    BayesianPrior or None
    """
    if not forward_map.extra_param_names:
        return None
    return uniform_prior(*extra_parameters_bounds(forward_map, extra_prior_settings))


def fit_prior(forward_map, bayesian_prior, extra_parameters_prior):
    """
    Prior of all the fit parameters, in the order of ``forward_map.param_names``:
    ``bayesian_prior`` (PDF parameters) and ``extra_parameters_prior`` (extra
    parameters of the forward map), taken as independent. Either may be
    overridden by apps.

    Without extra parameters this is ``bayesian_prior`` itself.

    Returns
    -------
    BayesianPrior
    """
    if extra_parameters_prior is None:
        return bayesian_prior

    n_pdf = forward_map.n_pdf_params

    # The PDF prior must cover the PDF parameters only
    test_point = bayesian_prior.prior_transform(jnp.full(n_pdf, 0.5))
    if jnp.shape(test_point) != (n_pdf,):
        raise ValueError(
            f"The PDF prior (bayesian_prior) returns parameters of shape "
            f"{jnp.shape(test_point)}, but the forward map has {n_pdf} PDF "
            "parameters. It must cover the PDF parameters only: the extra "
            "parameters of the forward map have their own prior."
        )

    def prior_transform(cube):
        cube = jnp.asarray(cube)
        return jnp.concatenate(
            [
                jnp.asarray(bayesian_prior.prior_transform(cube[..., :n_pdf])),
                extra_parameters_prior.prior_transform(cube[..., n_pdf:]),
            ],
            axis=-1,
        )

    def log_prob(x):
        return bayesian_prior.log_prob(
            x[..., :n_pdf]
        ) + extra_parameters_prior.log_prob(x[..., n_pdf:])

    def sample(rng_key, n_samples):
        key_pdf, key_extra = jax.random.split(rng_key)
        return jnp.concatenate(
            [
                bayesian_prior.sample(key_pdf, n_samples),
                extra_parameters_prior.sample(key_extra, n_samples),
            ],
            axis=-1,
        )

    return BayesianPrior(
        prior_transform=prior_transform,
        log_prob=log_prob,
        sample=sample,
    )
