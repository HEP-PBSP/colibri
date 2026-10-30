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
    if not param_names:
        return jnp.empty(0), jnp.empty(0)

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

    # Unconstrained parametrisation, used by NUTS: x = mins + (maxs - mins) * sigmoid(z)
    @jax.jit
    def from_unconstrained(position):
        return mins + (maxs - mins) * jax.nn.sigmoid(position)

    @jax.jit
    def log_prob_unconstrained(position):
        # For a uniform prior, its normalization cancels the
        # interval-width contribution to the Jacobian.
        return jnp.sum(
            jax.nn.log_sigmoid(position) + jax.nn.log_sigmoid(-position),
            axis=-1,
        )

    def sample_unconstrained(rng_key, n_samples):
        dtype = jnp.asarray(0.0).dtype
        eps = jnp.finfo(dtype).eps
        cube = jax.random.uniform(
            rng_key,
            shape=(n_samples, mins.shape[0]),
            dtype=dtype,
            minval=eps,
            maxval=1 - eps,
        )
        return jnp.log(cube) - jnp.log1p(-cube)

    return BayesianPrior(
        prior_transform=prior_transform,
        log_prob=log_prob,
        sample=sample,
        from_unconstrained=from_unconstrained,
        log_prob_unconstrained=log_prob_unconstrained,
        sample_unconstrained=sample_unconstrained,
    )


def bayesian_prior(prior_settings=None, pdf_model=None):
    """
    Prior of the PDF parameters (``pdf_model.param_names``).

    Apps can override this provider; the extra parameters of the forward map
    have their own prior (see ``extra_parameters_prior``).

    Parameters
    ----------
    prior_settings: PriorSettings, optional
        The settings for the prior. Only needed if the PDF model has
        parameters: without them (e.g. ``fixed_pdf``) the prior is empty.

    pdf_model: pdf_model.PDFModel or None
        The PDF model, used to determine parameter names and ordering. None
        means there are no PDF parameters.

    Returns
    -------
    BayesianPrior
    """
    pdf_param_names = list(pdf_model.param_names) if pdf_model is not None else []

    if prior_settings is None:
        if pdf_param_names:
            raise ValueError(
                f"The PDF model has parameters {pdf_param_names}: set their "
                "prior with prior_settings."
            )
        return uniform_prior(jnp.empty(0), jnp.empty(0))

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

    if prior_settings.prior_distribution == "prior_from_gauss_posterior":
        prior_fit = prior_settings.prior_distribution_specs["prior_fit"]

        # Only the PDF parameters of the prior fit
        df_fit = get_full_posterior(prior_fit)[pdf_param_names]

        # Compute mean and covariance matrix of the posterior
        mean_posterior = jnp.array(df_fit.mean().values)
        cov_posterior = jnp.array(df_fit.cov().values)

        sqrt_cov_posterior = jnp.linalg.cholesky(cov_posterior)

        @jax.jit
        def log_prob(x):
            return jax.scipy.stats.multivariate_normal.logpdf(
                x, mean_posterior, cov_posterior
            )

        def sample(rng_key, n_samples):
            independent_gaussian = jax.random.normal(
                rng_key, shape=(n_samples, mean_posterior.shape[0])
            )
            return mean_posterior + independent_gaussian @ sqrt_cov_posterior.T

        @cast_to_numpy
        @jax.jit
        def prior_transform(cube):
            # generate independent gaussian with mean 0 and std 1
            independent_gaussian = jax.scipy.stats.norm.ppf(cube)
            return mean_posterior + jnp.einsum(
                "ij,...j->...i", sqrt_cov_posterior, independent_gaussian
            )

        # A Gaussian is already unconstrained: NUTS samples the parameters
        # directly (identity map, no Jacobian).
        def from_unconstrained(position):
            return position

        return BayesianPrior(
            prior_transform=prior_transform,
            log_prob=log_prob,
            sample=sample,
            from_unconstrained=from_unconstrained,
            log_prob_unconstrained=log_prob,
            sample_unconstrained=sample,
        )

    raise ValueError("Invalid prior type.")


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
    if extra_prior_settings is None:
        raise ValueError(
            f"The forward map has extra parameters "
            f"{list(forward_map.extra_param_names)}: set their prior with "
            "extra_prior_settings (prior_distribution_specs with 'bounds' or "
            "'min_val' and 'max_val')."
        )
    mins, maxs = uniform_prior_bounds(
        forward_map.extra_param_names,
        extra_prior_settings.prior_distribution_specs,
        settings_name="extra_prior_settings prior_distribution_specs",
    )
    return uniform_prior(mins, maxs)


UNCONSTRAINED_FUNCTIONS = (
    "from_unconstrained",
    "log_prob_unconstrained",
    "sample_unconstrained",
)


def has_unconstrained(prior):
    """
    Whether ``prior`` provides the unconstrained parametrisation that NUTS
    needs (see ``UNCONSTRAINED_FUNCTIONS``).
    """
    return all(callable(getattr(prior, name, None)) for name in UNCONSTRAINED_FUNCTIONS)


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
    # Only NUTS needs the unconstrained parametrisation: report a missing one
    # here, where it is known which part lacks it.
    parts = {
        "the prior of the PDF parameters (bayesian_prior)": bayesian_prior,
        "the prior of the extra parameters of the forward map "
        "(extra_parameters_prior)": extra_parameters_prior,
    }
    for description, prior in parts.items():
        if prior is not None and not has_unconstrained(prior):
            log.info(
                f"{description.capitalize()} has no unconstrained "
                f"parametrisation ({', '.join(UNCONSTRAINED_FUNCTIONS)}): it "
                "cannot be used with NUTS."
            )

    if extra_parameters_prior is None:
        return bayesian_prior

    n_pdf = forward_map.n_pdf_params

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

    unconstrained = {}
    # The joint unconstrained parametrisation (used by NUTS) exists only if
    # both parts provide one.
    if has_unconstrained(bayesian_prior) and has_unconstrained(extra_parameters_prior):

        def from_unconstrained(position):
            return jnp.concatenate(
                [
                    bayesian_prior.from_unconstrained(position[..., :n_pdf]),
                    extra_parameters_prior.from_unconstrained(position[..., n_pdf:]),
                ],
                axis=-1,
            )

        def log_prob_unconstrained(position):
            return bayesian_prior.log_prob_unconstrained(
                position[..., :n_pdf]
            ) + extra_parameters_prior.log_prob_unconstrained(position[..., n_pdf:])

        def sample_unconstrained(rng_key, n_samples):
            key_pdf, key_extra = jax.random.split(rng_key)
            return jnp.concatenate(
                [
                    bayesian_prior.sample_unconstrained(key_pdf, n_samples),
                    extra_parameters_prior.sample_unconstrained(key_extra, n_samples),
                ],
                axis=-1,
            )

        unconstrained = dict(
            from_unconstrained=from_unconstrained,
            log_prob_unconstrained=log_prob_unconstrained,
            sample_unconstrained=sample_unconstrained,
        )

    return BayesianPrior(
        prior_transform=prior_transform,
        log_prob=log_prob,
        sample=sample,
        **unconstrained,
    )
