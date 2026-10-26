"""
colibri.param_initialisation.py

Initial values of the fit parameters, for the fits that need a starting point
(Monte Carlo and Hessian fits).

The fit parameters are the PDF parameters followed by the extra parameters of
the forward map (see ``forward_map.ForwardMap``). The two parts are
initialised separately:

- ``pdf_initial_parameters``: the PDF parameters, from
  ``param_initialiser_settings``. PDF model apps can override this provider.
- ``extra_initial_parameters``: the extra parameters, from the ``initialiser``
  of ``extra_parameters_settings`` (zeros by default).

``initial_parameters`` joins them, and is what the fits use.
"""

import logging

import jax
import jax.numpy as jnp

log = logging.getLogger(__name__)

INITIALISER_TYPES = ("zeros", "normal", "uniform")


def _initial_values(param_names, param_initialiser_settings, key_fn):
    """
    Initial values for ``param_names`` according to ``param_initialiser_settings``.

    ``key_fn`` returns the random key, given the ``random_seed`` of the settings
    (None if not set). It is only called for random initialisers.
    """
    param_names = list(param_names)

    if param_initialiser_settings["type"] not in INITIALISER_TYPES:
        log.warning(
            f"MC initialiser type {param_initialiser_settings['type']} not recognised, using default: 'zeros' instead."
        )

        param_initialiser_settings["type"] = "zeros"

    if param_initialiser_settings["type"] == "zeros":
        return jnp.array([0.0] * len(param_names))

    random_seed = key_fn(param_initialiser_settings.get("random_seed"))

    if param_initialiser_settings["type"] == "normal":
        means_setting = param_initialiser_settings.get("means", 0.0)
        stds_setting = param_initialiser_settings.get("stds", 1.0)

        if (
            "means" not in param_initialiser_settings
            and "stds" not in param_initialiser_settings
        ):
            log.warning(
                "param_initialiser_settings: No 'means' or 'stds' provided. "
                "Using default normal distribution N(0, 1) for all parameters."
            )

        if (
            "means" in param_initialiser_settings
            and "stds" not in param_initialiser_settings
        ):
            log.warning(
                "param_initialiser_settings: 'means' provided without 'stds'. "
                "Using default std=1.0 for all parameters."
            )

        if (
            "stds" in param_initialiser_settings
            and "means" not in param_initialiser_settings
        ):
            log.warning(
                "param_initialiser_settings: 'stds' provided without 'means'. "
                "Using default mean=0.0 for all parameters."
            )

        def expand(setting, default, name):
            # If dict → check consistency
            if isinstance(setting, dict):
                if len(setting) != len(param_names):
                    raise ValueError(
                        f"'{name}' dict must have one entry per parameter "
                        f"(got {len(setting)} for {len(param_names)} parameters)."
                    )
                return jnp.array([setting.get(p, default) for p in param_names])
            # If scalar → broadcast
            elif isinstance(setting, (int, float)):
                return jnp.full(len(param_names), setting)
            else:
                raise TypeError(f"'{name}' must be dict or scalar, got {type(setting)}")

        means = expand(means_setting, 0.0, "means")
        stds = expand(stds_setting, 1.0, "stds")

        normal_samples = jax.random.normal(key=random_seed, shape=(len(param_names),))
        return means + stds * normal_samples

    if param_initialiser_settings["type"] == "uniform":
        if "bounds" in param_initialiser_settings:
            # Use param names from the model to order bounds correctly
            bounds_dict = param_initialiser_settings["bounds"]

            missing = [p for p in param_names if p not in bounds_dict]
            if missing:
                raise ValueError(f"Missing bounds for parameters: {missing}")

            # Per-parameter bounds
            bounds = jnp.array([bounds_dict[param] for param in param_names])
            min_val = bounds[:, 0]
            max_val = bounds[:, 1]

        elif (
            "min_val" in param_initialiser_settings
            and "max_val" in param_initialiser_settings
        ):
            # Global bounds for all parameters

            max_val = param_initialiser_settings["max_val"]
            min_val = param_initialiser_settings["min_val"]

        else:
            raise ValueError(
                "param_initialiser_settings must define either 'bounds' or 'min_val' and 'max_val'"
            )

        initial_values = jax.random.uniform(
            key=random_seed,
            shape=(len(param_names),),
            minval=min_val,
            maxval=max_val,
        )

        return initial_values


def pdf_initial_parameters(forward_map, param_initialiser_settings, replica_index=-1):
    """
    Initial values of the PDF parameters (``forward_map.pdf_param_names``).

    PDF model apps can override this provider; the extra parameters of the
    forward map are initialised separately (see ``extra_initial_parameters``).

    Parameters
    ----------
    forward_map: forward_map.ForwardMap

    param_initialiser_settings: dict
        The settings for the initialiser.

    replica_index: int
        The index of the replica.
        Default is -1, in case no replica index is provided.

    Returns
    -------
    initial_values: jnp.array
        The initial values for the PDF parameters.
    """

    def key_fn(seed):
        if seed is None:
            return jax.random.PRNGKey(replica_index)
        return jax.random.PRNGKey(seed + replica_index)

    return _initial_values(
        forward_map.pdf_param_names, param_initialiser_settings, key_fn
    )


def extra_initial_parameters(
    forward_map, extra_parameters_settings=None, replica_index=-1
):
    """
    Initial values of the extra parameters of the forward map
    (``forward_map.extra_param_names``), from the ``initialiser`` of
    ``extra_parameters_settings``. The options are those of
    ``param_initialiser_settings``; the default is zeros.

    The random key differs from the one of the PDF parameters, so that the
    two are independent.

    Parameters
    ----------
    forward_map: forward_map.ForwardMap

    extra_parameters_settings: dict, optional
        Parsed ``extra_parameters_settings`` (see colibriConfig).

    replica_index: int
        The index of the replica.

    Returns
    -------
    jnp.array
        The initial values for the extra parameters (empty if there are none).
    """
    extra_param_names = list(forward_map.extra_param_names)
    if not extra_param_names:
        return jnp.array([])

    settings = dict((extra_parameters_settings or {}).get("initialiser") or {})
    settings.setdefault("type", "zeros")

    def key_fn(seed):
        key = jax.random.PRNGKey((seed or 0) + replica_index)
        return jax.random.fold_in(key, 1)

    return _initial_values(extra_param_names, settings, key_fn)


def _join_initial_parameters(forward_map, pdf_parameters, extra_parameters):
    """
    Joins the initial PDF and extra parameters, checking that each part has
    the size expected by the forward map.
    """
    pdf_parameters = jnp.atleast_1d(jnp.asarray(pdf_parameters))
    extra_parameters = jnp.atleast_1d(jnp.asarray(extra_parameters))

    if pdf_parameters.shape != (forward_map.n_pdf_params,):
        raise ValueError(
            f"pdf_initial_parameters returned {pdf_parameters.shape[0]} values, but "
            f"the forward map has {forward_map.n_pdf_params} PDF parameters. The "
            "PDF initialiser must cover the PDF parameters only: the extra "
            "parameters of the forward map are initialised by colibri."
        )
    if extra_parameters.shape != (len(forward_map.extra_param_names),):
        raise ValueError(
            f"Got {extra_parameters.shape[0]} initial values for "
            f"{len(forward_map.extra_param_names)} extra parameters."
        )

    if extra_parameters.size == 0:
        return pdf_parameters
    return jnp.concatenate(
        [pdf_parameters, extra_parameters.astype(pdf_parameters.dtype)]
    )


def initial_parameters(forward_map, pdf_initial_parameters, extra_initial_parameters):
    """
    Initial values of all the fit parameters, in the order of
    ``forward_map.param_names``: the PDF parameters (``pdf_initial_parameters``,
    which PDF model apps may override) followed by the extra parameters.
    """
    return _join_initial_parameters(
        forward_map, pdf_initial_parameters, extra_initial_parameters
    )


def all_initial_parameters(
    forward_map,
    param_initialiser_settings,
    extra_parameters_settings=None,
    replica_index=-1,
):
    """
    As ``initial_parameters``, for code that needs several initialisations
    (e.g. one per Hessian fit iteration). Uses colibri's PDF initialiser.
    """
    return _join_initial_parameters(
        forward_map,
        pdf_initial_parameters(forward_map, param_initialiser_settings, replica_index),
        extra_initial_parameters(forward_map, extra_parameters_settings, replica_index),
    )
