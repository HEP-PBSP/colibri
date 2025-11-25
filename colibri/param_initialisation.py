import jax
import jax.numpy as jnp
import logging

log = logging.getLogger(__name__)


def pdf_initial_parameters(pdf_model, param_initialiser_settings, replica_index=-1):
    """
    This function provides initial parameters for the PDF model.

    Parameters
    ----------
    pdf_model: pdf_mode.PDFModel
        The PDF model to initialise the parameters for.

    param_initialiser_settings: dict
        The settings for the initialiser.

    replica_index: int
        The index of the replica.
        Default is -1, in case no replica index is provided.

    Returns
    -------
    initial_values: jnp.array
        The initial values for the parameters.
    """
    if param_initialiser_settings["type"] not in ("zeros", "normal", "uniform"):
        log.warning(
            f"MC initialiser type {param_initialiser_settings['type']} not recognised, using default: 'zeros' instead."
        )

        param_initialiser_settings["type"] = "zeros"

    if param_initialiser_settings["type"] == "zeros":
        return jnp.array([0.0] * len(pdf_model.full_param_names))

    if "random_seed" in param_initialiser_settings:
        random_seed = jax.random.PRNGKey(
            param_initialiser_settings["random_seed"] + replica_index
        )
    else:
        random_seed = jax.random.PRNGKey(replica_index)

    full_param_names = pdf_model.full_param_names

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
                if len(setting) != len(full_param_names):
                    raise ValueError(
                        f"'{name}' dict must have one entry per parameter "
                        f"(got {len(setting)} for {len(full_param_names)} parameters)."
                    )
                return jnp.array([setting.get(p, default) for p in full_param_names])
            # If scalar → broadcast
            elif isinstance(setting, (int, float)):
                return jnp.full(len(full_param_names), setting)
            else:
                raise TypeError(f"'{name}' must be dict or scalar, got {type(setting)}")

        means = expand(means_setting, 0.0, "means")
        stds = expand(stds_setting, 1.0, "stds")

        normal_samples = jax.random.normal(
            key=random_seed, shape=(len(full_param_names),)
        )
        return means + stds * normal_samples

    if param_initialiser_settings["type"] == "uniform":
        if "bounds" in param_initialiser_settings:
            # Use param names from the model to order bounds correctly
            bounds_dict = param_initialiser_settings["bounds"]

            missing = [p for p in full_param_names if p not in bounds_dict]
            if missing:
                raise ValueError(f"Missing bounds for parameters: {missing}")

            # Per-parameter bounds
            bounds = jnp.array([bounds_dict[param] for param in full_param_names])
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
            shape=(len(pdf_model.full_param_names),),
            minval=min_val,
            maxval=max_val,
        )

        return initial_values
