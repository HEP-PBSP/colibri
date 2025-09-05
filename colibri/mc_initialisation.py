import jax
import jax.numpy as jnp
import logging

log = logging.getLogger(__name__)


def mc_initial_parameters(pdf_model, mc_initialiser_settings, replica_index):
    """
    This function initialises the parameters in a Monte Carlo fit.

    Parameters
    ----------
    pdf_model: pdf_mode.PDFModel
        The PDF model to initialise the parameters for.

    mc_initialiser_settings: dict
        The settings for the initialiser.

    replica_index: int
        The index of the replica.

    Returns
    -------
    initial_values: jnp.array
        The initial values for the parameters.
    """
    if mc_initialiser_settings["type"] not in ("zeros", "normal", "uniform"):
        log.warning(
            f"MC initialiser type {mc_initialiser_settings['type']} not recognised, using default: 'zeros' instead."
        )

        mc_initialiser_settings["type"] = "zeros"

    if mc_initialiser_settings["type"] == "zeros":
        return jnp.array([0.0] * len(pdf_model.param_names))

    if "random_seed" in mc_initialiser_settings:
        random_seed = jax.random.PRNGKey(
            mc_initialiser_settings["random_seed"] + replica_index
        )
    else:
        random_seed = jax.random.PRNGKey(replica_index)

    param_names = pdf_model.param_names

    if mc_initialiser_settings["type"] == "normal":
        means_setting = mc_initialiser_settings.get("means", 0.0)
        stds_setting = mc_initialiser_settings.get("stds", 1.0)

        if (
            "means" not in mc_initialiser_settings
            and "stds" not in mc_initialiser_settings
        ):
            log.warning(
                "mc_initialiser_settings: No 'means' or 'stds' provided. "
                "Using default normal distribution N(0, 1) for all parameters."
            )

        if "means" in mc_initialiser_settings and "stds" not in mc_initialiser_settings:
            log.warning(
                "mc_initialiser_settings: 'means' provided without 'stds'. "
                "Using default std=1.0 for all parameters."
            )

        if "stds" in mc_initialiser_settings and "means" not in mc_initialiser_settings:
            log.warning(
                "mc_initialiser_settings: 'stds' provided without 'means'. "
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

    if mc_initialiser_settings["type"] == "uniform":
        if "bounds" in mc_initialiser_settings:
            # Use param names from the model to order bounds correctly
            # param_names = pdf_model.param_names
            bounds_dict = mc_initialiser_settings["bounds"]

            missing = [p for p in param_names if p not in bounds_dict]
            if missing:
                raise ValueError(f"Missing bounds for parameters: {missing}")

            # Per-parameter bounds
            bounds = jnp.array([bounds_dict[param] for param in param_names])
            min_val = bounds[:, 0]
            max_val = bounds[:, 1]

        elif (
            "min_val" in mc_initialiser_settings
            and "max_val" in mc_initialiser_settings
        ):
            # Global bounds for all parameters

            max_val = mc_initialiser_settings["max_val"]
            min_val = mc_initialiser_settings["min_val"]

        else:
            raise ValueError(
                "mc_initialiser_settings must define either 'bounds' or 'min_val' and 'max_val'"
            )

        initial_values = jax.random.uniform(
            key=random_seed,
            shape=(len(pdf_model.param_names),),
            minval=min_val,
            maxval=max_val,
        )

        return initial_values
