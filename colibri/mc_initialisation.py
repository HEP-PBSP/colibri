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
        mean_dict = mc_initialiser_settings.get("means", None)
        std_dict = mc_initialiser_settings.get("stds", None)

        mean_val = mc_initialiser_settings.get("mean_val", None)
        std_val = mc_initialiser_settings.get("std_val", None)

        # Both 'means' and 'stds' provided
        if (mean_dict is not None) and (std_dict is not None):
            if len(mean_dict) != len(param_names) or len(std_dict) != len(param_names):
                raise ValueError(
                    f"'means' and 'stds' must have exactly one entry per parameter "
                    f"(you wrote {len(mean_dict)} means and {len(std_dict)} stds for {len(param_names)} parameters)"
                )
            means = jnp.array([mean_dict[p] for p in param_names])
            stds = jnp.array([std_dict[p] for p in param_names])

        # Only 'means' provided
        elif mean_dict is not None:
            log.warning(
                "mc_initialiser_settings: 'means' provided without 'stds'. "
                "Using std=1.0 for all parameters."
            )
            means = jnp.array([mean_dict.get(p, 0.0) for p in param_names])
            stds = jnp.ones(len(param_names))

        # Only 'stds' provided
        elif std_dict is not None:
            log.warning(
                "mc_initialiser_settings: 'stds' provided without 'means'. "
                "Using mean=0.0 for all parameters."
            )
            means = jnp.zeros(len(param_names))
            stds = jnp.array([std_dict.get(p, 1.0) for p in param_names])

        # Nothing provided
        else:
            if (mean_val is None) and (std_val is None):
                log.warning(
                    "mc_initialiser_settings: 'means' and 'stds' not provided. "
                    "Using default mean=0.0 and std=1.0 for all parameters."
                )
                mval = 0.0
                sval = 1.0
            elif (mean_val is not None) and (std_val is not None):
                mval = mean_val
                sval = std_val
            elif (mean_val is not None) and (std_val is None):
                raise ValueError("mc_initialiser_settings: 'std_val' missing.")
            elif (mean_val is None) and (std_val is not None):
                raise ValueError("mc_initialiser_settings: 'mean_val' missing.")
            means = jnp.full(len(param_names), mval)
            stds = jnp.full(len(param_names), sval)

        normal_samples = jax.random.normal(
            key=random_seed,
            shape=(len(param_names),),
        )

        initial_values = means + stds * normal_samples
        return initial_values

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
