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
        mean_dict = mc_initialiser_settings.get("means", {})
        std_dict = mc_initialiser_settings.get("stds", {})

        # Default mean = 0.0, std = 1.0 if not specified
        means = jnp.array([mean_dict.get(p, 0.0) for p in param_names])
        stds = jnp.array([std_dict.get(p, 1.0) for p in param_names])

        # Check that, if mean is specified, so is std and vice versa
        if ("means" in mc_initialiser_settings) ^ ("stds" in mc_initialiser_settings):
            raise ValueError("Both 'means' and 'stds' must be specified together.")

        # Check that there is exactly one mean/std value per parameter

        if ("means" in mc_initialiser_settings) and ("stds" in mc_initialiser_settings):
            if len(mean_dict) != len(param_names) or len(std_dict) != len(param_names):
                raise ValueError(
                    f"'means' and 'stds' must have exactly one entry per parameter "
                    f"(you wrote {len(mean_dict)} means and {len(std_dict)} stds for {len(param_names)} parameters)"
                )

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
