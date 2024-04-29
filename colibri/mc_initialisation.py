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

    if mc_initialiser_settings["type"] == "normal":
        # Currently, only one standard deviation around a zero mean is implemented
        initial_values = jax.random.normal(
            key=random_seed,
            shape=(len(pdf_model.param_names),),
        )
        return initial_values

    if mc_initialiser_settings["type"] == "uniform":
        max_val = mc_initialiser_settings["max_val"]
        min_val = mc_initialiser_settings["min_val"]
        initial_values = jax.random.uniform(
            key=random_seed,
            shape=(len(pdf_model.param_names),),
            minval=min_val,
            maxval=max_val,
        )
        return initial_values
