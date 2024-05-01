import jax


def bayesian_prior(prior_settings):
    """
    Produces a prior transform function.

    Parameters
    ----------
    prior_settings: dict
        The settings for the prior transform.

    Returns
    -------
    prior_transform: @jax.jit CompiledFunction
        The prior transform function.
    """
    if prior_settings["type"] == "uniform_parameter_prior":
        max_val = prior_settings["max_val"]
        min_val = prior_settings["min_val"]

        @jax.jit
        def prior_transform(cube):
            return cube * (max_val - min_val) + min_val

    else:
        raise ValueError("Invalid prior type.")
    return prior_transform
