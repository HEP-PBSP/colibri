import jax.numpy as jnp


def sum_rules_constraint(params):
    """
    Implements the sum rule constraint to be imposed when
    fitting PDFs in the weight minimization parametrization.

    Note that in the weight minimization parametrization the
    sum rules constraint is equivalent to sum(w_i) =1

    Parameters
    ----------
    params : jnp.ndarray
            array of the fitted parameters
    
    Returns
    -------
    float
        the deviation of the sum of the weights from 1
    """
    # compute the sum of the weight min. parameters
    params_sum = jnp.sum(params)

    # calculate constraint violation
    constraint_violation = jnp.abs(params_sum - 1.0)
    return constraint_violation