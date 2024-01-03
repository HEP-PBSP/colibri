from wmin.wmin_utils import weights_initializer_provider

import numpy as np
import jax
import jax.numpy as jnp

from wmin.tests.conftest import (
    RNG_SEED_WEIGHTS,
    WEIGHTS_INIT_SHAPE,
    WEIGHTS_INITI_UNIF_MINVAL,
    WEIGHTS_INITI_UNIF_MAXVAL,
)


def test_weights_initializer_provider():
    """
    test that the initializer of the weights in a 'weight minimization fit' behaves as expected
    """

    # test zeros, in case unknown initializer is given
    w_init_zeros = weights_initializer_provider(weights_initializer="something else")(
        WEIGHTS_INIT_SHAPE
    )
    np.testing.assert_allclose(w_init_zeros, jnp.zeros(WEIGHTS_INIT_SHAPE))

    # test random normal
    w_init_normal = weights_initializer_provider(
        weights_initializer="normal", weights_seed=RNG_SEED_WEIGHTS
    )(WEIGHTS_INIT_SHAPE)
    np.testing.assert_allclose(
        w_init_normal,
        jax.random.normal(
            key=jax.random.PRNGKey(RNG_SEED_WEIGHTS), shape=(WEIGHTS_INIT_SHAPE,)
        ),
    )

    # test random uniform
    w_init_uniform = weights_initializer_provider(
        weights_initializer="uniform",
        weights_seed=RNG_SEED_WEIGHTS,
        uniform_minval=-0.1,
        uniform_maxval=0.1,
    )(WEIGHTS_INIT_SHAPE)
    np.testing.assert_allclose(
        w_init_uniform,
        jax.random.uniform(
            key=jax.random.PRNGKey(RNG_SEED_WEIGHTS),
            shape=(WEIGHTS_INIT_SHAPE,),
            minval=WEIGHTS_INITI_UNIF_MINVAL,
            maxval=WEIGHTS_INITI_UNIF_MAXVAL,
        ),
    )
