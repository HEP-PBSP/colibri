"""
Module for testing the utils module.
"""

import jax
import numpy as np

from colibri.utils import cast_to_numpy

def test_cast_to_numpy():
    """
    test the cast_to_numpy function
    """

    @cast_to_numpy
    @jax.jit
    def test_func(x):
        return x
    
    x = jax.numpy.array([1, 2, 3])
    
    assert type(test_func(x)) == np.ndarray

    

