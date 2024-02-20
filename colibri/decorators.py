"""
colibri.decorators

This module contains decorators that can be used to modify the behaviour of functions.
"""

import functools

import jax


def enable_x64(func):
    """
    Enable 64-bit precision
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        jax.config.update("jax_enable_x64", True)
        result = func(*args, **kwargs)
        return result

    return wrapper
