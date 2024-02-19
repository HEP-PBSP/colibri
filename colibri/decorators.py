import functools

import jax


def enable_x64(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        jax.config.update("jax_enable_x64", True)
        result = func(*args, **kwargs)
        jax.config.update("jax_enable_x64", False)
        return result
    return wrapper
