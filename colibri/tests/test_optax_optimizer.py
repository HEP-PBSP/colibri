"""
colibri.tests.test_optax_optimizer

Tests for the optax_optimizer module.
"""

import pytest
import jax.numpy as jnp
import optax

from colibri.optax_optimizer import optimizer_provider


def test_optimizer_provider_returns_adam():
    """Check that optimizer_provider returns an Adam optimizer with given hyperparams."""
    optimizer_settings = {
        "optimizer": "adam",
        "clipnorm": None,
        "optimizer_hyperparams": {"learning_rate": 0.01},
    }

    opt = optimizer_provider(optimizer_settings=optimizer_settings)

    # It should be an Optax optimizer
    assert isinstance(opt, optax.GradientTransformation)

    # Test it can initialize state on dummy params
    params = {"w": jnp.array([1.0, 2.0])}
    state = opt.init(params)
    assert state is not None


def test_optimizer_provider_with_clipnorm():
    """Check that optimizer_provider wraps optimizer with gradient clipping when clipnorm is set."""
    optimizer_settings = {
        "optimizer": "adam",
        "optimizer_hyperparams": {"learning_rate": 0.01},
        "clipnorm": 1.0,
    }

    opt = optimizer_provider(optimizer_settings=optimizer_settings)

    assert isinstance(opt, optax.GradientTransformation)

    # Check it still works end-to-end (init + update)
    params = {"w": jnp.array([1.0, 2.0])}
    state = opt.init(params)
    grads = {"w": jnp.array([0.1, -0.1])}
    updates, new_state = opt.update(grads, state, params)
    new_params = optax.apply_updates(params, updates)

    assert new_state is not None
    assert all(k in new_params for k in params)


def test_optimizer_provider_with_scheduler():
    """Check that optimizer_provider correctly applies a learning rate scheduler."""
    optimizer_settings = {
        "optimizer": "adam",
        "optimizer_hyperparams": {},
        "scheduler": {
            "name": "linear_schedule",
            "params": {
                "init_value": 0.001,
                "end_value": 0.0,
                "transition_begin": 0,
                "transition_steps": 10,
            },
        },
        "clipnorm": None,
    }

    opt = optimizer_provider(optimizer_settings=optimizer_settings)

    assert isinstance(opt, optax.GradientTransformation)

    # Test init and update with dummy params
    params = {"w": jnp.array([1.0, 2.0])}
    state = opt.init(params)
    grads = {"w": jnp.array([0.1, -0.1])}
    updates, new_state = opt.update(grads, state, params)
    new_params = optax.apply_updates(params, updates)

    assert new_state is not None
    assert all(k in new_params for k in params)


def test_optimizer_provider_invalid_optimizer_raises():
    """Invalid optimizer name should raise an AttributeError."""
    optimizer_settings = {
        "optimizer": "not_an_optimizer",
        "optimizer_hyperparams": {},
        "clipnorm": None,
    }

    with pytest.raises(AttributeError):
        optimizer_provider(optimizer_settings=optimizer_settings)
