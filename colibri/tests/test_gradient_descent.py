"""
colibri.tests.test_gradient_descent

Tests for the generic gradient descent training loop in `gradient_descent.py`.
"""

import jax.numpy as jnp
import optax
from flax.training.early_stopping import EarlyStopping

from colibri.gradient_descent import run_gradient_descent, GradientDescentResult
from colibri.data_batch import data_batches


def test_run_gradient_descent_no_batch_converges_and_early_stop():
    """Test gradient descent without batching converges near optimum and early stops."""

    target = 3.0

    # Training loss ignores batch indices when no data_batch provided
    def training_loss_fn(params, _batch_idx):  # params is scalar array
        return (params - target) ** 2

    # Validation loss identical
    def validation_loss_fn(params):
        return (params - target) ** 2

    initial_parameters = jnp.array(0.0)
    optimizer = optax.sgd(learning_rate=0.2)

    # Small patience to trigger early stopping well before max_epochs
    early_stopper = EarlyStopping(min_delta=1e-8, patience=5)

    result = run_gradient_descent(
        initial_parameters=initial_parameters,
        training_loss_fn=training_loss_fn,
        validation_loss_fn=validation_loss_fn,
        optimizer=optimizer,
        early_stopper=early_stopper,
        max_epochs=200,
        data_batch=None,  # no batching path
        record_every=1,
    )

    assert isinstance(result, GradientDescentResult)
    # Should early stop before exhausting all epochs
    assert result.training_loss.size < 200
    # Final parameter close to target
    assert jnp.allclose(result.optimized_parameters, target, atol=1e-2)
    # Specs should reflect no batching
    assert result.specs["batch_size"] is None


def test_run_gradient_descent_with_batches_converges():
    """Test gradient descent with simple batching setup converges to mean of targets."""

    # Create synthetic dataset y_i all equal to 2.0 so optimum is p=2.0
    n_points = 20
    y = jnp.full((n_points,), 2.0)
    batch_size = 5
    batches = data_batches(
        n_training_points=n_points, batch_size=batch_size, batch_seed=0
    )

    def training_loss_fn(params, batch_idx):
        # Mean squared error on the batch
        batch_vals = y[batch_idx]
        return jnp.mean((params - batch_vals) ** 2)

    def validation_loss_fn(params):
        return jnp.mean((params - y) ** 2)

    initial_parameters = jnp.array(0.0)
    optimizer = optax.sgd(learning_rate=0.3)
    early_stopper = EarlyStopping(min_delta=1e-8, patience=30)

    result = run_gradient_descent(
        initial_parameters=initial_parameters,
        training_loss_fn=training_loss_fn,
        validation_loss_fn=validation_loss_fn,
        optimizer=optimizer,
        early_stopper=early_stopper,
        max_epochs=80,
        data_batch=batches,
        record_every=5,
    )

    assert isinstance(result, GradientDescentResult)
    # Should have recorded every 5 epochs starting at 0
    assert result.training_loss.size >= 5  # enough progress recorded
    # Parameter close to 2
    assert jnp.allclose(result.optimized_parameters, 2.0, atol=5e-2)
    # Specs batch_size matches
    assert result.specs["batch_size"] == batch_size


def test_run_gradient_descent_record_every_behavior():
    """Test that record_every stores losses at the specified interval."""

    target = -1.0

    def training_loss_fn(params, _batch_idx):
        return (params - target) ** 2

    def validation_loss_fn(params):
        return (params - target) ** 2

    initial_parameters = jnp.array(5.0)
    optimizer = optax.sgd(learning_rate=0.5)
    early_stopper = EarlyStopping(
        min_delta=0.0, patience=100
    )  # effectively no early stop

    max_epochs = 6
    record_every = 2

    result = run_gradient_descent(
        initial_parameters=initial_parameters,
        training_loss_fn=training_loss_fn,
        validation_loss_fn=validation_loss_fn,
        optimizer=optimizer,
        early_stopper=early_stopper,
        max_epochs=max_epochs,
        data_batch=None,
        record_every=record_every,
    )

    # Epochs recorded should be 0,2,4
    assert result.training_loss.size == 3
    assert result.validation_loss.size == 3
    # Monotonic non-increasing validation loss across recorded epochs
    assert jnp.all(result.validation_loss[1:] <= result.validation_loss[:-1] + 1e-10)
