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
    training_indices = jnp.arange(n_points)
    y = jnp.full((n_points,), 2.0)
    batch_size = 5
    batches = data_batches(
        training_indices=training_indices, batch_size=batch_size, batch_seed=0
    )

    def training_loss_fn(params, batch):
        # Mean squared error on the batch
        batch_vals = y[batch.idx]
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

    def training_loss_fn(params, _batch):
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


def test_epoch_zero_can_be_selected_as_best_epoch():
    """An optimum reached at epoch zero must not trigger last-epoch fallback."""

    def training_loss_fn(params, _batch):
        return params**2

    # Deliberately worsens as training moves toward its own optimum.
    def validation_loss_fn(params):
        return (params - 2.0) ** 2

    result = run_gradient_descent(
        initial_parameters=jnp.array(1.0),
        training_loss_fn=training_loss_fn,
        validation_loss_fn=validation_loss_fn,
        optimizer=optax.sgd(learning_rate=0.25),
        early_stopper=EarlyStopping(min_delta=0.0, patience=100),
        max_epochs=3,
        data_batch=None,
        record_every=1,
        threshold_chi2=10.0,
    )

    assert result.best_epoch["epoch"] == 0
    assert jnp.allclose(result.best_epoch["best_parameters"], 0.5)


def test_threshold_uses_chi2_per_data_point_but_improvement_uses_total_loss():
    """The threshold follows n3fit's normalized-chi2 selection logic."""

    def training_loss_fn(params, _batch):
        return params**2

    def validation_loss_fn(params):
        # After epoch zero params=0.5, giving total chi2=20.25. This fails a
        # raw threshold of 3.5 but passes 20.25 / 10 < 3.5.
        return (params + 4.0) ** 2

    result = run_gradient_descent(
        initial_parameters=jnp.array(1.0),
        training_loss_fn=training_loss_fn,
        validation_loss_fn=validation_loss_fn,
        optimizer=optax.sgd(learning_rate=0.25),
        early_stopper=EarlyStopping(min_delta=0.0, patience=100),
        max_epochs=1,
        data_batch=None,
        record_every=1,
        threshold_chi2=3.5,
        validation_ndata=10,
    )

    assert result.best_epoch["epoch"] == 0
    assert jnp.allclose(result.best_epoch["best_val_loss"], 20.25)
