"""
colibri.gradient_descent

Generic JAX + Optax gradient descent training loop.

Design goals:
 - Keep lightweight: caller provides (jit'ed) training & validation loss fns.
 - Handle batching externally: caller supplies an iterator over batch indices.
 - Optional early stopping via a flax EarlyStopping-like object with an
   .update(val_loss) -> object and .should_stop boolean attribute.
 - Record losses every `record_every` epochs.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable, Any, Dict

import jax
import jax.numpy as jnp
import optax
import colibri

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GradientDescentResult:
    """Result of a generic gradient descent run.

    Attributes
    ----------
    optimized_parameters: jnp.ndarray
        Final optimized parameters.
    training_loss: jnp.array
        Recorded (epoch) training losses (sampled according to record_every).
    validation_loss: jnp.array
        Recorded (epoch) validation losses (sampled according to record_every).
    specs: dict
        Dictionary of settings used for the run (epochs, batch size, etc.).
    """

    optimized_parameters: Any
    training_loss: jnp.array
    validation_loss: jnp.array
    specs: Dict[str, Any]


def run_gradient_descent(
    initial_parameters: jnp.ndarray,
    training_loss_fn: Callable[[jnp.ndarray, int], jnp.ndarray],
    validation_loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
    optimizer: optax.GradientTransformation,
    early_stopper: Any,
    max_epochs: int,
    data_batch: colibri.DataBatches,
    record_every: int = 50,
) -> GradientDescentResult:
    """Generic gradient descent loop.

    Parameters
    ----------
    initial_parameters : jnp.ndarray
        Starting parameters.

    training_loss_fn : callable -> scalar
        Per-batch loss function (already jit'ed by caller if desired).

    validation_loss_fn : callable -> scalar
        Validation loss function (already jit'ed by caller if desired).

    optimizer : optax.GradientTransformation
        Optax optimizer.

    early_stopper : object
        Must expose .update(val_loss) -> object and .should_stop bool.

    max_epochs : int
        Maximum epochs to run.

    data_batch : colibri.DataBatches
        DataBatches object providing batching information.

    record_every : int, default 50
        Record losses every this many epochs.
    """

    params = initial_parameters
    opt_state = optimizer.init(params)

    @jax.jit
    def _step(p, ostate, batch_idx):
        (loss_value, grads) = jax.value_and_grad(training_loss_fn)(p, batch_idx)
        updates, ostate = optimizer.update(grads, ostate, p)
        p = optax.apply_updates(p, updates)
        return p, ostate, loss_value

    train_losses = []
    val_losses = []

    batches_iter = data_batch.data_batch_stream_index()
    num_batches = data_batch.num_batches
    batch_size = data_batch.batch_size

    for epoch in range(max_epochs):
        epoch_train_loss = 0.0
        for _ in range(num_batches):
            batch_idx = next(batches_iter)
            params, opt_state, batch_loss = _step(params, opt_state, batch_idx)
            epoch_train_loss += batch_loss
        epoch_val_loss = validation_loss_fn(params)

        early_stopper = early_stopper.update(epoch_val_loss)

        if record_every and (epoch % record_every == 0):
            log.info(
                f"Epoch {epoch}, loss: {epoch_train_loss:.3f}, validation_loss: {epoch_val_loss:.3f}"
            )
            log.info(f"    Early_stopper: {early_stopper}")
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)

        if early_stopper.should_stop:
            log.info(f"Early stopping at epoch {epoch}")
            break

    return GradientDescentResult(
        optimized_parameters=params,
        training_loss=jnp.array(train_losses),
        validation_loss=jnp.array(val_losses),
        specs={
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "record_every": record_every,
        },
    )
