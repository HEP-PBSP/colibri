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
from typing import Callable, Iterable, Any, Dict

import jax
import jax.numpy as jnp
import optax

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
    fast_kernel_arrays: jnp.ndarray,
    positivity_fast_kernel_arrays: jnp.ndarray,
    optimizer: optax.GradientTransformation,
    early_stopper: Any,
    max_epochs: int,
    batch_indices: Iterable[int],
    num_batches: int,
    batch_size: int,
    record_every: int = 50,
    alpha: float = 1e-7,
    lambda_positivity: float = 1000,
) -> GradientDescentResult:
    """Generic gradient descent loop.

    Parameters
    ----------
    initial_parameters : PyTree
        Starting parameters.

    training_loss_fn : callable(params, batch_idx) -> scalar
        Per-batch loss function (already jit'ed by caller if desired).

    validation_loss_fn : callable(params) -> scalar
        Validation loss function (already jit'ed by caller if desired).

    fast_kernel_arrays : jnp.ndarray
        Fast kernel arrays for convolutions.

    positivity_fast_kernel_arrays : jnp.ndarray
        Fast kernel arrays for positivity constraints.

    optimizer : optax.GradientTransformation
        Optax optimizer.

    early_stopper : object
        Must expose .update(val_loss) -> object and .should_stop bool.

    max_epochs : int
        Maximum epochs to run.

    batch_indices : iterable
        Infinite (or long) iterator yielding batch indices for each step.

    num_batches : int
        Number of batches per epoch.

    batch_size : int
        Size of each batch.

    record_every : int, default 50
        Record losses every this many epochs.

    alpha : float, default 1e-7
        Alpha parameter of the ELU positivity penalty term.

    lambda_positivity : float, default 1000
        Lagrange multiplier of the positivity penalty.
    """

    params = initial_parameters
    opt_state = optimizer.init(params)

    # Wrap the gradient computation
    def _step(
        p,
        ostate,
        batch_idx,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        alpha,
        lambda_positivity,
    ):
        (loss_value, grads) = jax.value_and_grad(training_loss_fn)(
            p,
            batch_idx,
            fast_kernel_arrays,
            positivity_fast_kernel_arrays,
            alpha,
            lambda_positivity,
        )
        updates, ostate = optimizer.update(grads, ostate, p)
        p = optax.apply_updates(p, updates)
        return p, ostate, loss_value

    train_losses = []
    val_losses = []

    # We need a re-iterable / generator consumption; user provides an iterator.
    batches_iter = batch_indices

    for epoch in range(max_epochs):
        epoch_train_loss = 0.0
        for _ in range(num_batches):
            batch_idx = next(batches_iter)
            params, opt_state, _ = _step(
                params,
                opt_state,
                batch_idx,
                fast_kernel_arrays,
                positivity_fast_kernel_arrays,
                alpha,
                lambda_positivity,
            )
            epoch_train_loss += (
                training_loss_fn(
                    params,
                    batch_idx,
                    fast_kernel_arrays,
                    positivity_fast_kernel_arrays,
                    alpha,
                    lambda_positivity,
                )
                / batch_size
            )

        epoch_train_loss /= num_batches

        epoch_val_loss = validation_loss_fn(
            params,
            fast_kernel_arrays,
            positivity_fast_kernel_arrays,
            alpha,
            lambda_positivity,
        )

        early_stopper = early_stopper.update(epoch_val_loss)
        if early_stopper.should_stop:
            log.info(f"Early stopping at epoch {epoch}")
            break

        if record_every and (epoch % record_every == 0):
            log.info(
                f"step {epoch}, loss: {epoch_train_loss:.3f}, validation_loss: {epoch_val_loss:.3f}"
            )
            log.info(f"    early_stopper: {early_stopper}")
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)

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
