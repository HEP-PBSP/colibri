"""
TODO
"""
import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from reportengine import collect

from super_net.data_batch import data_batches
from wmin.wmin_model import WeightMinimizationGrid

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WeightMinimizationFit(WeightMinimizationGrid):
    optimised_wmin_weights: jnp.array
    training_loss: jnp.array
    validation_loss: jnp.array


def weight_minimization_fit(
    make_chi2_training_data_with_positivity,
    make_chi2_validation_data_with_positivity,
    make_data_values,
    weight_minimization_grid,
    optimizer_provider,
    early_stopper,
    max_epochs,
    batch_size=128,
    batch_seed=1,
    alpha=1e-7,
    lambda_positivity=1000,
):
    """
    TODO
    """

    @jax.jit
    def loss_training(weights, batch_idx):
        wmin_weights = jnp.concatenate((jnp.array([1.0]), weights))
        pdf = jnp.einsum(
            "i,ijk", wmin_weights, weight_minimization_grid.wmin_INPUT_GRID
        )
        return make_chi2_training_data_with_positivity(
            pdf, batch_idx, alpha, lambda_positivity
        )

    @jax.jit
    def loss_validation(weights):
        wmin_weights = jnp.concatenate((jnp.array([1.0]), weights))
        pdf = jnp.einsum(
            "i,ijk", wmin_weights, weight_minimization_grid.wmin_INPUT_GRID
        )
        return make_chi2_validation_data_with_positivity(pdf, alpha, lambda_positivity)

    @jax.jit
    def step(params, opt_state, batch_idx):
        loss_value, grads = jax.value_and_grad(loss_training)(params, batch_idx)
        updates, opt_state = optimizer_provider.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    loss = []
    val_loss = []

    opt_state = optimizer_provider.init(weight_minimization_grid.init_wmin_weights)
    weights = weight_minimization_grid.init_wmin_weights

    data_batch = data_batches(
        make_data_values.training_data.n_training_points, batch_size, batch_seed
    )
    batches = data_batch.data_batch_stream_index()
    num_batches = data_batch.num_batches
    batch_size = data_batch.batch_size

    for i in range(max_epochs):
        epoch_loss = 0
        epoch_val_loss = 0

        for _ in range(num_batches):
            batch = next(batches)

            weights, opt_state, loss_value = step(weights, opt_state, batch)

            epoch_loss += loss_training(weights, batch) / batch_size

        epoch_val_loss += (
            loss_validation(weights)
            / make_data_values.validation_data.n_validation_points
        )
        epoch_loss /= num_batches

        loss.append(epoch_loss)
        val_loss.append(epoch_val_loss)

        _, early_stopper = early_stopper.update(epoch_val_loss)
        if early_stopper.should_stop:
            log.info("Met early stopping criteria, breaking...")
            break

        if i % 50 == 0:
            log.info(
                f"step {i}, loss: {epoch_loss:.3f}, validation_loss: {epoch_val_loss:.3f}"
            )
            log.info(f"epoch:{i}, early_stopper: {early_stopper}")

    return WeightMinimizationFit(
        **weight_minimization_grid.to_dict(),
        optimised_wmin_weights=weights,
        training_loss=loss,
        validation_loss=val_loss,
    )


"""
Collect over multiple replica fits.
"""
mc_replicas_weight_minimization_fit = collect(
    "weight_minimization_fit", ("all_wmin_collect_indices",)
)


def monte_carlo_wmin_fit(lhapdf_from_collected_weights):
    log.info("Monte Carlo weight minimization fit completed!")
