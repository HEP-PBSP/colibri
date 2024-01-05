"""
super_net.mc_fit.py

This module provides the functions necessary for performing a MC fit with super_net.

Author: James Moore
Date: 5.1.2023
"""

import jax
import optax
import logging

from super_net.data_batch import data_batches

log = logging.getLogger(__name__)

def mc_fit(
    _chi2_training_data_with_positivity,
    _chi2_validation_data_with_positivity,
    _data_values,
    pdf_model,
    optimizer_provider,
    early_stopper,
    max_epochs,
    batch_size=128,
    batch_seed=1,
    alpha=1e-7,
    lambda_positivity=1000,
    ):

    @jax.jit
    def loss_training(params, batch_idx):
        pdf = pdf_model.grid_values(params)
        return _chi2_training_data_with_positivity(
            pdf, batch_idx, alpha, lambda_positivity
        )

    @jax.jit
    def loss_validation(params):
        pdf = pdf_model.grid_values(params)
        return _chi2_validation_data_with_positivity(pdf, alpha, lambda_positivity)

    @jax.jit
    def step(params, opt_state, batch_idx):
        loss_value, grads = jax.value_and_grad(loss_training)(params, batch_idx)
        updates, opt_state = optimizer_provider.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    loss = []
    val_loss = []

    opt_state = optimizer_provider.init(pdf_model.init_params)
    params = pdf_model.init_params

    data_batch = data_batches(
        _data_values.training_data.n_training_points, batch_size, batch_seed
    )
    batches = data_batch.data_batch_stream_index()
    num_batches = data_batch.num_batches
    batch_size = data_batch.batch_size

    for i in range(max_epochs):
        epoch_loss = 0
        epoch_val_loss = 0

        for _ in range(num_batches):
            batch = next(batches)

            params, opt_state, loss_value = step(params, opt_state, batch)

            epoch_loss += loss_training(params, batch) / batch_size

        epoch_val_loss += (
            loss_validation(params) / _data_values.validation_data.n_validation_points
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

    return 0
