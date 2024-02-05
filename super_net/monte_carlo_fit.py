"""
super_net.mc_fit.py

This module contains the main Monte Carlo fitting routine of super_net.

"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import optax
import logging
import pandas as pd
import os


from super_net.constants import XGRID
from super_net.data_batch import data_batches
from super_net.lhapdf import write_exportgrid

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MonteCarloFit:
    """
    Dataclass containing the results and specs of a Monte Carlo fit.

    Attributes
    ----------
    monte_carlo_specs: dict
        Dictionary containing the settings of the Monte Carlo fit.
    training_loss: jnp.array
        Array containing the training loss.
    validation_loss: jnp.array
        Array containing the validation loss.

    """

    monte_carlo_specs: dict
    training_loss: jnp.array
    validation_loss: jnp.array


def monte_carlo_fit(
    _chi2_training_data_with_positivity,
    _chi2_validation_data_with_positivity,
    len_trval_data,
    pdf_model,
    mc_initial_parameters,
    replica_index,
    output_path,
    optimizer_provider,
    early_stopper,
    max_epochs,
    batch_size=128,
    batch_seed=1,
    alpha=1e-7,
    lambda_positivity=1000,
):
    """
    This function performs a Monte Carlo fit.


    Parameters
    ----------
    _chi2_training_data_with_positivity: PjitFunction
        Function that computes the chi2 of the training data.

    _chi2_validation_data_with_positivity: PjitFunction
        Function that computes the chi2 of the validation data.

    len_trval_data: tuple
        Tuple containing the length of the training and validation data.

    pdf_model: PDFModel
        A PDFModel specifying the way in which the PDF is constructed from
        the parameters.

    mc_initial_parameters: jnp.array
        Initial parameters for the Monte Carlo fit.

    replica_index: int
        Index of the replica.

    output_path: str
        Path to the output directory.

    optimizer_provider: optax._src.base.GradientTransformationExtraArgs
        Optax optimizer.

    early_stopper: flax.training.early_stopping.EarlyStopping
        Early stopping criteria.

    max_epochs: int
        Number of maximum epochs.

    batch_size: int, optional
        Size of batches during training. Defaults to 128.

    batch_seed: int, optional
        Seed used to construct the batches. Defaults to 1.

    alpha: float, optional
        Alpha parameter of the ELU positivity penalty term. Defaults to 1e-7.

    lambda_positivity: int, optional
        Lagrange multiplier of the positivity penalty. Defaults to 1000.

    Returns
    -------
    MonteCarloFit: The result of the fit with following attributes:
        monte_carlo_specs: dict
        training_loss: jnp.array
        validation_loss: jnp.array
    """

    fit_grid_values_func = pdf_model.grid_values_func(XGRID)

    @jax.jit
    def loss_training(parameters, batch_idx):
        pdf = fit_grid_values_func(parameters)

        return _chi2_training_data_with_positivity(
            pdf, batch_idx, alpha, lambda_positivity
        )

    @jax.jit
    def loss_validation(parameters):
        pdf = fit_grid_values_func(parameters)

        return _chi2_validation_data_with_positivity(pdf, alpha, lambda_positivity)

    @jax.jit
    def step(params, opt_state, batch_idx):
        loss_value, grads = jax.value_and_grad(loss_training)(params, batch_idx)
        updates, opt_state = optimizer_provider.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    log.info("Starting Monte Carlo fit...")

    len_tr_idx, len_val_idx = len_trval_data

    log.debug(f"len_tr_idx: {len_tr_idx}, len_val_idx: {len_val_idx}")

    loss = []
    val_loss = []

    opt_state = optimizer_provider.init(mc_initial_parameters)
    parameters = mc_initial_parameters.copy()

    data_batch = data_batches(len_tr_idx, batch_size, batch_seed)
    batches = data_batch.data_batch_stream_index()
    num_batches = data_batch.num_batches
    batch_size = data_batch.batch_size

    for i in range(max_epochs):
        epoch_loss = 0
        epoch_val_loss = 0

        for _ in range(num_batches):
            batch = next(batches)

            parameters, opt_state, loss_value = step(parameters, opt_state, batch)

            epoch_loss += loss_training(parameters, batch) / batch_size

        epoch_val_loss += loss_validation(parameters) / len_val_idx
        epoch_loss /= num_batches

        _, early_stopper = early_stopper.update(epoch_val_loss)
        if early_stopper.should_stop:
            log.info("Met early stopping criteria, breaking...")
            break

        if i % 50 == 0:
            log.info(
                f"step {i}, loss: {epoch_loss:.3f}, validation_loss: {epoch_val_loss:.3f}"
            )
            log.info(f"epoch:{i}, early_stopper: {early_stopper}")
            # store loss values every 50 epochs
            loss.append(epoch_loss)
            val_loss.append(epoch_val_loss)

    df = pd.DataFrame(parameters, index=pdf_model.param_names).T

    # In a Monte Carlo fit, replicas are written to the fit_replicas
    # directory, and mc_postfit must then be applied to select valid ones
    # based on a chi2 threshold. No such problem exists for Nested Sampling
    replicas_path = str(output_path) + "/fit_replicas"
    if not os.path.exists(replicas_path):
        os.mkdir(replicas_path)

    # Finish by writing the export grid, ready for evolution
    log.info(f"Writing exportgrid for replica {replica_index}")
    write_exportgrid(
        jnp.array(df.iloc[0, :].tolist()),
        pdf_model,
        replica_index,
        output_path,
        monte_carlo=True,
    )

    # Save the output to csv
    df.to_csv(
        replicas_path
        + f"/replica_{replica_index}/"
        + f"/mc_result_replica_{replica_index}.csv"
    )

    # Save the training and validation loss
    df = pd.DataFrame(
        {
            "epochs": range(len(loss)),
            "training_loss": loss,
            "validation_loss": val_loss,
        }
    )
    df.to_csv(
        str(output_path) + f"/fit_replicas/replica_{replica_index}" + "/mc_loss.csv",
        index=False,
    )

    return MonteCarloFit(
        monte_carlo_specs={
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "batch_seed": batch_seed,
            "alpha": alpha,
            "lambda_positivity": lambda_positivity,
        },
        training_loss=jnp.array(loss),
        validation_loss=jnp.array(val_loss),
    )
