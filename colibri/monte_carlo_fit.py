"""
colibri.monte_carlo_fit.py

This module contains the main Monte Carlo fitting routine of colibri.

"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax.extend import backend as jbackend
import logging
import pandas as pd
import os
import time

from colibri.data_batch import data_batches
from colibri.mc_utils import write_exportgrid_mc
from colibri.gradient_descent import run_gradient_descent

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
    optimized_parameters: jnp.array
        Array containing the optimized parameters.
    """

    monte_carlo_specs: dict
    training_loss: jnp.array
    validation_loss: jnp.array
    optimized_parameters: jnp.array


def monte_carlo_fit(
    mc_log_likelihood,
    len_trval_data,
    pdf_initial_parameters,
    optimizer_provider,
    early_stopper,
    max_epochs,
    batch_size=None,
    batch_seed=1,
):
    """
    This function performs a Monte Carlo fit.


    Parameters
    ----------
    mc_log_likelihood: tuple
        Tuple containing the training and validation likelihoods.

    len_trval_data: tuple
        Tuple containing the length of the training and validation data.

    pdf_initial_parameters: jnp.array
        Initial parameters for the Monte Carlo fit.

    optimizer_provider: optax._src.base.GradientTransformationExtraArgs
        Optax optimizer.

    early_stopper: flax.training.early_stopping.EarlyStopping
        Early stopping criteria.

    max_epochs: int
        Number of maximum epochs.

    batch_size: int, default is None which sets it to the full size of data
        Size of batches during training.

    batch_seed: int, optional
        Seed used to construct the batches. Defaults to 1.

    Returns
    -------
    MonteCarloFit: The result of the fit with following attributes:
        monte_carlo_specs: dict
        training_loss: jnp.array
        validation_loss: jnp.array
    """

    len_tr_idx, len_val_idx = len_trval_data

    @jax.jit
    def loss_training(
        parameters,
        batch_idx,
    ):
        return -2 * mc_log_likelihood[0](parameters, batch_idx) / len_tr_idx

    @jax.jit
    def loss_validation(parameters):

        val = -2 * mc_log_likelihood[1](parameters)

        return val / len_val_idx if len_val_idx > 0 else val

    log.info(f"Running fit with backend: {jbackend.get_backend().platform}")
    log.info("Starting Monte Carlo fit...")
    t0 = time.time()

    data_batch = data_batches(len_tr_idx, batch_size, batch_seed)

    # Delegate to generic gradient descent
    gd_result = run_gradient_descent(
        initial_parameters=pdf_initial_parameters.copy(),
        training_loss_fn=loss_training,
        validation_loss_fn=loss_validation,
        optimizer=optimizer_provider,
        early_stopper=early_stopper,
        max_epochs=max_epochs,
        data_batch=data_batch,
        record_every=50,
    )

    t1 = time.time()
    log.info("MONTE CARLO RUNNING TIME: %f" % (t1 - t0))

    return MonteCarloFit(
        monte_carlo_specs={
            "max_epochs": max_epochs,
            "batch_size": data_batch.batch_size,
            "batch_seed": batch_seed,
        },
        training_loss=gd_result.training_loss,
        validation_loss=gd_result.validation_loss,
        optimized_parameters=gd_result.optimized_parameters,
    )


def run_monte_carlo_fit(monte_carlo_fit, pdf_model, output_path, replica_index, Q0):
    """
    Runs the Monte Carlo fit and writes the output to the output directory.

    Parameters
    ----------
    monte_carlo_fit: MonteCarloFit
        The results of the Monte Carlo fit.

    pdf_model: pdf_model.PDFModel
        The PDF model used in the fit.

    output_path: pathlib.PosixPath
        Path to the output folder.

    replica_index: int
        The index of the replica being written.

    Q0: float
        The scale at which to export the PDFs.
    """
    mc_fit = monte_carlo_fit

    df = pd.DataFrame(mc_fit.optimized_parameters, index=pdf_model.param_names).T

    # In a Monte Carlo fit, replicas are written to the fit_replicas
    # directory, and mc_postfit must then be applied to select valid ones
    # based on a chi2 threshold. No such problem exists for Nested Sampling
    replicas_path = str(output_path) + "/fit_replicas"
    if not os.path.exists(replicas_path):
        os.mkdir(replicas_path)

    # Finish by writing the export grid, ready for evolution
    log.info(f"Writing exportgrid for replica {replica_index}")
    write_exportgrid_mc(
        jnp.array(df.iloc[0, :].tolist()),
        pdf_model,
        replica_index,
        output_path,
        Q0,
    )

    df.to_csv(
        replicas_path
        + f"/replica_{replica_index}/"
        + f"/mc_result_replica_{replica_index}.csv"
    )
    # Save the training and validation loss
    df = pd.DataFrame(
        {
            "epochs": range(len(mc_fit.training_loss)),
            "training_loss": mc_fit.training_loss,
            "validation_loss": mc_fit.validation_loss,
        }
    )
    df.to_csv(
        str(output_path) + f"/fit_replicas/replica_{replica_index}" + "/mc_loss.csv",
        index=False,
        float_format="%.5e",
    )
