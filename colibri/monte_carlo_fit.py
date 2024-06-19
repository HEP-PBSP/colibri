"""
colibri.monte_carlo_fit.py

This module contains the main Monte Carlo fitting routine of colibri.

"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import optax
import logging
import pandas as pd
import os
import time

from colibri.data_batch import data_batches
from colibri.lhapdf import write_exportgrid

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
    _chi2_training_data_with_positivity,
    _chi2_validation_data_with_positivity,
    _pred_data,
    fast_kernel_arrays,
    positivity_fast_kernel_arrays,
    len_trval_data,
    pdf_model,
    mc_initial_parameters,
    optimizer_provider,
    early_stopper,
    max_epochs,
    FIT_XGRID,
    batch_size=None,
    batch_seed=1,
    alpha=1e-7,
    lambda_positivity=1000,
    float_type=None,
):
    """
    This function performs a Monte Carlo fit.


    Parameters
    ----------
    _chi2_training_data_with_positivity: PjitFunction
        Function that computes the chi2 of the training data.

    _chi2_validation_data_with_positivity: PjitFunction
        Function that computes the chi2 of the validation data.

    _pred_data: theory_predictions.make_pred_data
        The function to compute the theory predictions.

    len_trval_data: tuple
        Tuple containing the length of the training and validation data.

    pdf_model: pdf_model.PDFModel
        A PDFModel specifying the way in which the PDF is constructed from
        the parameters.

    mc_initial_parameters: jnp.array
        Initial parameters for the Monte Carlo fit.

    optimizer_provider: optax._src.base.GradientTransformationExtraArgs
        Optax optimizer.

    early_stopper: flax.training.early_stopping.EarlyStopping
        Early stopping criteria.

    max_epochs: int
        Number of maximum epochs.

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    batch_size: int, default is None which sets it to the full size of data
        Size of batches during training.

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

    pred_and_pdf = pdf_model.pred_and_pdf_func(
        FIT_XGRID, forward_map=_pred_data, float_type=float_type
    )

    @jax.jit
    def loss_training(
        parameters,
        batch_idx,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        alpha,
        lambda_positivity,
    ):
        predictions, pdf = pred_and_pdf(parameters, fast_kernel_arrays)

        return _chi2_training_data_with_positivity(
            predictions,
            pdf,
            batch_idx,
            alpha,
            lambda_positivity,
            positivity_fast_kernel_arrays,
        )

    @jax.jit
    def loss_validation(
        parameters,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        alpha,
        lambda_positivity,
    ):
        predictions, pdf = pred_and_pdf(parameters, fast_kernel_arrays)

        return _chi2_validation_data_with_positivity(
            predictions, pdf, alpha, lambda_positivity, positivity_fast_kernel_arrays
        )

    @jax.jit
    def step(
        params,
        opt_state,
        batch_idx,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        alpha,
        lambda_positivity,
    ):
        loss_value, grads = jax.value_and_grad(loss_training)(
            params,
            batch_idx,
            fast_kernel_arrays,
            positivity_fast_kernel_arrays,
            alpha,
            lambda_positivity,
        )
        updates, opt_state = optimizer_provider.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    log.info(f"Running fit with backend: {jax.lib.xla_bridge.get_backend().platform}")

    log.info("Starting Monte Carlo fit...")
    t0 = time.time()

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

            parameters, opt_state, loss_value = step(
                parameters,
                opt_state,
                batch,
                fast_kernel_arrays,
                positivity_fast_kernel_arrays,
                alpha,
                lambda_positivity,
            )

            epoch_loss += (
                loss_training(
                    parameters,
                    batch,
                    fast_kernel_arrays,
                    positivity_fast_kernel_arrays,
                    alpha,
                    lambda_positivity,
                )
                / batch_size
            )

        epoch_val_loss += (
            loss_validation(
                parameters,
                fast_kernel_arrays,
                positivity_fast_kernel_arrays,
                alpha,
                lambda_positivity,
            )
            / len_val_idx
        )
        epoch_loss /= num_batches

        early_stopper = early_stopper.update(epoch_val_loss)
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

    t1 = time.time()

    log.info("MONTE CARLO RUNNING TIME: %f" % (t1 - t0))

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
        optimized_parameters=parameters,
    )


def run_monte_carlo_fit(monte_carlo_fit, pdf_model, output_path, replica_index):
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
    write_exportgrid(
        jnp.array(df.iloc[0, :].tolist()),
        pdf_model,
        replica_index,
        output_path,
        monte_carlo=True,
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
