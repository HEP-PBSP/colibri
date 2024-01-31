from validphys.convolution import FK_FLAVOURS

from grid_pdf.grid_pdf_lhapdf import (
    write_exportgrid_from_fit_samples,
)
from super_net.utils import resample_from_ns_posterior

import ultranest
import jax
import jax.numpy as jnp
import pandas as pd
import optax
from super_net.data_batch import data_batches

from dataclasses import dataclass
import time
import logging
import os
import sys

from reportengine import collect

log = logging.getLogger(__name__)

# Check if --debug flag is present
debug_flag = "--debug" in sys.argv

# Set the Ultrnest logging level based on the presence of --debug flag
ultranest_logger = logging.getLogger("ultranest")
ultranest_logger.setLevel(logging.DEBUG if debug_flag else logging.INFO)

# Configure the handler and formatter
handler = logging.StreamHandler(sys.stdout)
ultranest_logger.addHandler(handler)


def ultranest_grid_fit(
    _chi2_with_positivity,
    grid_pdf_model_prior,
    interpolate_grid,
    reduced_xgrids,
    flavour_indices,
    ns_settings,
):
    """
    TODO

    Parameters
    ----------

    Returns
    -------

    """

    @jax.jit
    def log_likelihood(stacked_pdf_grid):
        """
        TODO

        Parameters
        ----------
        stacked_pdf_grid: jnp.array

        Returns
        -------

        """

        pdf = interpolate_grid(stacked_pdf_grid)
        return -0.5 * _chi2_with_positivity(pdf)

    log.info("Starting ULTRANEST run...")
    log.debug(f"ULTRANEST settings: {ns_settings}")

    parameters = [
        f"{FK_FLAVOURS[i]}({j})" for i in flavour_indices for j in reduced_xgrids[i]
    ]

    log.debug(f"ULTRANEST parameters: {parameters}")

    sampler = ultranest.ReactiveNestedSampler(
        parameters,
        log_likelihood,
        grid_pdf_model_prior,
        **ns_settings["ReactiveNS_settings"],
    )

    if ns_settings["SliceSampler_settings"]:
        import ultranest.stepsampler as ustepsampler

        sampler.stepsampler = ustepsampler.SliceSampler(
            generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
            **ns_settings["SliceSampler_settings"],
        )

    t0 = time.time()
    ultranest_result = sampler.run(**ns_settings["Run_settings"])
    t1 = time.time()
    log.info("ULTRANEST RUNNING TIME: %f" % (t1 - t0))

    n_posterior_samples = ns_settings["n_posterior_samples"]
    if n_posterior_samples > ultranest_result["samples"].shape[0]:
        n_posterior_samples = ultranest_result["samples"].shape[0]
        log.warning(
            f"The chosen number of posterior samples exceeds the number of posterior"
            "samples computed by ultranest. Setting the number of resampled posterior"
            f"samples to {n_posterior_samples}"
        )

    resampled_posterior = resample_from_ns_posterior(
        ultranest_result["samples"],
        n_posterior_samples,
        ns_settings["posterior_resampling_seed"],
    )

    # Store run plots to ultranest output folder
    sampler.plot()

    return (parameters, resampled_posterior)


def perform_nested_sampling_grid_pdf_fit(
    ultranest_grid_fit,
    reduced_xgrids,
    flavour_indices,
    length_reduced_xgrids,
    ns_settings,
    output_path,
):
    """
    Performs a Nested Sampling fit using the grid.
    """

    # Save the resampled posterior to a csv file
    parameter_names, ultranest_grid_fit = ultranest_grid_fit
    df = pd.DataFrame(ultranest_grid_fit, columns=parameter_names)
    df.to_csv(str(output_path) + "/ns_result.csv")

    # Produce exportgrid files for each posterior sample
    write_exportgrid_from_fit_samples(
        samples=ultranest_grid_fit,
        n_posterior_samples=ns_settings["n_posterior_samples"],
        reduced_xgrids=reduced_xgrids,
        length_reduced_xgrids=length_reduced_xgrids,
        flavour_indices=flavour_indices,
        output_path=output_path,
    )

    log.info("Nested Sampling grid PDF fit completed!")


@dataclass(frozen=True)
class GridPdfFit:
    stacked_pdf_grid: jnp.array = None
    pdf_grid: jnp.array = None
    training_loss: jnp.array = None
    validation_loss: jnp.array = None
    xgrids: dict = None


def grid_pdf_mc_fit(
    _chi2_training_data_with_positivity,
    _chi2_validation_data_with_positivity,
    len_trval_data,
    xgrids,
    interpolate_grid,
    init_stacked_pdf_grid,
    optimizer_provider,
    early_stopper,
    max_epochs,
    batch_size=128,
    batch_seed=1,
    alpha=1e-7,
    lambda_positivity=1000,
):
    """This functions performs a Monte Carlo fit using the grid_pdf parametrisation.

    Parameters
    ----------
    _chi2_training_data_with_positivity (PjitFunction):
        Function that computes the chi2 of the training data.

    _chi2_validation_data_with_positivity (PjitFunction):
        Function that computes the chi2 of the validation data.

    _data_values (dataclass):
        Dataclass containing the training and validation data.

    xgrids (dict):
        Dictionary containing the xgrids for each flavour.

    interpolate_grid (PjitFunction):
        Function that performs the interpolation of the initial grid to the (14, 50) standard grid.

    init_stacked_pdf_grid (jnp.array):
        1D array containing the initial grid.

    optimizer_provider (optax._src.base.GradientTransformationExtraArgs):
        Optax optimizer.

    early_stopper (flax.training.early_stopping.EarlyStopping):
        Early stopping criteria.

    max_epochs (int):
        Number of maximum epochs.

    batch_size (int, optional):
        Size of batches during training. Defaults to 128.

    batch_seed (int, optional):
        Seed used to construct the batches. Defaults to 1.

    alpha (float, optional):
        Alpha parameter of the ELU positivity penalty term. Defaults to 1e-7.

    lambda_positivity (int, optional):
        Lagrange multiplier of the positivity penalty. Defaults to 1000.

    Returns
    -------
    GridPdfFit: The result of the fit with following attributes:
        stacked_pdf_grid: jnp.array
        pdf_grid: jnp.array
        training_loss: jnp.array
        validation_loss: jnp.array
        xgrids: dict
    """

    @jax.jit
    def loss_training(stacked_pdf_grid, batch_idx):
        pdf = interpolate_grid(stacked_pdf_grid)

        return _chi2_training_data_with_positivity(
            pdf, batch_idx, alpha, lambda_positivity
        )

    @jax.jit
    def loss_validation(stacked_pdf_grid):
        pdf = interpolate_grid(stacked_pdf_grid)

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

    opt_state = optimizer_provider.init(init_stacked_pdf_grid)
    stacked_pdf_grid = init_stacked_pdf_grid.copy()

    data_batch = data_batches(len_tr_idx, batch_size, batch_seed)
    batches = data_batch.data_batch_stream_index()
    num_batches = data_batch.num_batches
    batch_size = data_batch.batch_size

    for i in range(max_epochs):
        epoch_loss = 0
        epoch_val_loss = 0

        for _ in range(num_batches):
            batch = next(batches)

            stacked_pdf_grid, opt_state, loss_value = step(
                stacked_pdf_grid, opt_state, batch
            )

            epoch_loss += loss_training(stacked_pdf_grid, batch) / batch_size

        epoch_val_loss += loss_validation(stacked_pdf_grid) / len_val_idx
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

    return GridPdfFit(
        stacked_pdf_grid=stacked_pdf_grid,
        pdf_grid=interpolate_grid(stacked_pdf_grid),
        training_loss=loss,
        validation_loss=val_loss,
        xgrids=xgrids,
    )


"""
Collect over multiple replica fits.
"""
mc_replicas_gridpdf_fit = collect("grid_pdf_mc_fit", ("trval_replica_indices",))


def perform_mc_gridpdf_fit(
    mc_replicas_gridpdf_fit,
    reduced_xgrids,
    flavour_indices,
    length_reduced_xgrids,
    n_replicas,
    output_path,
):
    """
    Performs a Monte Carlo fit using the grid_pdf parametrisation.
    """

    samples = [
        mc_replicas_gridpdf_fit[i].stacked_pdf_grid
        for i in range(len(mc_replicas_gridpdf_fit))
    ]

    # Save the samples
    parameters = [
        f"{FK_FLAVOURS[i]}({j})" for i in flavour_indices for j in reduced_xgrids[i]
    ]

    df = pd.DataFrame(samples, columns=parameters)
    df.to_csv(str(output_path) + "/fit_mc_result.csv")

    # Produce exportgrid files for each posterior sample
    write_exportgrid_from_fit_samples(
        samples,
        n_posterior_samples=n_replicas,
        reduced_xgrids=reduced_xgrids,
        length_reduced_xgrids=length_reduced_xgrids,
        flavour_indices=flavour_indices,
        replica_index=None,
        single_replica_fit=False,
        output_path=output_path,
        replicas_folder="fit_replicas",
    )

    log.info("Monte Carlo fit completed!")


def perform_single_mc_gridpdf_fit(
    grid_pdf_mc_fit,
    replica_index,
    reduced_xgrids,
    flavour_indices,
    length_reduced_xgrids,
    output_path,
):
    """
    Performs a Monte Carlo fit using the grid_pdf parametrisation.
    """

    sample = grid_pdf_mc_fit.stacked_pdf_grid

    # Save the samples
    parameters = [
        f"{FK_FLAVOURS[i]}({j})" for i in flavour_indices for j in reduced_xgrids[i]
    ]

    df = pd.DataFrame([sample], columns=parameters, index=[replica_index])
    # if mc_result.csv already exists, append to it
    if os.path.isfile(str(output_path) + "/fit_mc_result.csv"):
        df.to_csv(str(output_path) + "/fit_mc_result.csv", mode="a", header=False)
    else:
        df.to_csv(str(output_path) + "/fit_mc_result.csv")

    # Produce exportgrid file
    write_exportgrid_from_fit_samples(
        [sample],
        n_posterior_samples=1,
        reduced_xgrids=reduced_xgrids,
        length_reduced_xgrids=length_reduced_xgrids,
        flavour_indices=flavour_indices,
        replica_index=replica_index,
        single_replica_fit=True,
        output_path=output_path,
        replicas_folder="fit_replicas",
    )

    # Save the training and validation loss
    df = pd.DataFrame(
        {
            "epochs": range(len(grid_pdf_mc_fit.training_loss)),
            "training_loss": grid_pdf_mc_fit.training_loss,
            "validation_loss": grid_pdf_mc_fit.validation_loss,
        }
    )
    df.to_csv(
        str(output_path) + f"/fit_replicas/replica_{replica_index}" + "/mc_loss.csv",
        index=False,
    )

    log.info("Monte Carlo fit completed!")
