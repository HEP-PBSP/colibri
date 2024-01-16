from validphys.convolution import FK_FLAVOURS

from grid_pdf.grid_pdf_lhapdf import lhapdf_grid_pdf_from_samples
from super_net.utils import resample_from_ns_posterior

from validphys.loader import Loader
from validphys.lhio import generate_replica0

import ultranest
import jax
import jax.numpy as jnp
import pandas as pd
import optax
from super_net.data_batch import data_batches

from dataclasses import dataclass
import time
import logging
from reportengine import collect

log = logging.getLogger(__name__)


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

    parameters = [
        f"{FK_FLAVOURS[i]}({j})" for i in flavour_indices for j in reduced_xgrids[i]
    ]

    sampler = ultranest.ReactiveNestedSampler(
        parameters,
        log_likelihood,
        grid_pdf_model_prior,
        log_dir=ns_settings["log_dir"],
        resume=ns_settings["resume"],
        vectorized=ns_settings["vectorized"],
        ndraw_max=ns_settings["ndraw_max"],
    )

    if ns_settings["slice_sampler"]:
        import ultranest.stepsampler as ustepsampler

        sampler.stepsampler = ustepsampler.SliceSampler(
            nsteps=ns_settings["slice_steps"],
            generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
        )

    t0 = time.time()
    ultranest_result = sampler.run(
        min_num_live_points=ns_settings["min_num_live_points"],
        min_ess=ns_settings["min_ess"],
    )
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
    lhapdf_path,
    output_path,
    theoryid,
):
    """
    Performs a Nested Sampling fit using the grid.
    """

    # Save the resampled posterior as a pandas df
    parameter_names, ultranest_grid_fit = ultranest_grid_fit
    df = pd.DataFrame(ultranest_grid_fit, columns=parameter_names)
    df.to_csv(str(output_path) + "/ns_result.csv")

    # Produce the LHAPDF grid
    lhapdf_grid_pdf_from_samples(
        ultranest_grid_fit,
        reduced_xgrids,
        flavour_indices,
        length_reduced_xgrids,
        ns_settings["n_posterior_samples"],
        theoryid,
        folder=lhapdf_path,
        output_path=output_path,
    )

    # Produce the central replica
    l = Loader()
    pdf = l.check_pdf(str(output_path).split("/")[-1])
    generate_replica0(pdf)

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
    _data_values,
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

    loss = []
    val_loss = []

    opt_state = optimizer_provider.init(init_stacked_pdf_grid)
    stacked_pdf_grid = init_stacked_pdf_grid.copy()

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

            stacked_pdf_grid, opt_state, loss_value = step(
                stacked_pdf_grid, opt_state, batch
            )

            epoch_loss += loss_training(stacked_pdf_grid, batch) / batch_size

        epoch_val_loss += (
            loss_validation(stacked_pdf_grid)
            / _data_values.validation_data.n_validation_points
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
    theoryid,
    lhapdf_path,
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
    df.to_csv(str(output_path) + "/mc_result.csv")

    # Produce the LHAPDF grid
    lhapdf_grid_pdf_from_samples(
        samples,
        reduced_xgrids,
        flavour_indices,
        length_reduced_xgrids,
        n_replicas,
        theoryid,
        folder=lhapdf_path,
        output_path=output_path,
    )

    # Produce the central replica
    l = Loader()
    pdf = l.check_pdf(str(output_path).split("/")[-1])
    generate_replica0(pdf)

    log.info("Monte Carlo fit completed!")
