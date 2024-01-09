from validphys.convolution import FK_FLAVOURS

import ultranest
import jax
import jax.numpy as jnp
import optax
from super_net.data_batch import data_batches

from dataclasses import dataclass
import time
import logging
from reportengine import collect

log = logging.getLogger(__name__)


def make_bayesian_pdf_grid_fit(
    _chi2_with_positivity,
    grid_pdf_model_prior,
    interpolate_grid,
    reduced_xgrids,
    flavour_indices,
    min_num_live_points=400,
    min_ess=40,
    log_dir="ultranest_logs",
    resume=True,
    vectorized=False,
    slice_sampler=False,
    slice_steps=100,
    ndraw_max=500,
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
        log_dir=log_dir,
        resume=resume,
        vectorized=vectorized,
        ndraw_max=ndraw_max,
    )

    if slice_sampler:
        import ultranest.stepsampler as ustepsampler

        sampler.stepsampler = ustepsampler.SliceSampler(
            nsteps=slice_steps,
            generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
        )

    t0 = time.time()
    sampler.run(
        min_num_live_points=min_num_live_points,
        min_ess=min_ess,
    )
    t1 = time.time()
    log.info("ULTRANEST RUNNING TIME: %f" % (t1 - t0))

    # Store run plots to ultranest output folder
    sampler.plot()


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
    """ """

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
):
    """
    Performs a Monte Carlo fit using the grid_pdf parametrisation.
    """

    log.info("Monte Carlo fit completed!")
