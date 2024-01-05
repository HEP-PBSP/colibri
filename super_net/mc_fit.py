"""
super_net.mc_fit.py

This module provides the functions necessary for performing a MC fit with super_net.

Author: James Moore
Date: 5.1.2023
"""

import jax
import optax
import logging

import ultranest
import ultranest.stepsampler as ustepsampler
import time

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

def ns_fit(
    _chi2_with_positivity,
    pdf_model,
    output_path,
    min_num_live_points,
    min_ess,
    n_posterior_samples=1000,
    posterior_resampling_seed=123456,
    vectorized=False,
    ndraw_max=1000,
    slice_sampler=False,
    slice_steps=100,
    resume=True,
):
    """
    TODO
    """

    parameters = pdf_model.param_names
    log_dir = output_path / "ultranest"

    @jax.jit
    def log_likelihood(params):
        """
        TODO
        """
        pdf = pdf_model.grid_values(params)
        return -0.5 * _chi2_with_positivity(pdf)

    """TODO
    @jax.jit
    def log_likelihood_vectorized(weights):
        wmin_weights = jnp.c_[jnp.ones(weights.shape[0]), weights]
        pdf = jnp.einsum(
            "ri,ijk -> rjk", wmin_weights, weight_minimization_grid.wmin_INPUT_GRID
        )

        return -0.5 * _chi2_with_positivity(pdf)

    if vectorized:
        sampler = ultranest.ReactiveNestedSampler(
            parameters,
            log_likelihood_vectorized,
            weight_minimization_prior,
            vectorized=True,
            ndraw_max=ndraw_max,
            log_dir=log_dir,
            resume=resume,
        )

    else:
    """
    sampler = ultranest.ReactiveNestedSampler(
        parameters,
        log_likelihood,
        pdf_model.bayesian_prior,
        log_dir=log_dir,
        resume=resume,
    )

    if slice_sampler:
        sampler.stepsampler = ustepsampler.SliceSampler(
            nsteps=slice_steps,
            generate_direction=ustepsampler.generate_mixture_random_direction,
        )

    t0 = time.time()
    ultranest_result = sampler.run(
        min_num_live_points=min_num_live_points,
        min_ess=min_ess,
    )
    t1 = time.time()
    log.info("ULTRANEST RUNNING TIME: %f" % (t1 - t0))

    if n_posterior_samples > ultranest_result["samples"].shape[0]:
        n_posterior_samples = ultranest_result["samples"].shape[0] - int(
            0.1 * ultranest_result["samples"].shape[0]
        )
        log.warning(
            f"The chosen number of posterior samples exceeds the number of posterior"
            "samples computed by ultranest. Setting the number of resampled posterior"
            f"samples to {n_posterior_samples}"
        )

    """
    resampled_posterior = resample_from_wmin_posterior(
        ultranest_result["samples"],
        n_wmin_posterior_samples,
        wmin_posterior_resampling_seed,
    )

    # Store run plots to ultranest output folder
    sampler.plot()

    return UltranestWeightMinimizationFit(
        **weight_minimization_grid.to_dict(),
        optimised_wmin_weights=resampled_posterior,
        ultranest_result=ultranest_result,
    )
    """
    return 0
