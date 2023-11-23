"""
wmin.wmin_fit.py

Module containing functions used to perform a weight minimisation PDF fit.

Author: Mark N. Costantini
Date: 11.11.2023
"""

from collections.abc import Mapping

import logging
from dataclasses import dataclass

import lhapdf

import jax
import jax.numpy as jnp
import optax

import ultranest
import time

from reportengine import collect

from super_net.data_batch import data_batches
from wmin.wmin_model import WeightMinimizationGrid
from wmin.wmin_utils import resample_from_wmin_posterior

from wmin.wmin_lhapdf import lhapdf_from_collected_weights, lhapdf_from_collected_ns_weights

from validphys.loader import Loader
from validphys.lhio import generate_replica0

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WeightMinimizationFit(WeightMinimizationGrid):
    optimised_wmin_weights: jnp.array = None
    training_loss: jnp.array = None
    validation_loss: jnp.array = None


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

def wmin_fit_name(wminpdfset, set_name=None):
    if set_name:
        return set_name
    return str(wminpdfset)

def perform_nested_sampling_wmin_fit(
    wminpdfset,
    weight_minimization_ultranest,
    n_wmin_posterior_samples,
    wmin_fit_name,
    lhapdf_path,
):
    """
    Performs a Nested Sampling fit using the weight-minimisation parametrisation.
    """

    # Produce the LHAPDF grid
    lhapdf_from_collected_ns_weights(
        wminpdfset,
        weight_minimization_ultranest,
        n_wmin_posterior_samples,
        folder=lhapdf_path,
        set_name=wmin_fit_name,
    )

    # Produce the central replica
    l = Loader()
    pdf = l.check_pdf(wmin_fit_name)
    generate_replica0(pdf)

    log.info("Nested Sampling weight minimization fit completed!")

def perform_monte_carlo_wmin_fit(
    wminpdfset,
    mc_replicas_weight_minimization_fit,
    n_replicas,
    wmin_fit_name,
    output_path,
    lhapdf_path,
):
    """
    Performs a Monte Carlo fit using the weight-minimisation parametrisation.
    """

    # Produce the LHAPDF grid
    lhapdf_from_collected_weights(
        wminpdfset,
        mc_replicas_weight_minimization_fit,
        n_replicas,
        folder=lhapdf_path,
        set_name=wmin_fit_name, 
        output_path=output_path,
    )

    # Produce the central replica
    l = Loader()
    pdf = l.check_pdf(wmin_fit_name)
    generate_replica0(pdf)

    log.info("Monte Carlo weight minimization fit completed!")

@dataclass(frozen=True)
class UltranestWeightMinimizationFit(WeightMinimizationFit):
    ultranest_result: Mapping = None


def weight_minimization_ultranest(
    make_chi2,
    weight_minimization_grid,
    weight_minimization_prior,
    n_replicas_wmin,
    min_num_live_points,
    min_ess,
    n_wmin_posterior_samples=1000,
    wmin_posterior_resampling_seed=123456,
):
    """
    TODO
    note: not including positivity for the time being
    The function being used for the chi2 is not the optimised one
    in wmin.wmin_loss_functions. Performances are similar with
    the generic one.
    """

    parameters = [f"w{i+1}" for i in range(n_replicas_wmin - 1)]

    @jax.jit
    def log_likelihood(weights):
        """
        TODO
        """
        wmin_weights = jnp.concatenate((jnp.array([1.0]), weights))
        pdf = jnp.einsum(
            "i,ijk", wmin_weights, weight_minimization_grid.wmin_INPUT_GRID
        )
        return -0.5 * make_chi2(pdf)

    sampler = ultranest.ReactiveNestedSampler(
        parameters,
        log_likelihood,
        weight_minimization_prior,
    )

    t0 = time.time()
    ultranest_result = sampler.run(
        min_num_live_points=min_num_live_points,
        min_ess=min_ess,
    )
    t1 = time.time()
    log.info("ULTRANEST RUNNING TIME: %f" % (t1 - t0))

    if n_wmin_posterior_samples > ultranest_result["samples"].shape[0]:
        n_wmin_posterior_samples = ultranest_result["samples"].shape[0] - int(
            0.1 * ultranest_result["samples"].shape[0]
        )
        log.warning(
            f"The chosen number of posterior samples exceeds the number of posterior"
            "samples computed by ultranest. Setting the number of resampled posterior"
            f"samples to {n_wmin_posterior_samples}"
        )

    resampled_posterior = resample_from_wmin_posterior(
        ultranest_result["samples"],
        n_wmin_posterior_samples,
        wmin_posterior_resampling_seed,
    )

    return UltranestWeightMinimizationFit(
        **weight_minimization_grid.to_dict(),
        optimised_wmin_weights=resampled_posterior,
        ultranest_result=ultranest_result,
    )
