"""
colibri.time_likelihood.py

This module times the execution time of the vectorised likelihood.

"""

import logging
import time
import csv

import jax
import jax.numpy as jnp
import numpy as np
from colibri.param_initialisation import pdf_initial_parameters

log = logging.getLogger(__name__)

DEFAULT_BATCH_SAMPLE_SIZES = [1, 10, 100, 1000, 5000, 10000, 20000, 50000, 100000]
"""
Batch sizes timed when ``batch_sample_sizes`` is not given. Note that the cost of
``time_log_likelihood`` is driven by ``max(batch_sample_sizes)``, since that many
parameter vectors have to be generated up front.
"""


def time_log_likelihood(
    log_likelihood,
    param_initialiser_settings,
    forward_map,
    output_path,
    batch_sample_sizes=None,
):
    """
    Time the vectorized log likelihood across different batch sizes.

    Parameters
    ----------
    log_likelihood : callable
        The log likelihood function that takes parameter vector(s)
    param_initialiser_settings : dict
        Settings for parameter initialization
    forward_map : forward_map.ForwardMap
        The forward map whose .param_names are used for parameter initialization.
    batch_sample_sizes : sequence of int, optional
        Batch sizes (number of parameter vectors per batch) to time
    output_path : pathlib.PosixPath
        Path to the output folder where log_likelihood_times.csv will be saved
    """

    # Create vectorized version
    log_likelihood_vec = jax.jit(jax.vmap(log_likelihood, in_axes=(0,), out_axes=0))

    # Batch sizes to test - use provided or default
    if batch_sample_sizes is None:
        sizes = list(DEFAULT_BATCH_SAMPLE_SIZES)
        log.info("Using default batch sample sizes")
    else:
        sizes = batch_sample_sizes
        log.info(f"Using custom batch sample sizes: {sizes}")

    # Set up CSV file path
    save_path = output_path / "log_likelihood_times.csv"

    # Initialize CSV file with headers
    with open(save_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["batch_size", "avg_time_seconds", "relative_time"])

    log.info(f"Results will be saved incrementally to {save_path}")

    # Pre-generate samples for the largest size only
    log.info("Generating samples for log likelihood timing...")
    max_size = max(sizes)

    all_samples = []
    for replica_idx in range(max_size):
        params = pdf_initial_parameters(
            forward_map, param_initialiser_settings, replica_idx
        )
        all_samples.append(np.asarray(params))

    # Stack all samples into one large batch.
    # NOTE: the stacking is done on the host and transferred to the device once.
    # jnp.stack on a list of max_size device arrays instead builds a single XLA op
    # with max_size operands, which is ~9x slower here at the default max_size.
    all_samples_batch = jnp.asarray(np.stack(all_samples))

    # Create subsets for each size
    samples_list = []
    for size in sizes:
        samples_list.append(all_samples_batch[:size])

    # Now time each batch size
    log.info("Timing different batch sizes...")
    times = []
    successful_sizes = []
    n_repeats = 100  # Number of times to repeat for averaging

    for i, size in enumerate(sizes):
        # Warm-up: compile the function by calling it a couple times
        log.info("Warming up (JIT compilation)...")
        try:
            _ = log_likelihood_vec(samples_list[i])
            _ = log_likelihood_vec(samples_list[i])
            jax.block_until_ready(_)  # Wait for compilation to finish
        except Exception as e:
            log.error(f"Warm-up failed: {e}")
            raise
        try:
            log.info(f"Timing batch size: {size}")
            t0 = time.perf_counter()
            for _ in range(n_repeats):
                result = log_likelihood_vec(samples_list[i])
                jax.block_until_ready(result)  # ensure this iteration finished
            t1 = time.perf_counter()
            avg_time = (t1 - t0) / n_repeats
            times.append(avg_time)
            successful_sizes.append(size)

            # Compute relative time (relative to first successful timing)
            relative_time = avg_time / times[0] if times else 1.0

            # Append result to CSV immediately
            with open(save_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([size, avg_time, relative_time])

            log.info(
                f"Size: {size:6d}, Time: {avg_time:.6f} s, Relative: {relative_time:.4f}x"
            )

        except Exception as e:
            log.error(f"Error at batch size {size}: {e}")
            log.warning(
                f"Stopping timing. Results for batch sizes up to {successful_sizes[-1] if successful_sizes else 'none'} have been saved."
            )
            break

    if successful_sizes:
        log.info(f"Timing completed for {len(successful_sizes)} batch sizes")
        log.info(f"Final results saved to {save_path}")
    else:
        log.error("No batch sizes were successfully timed")

    return successful_sizes, times
