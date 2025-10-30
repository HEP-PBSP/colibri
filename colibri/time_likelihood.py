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


def time_log_likelihood(
    log_likelihood, param_initialiser_settings, pdf_model, output_path
):
    """
    Time the vectorized log likelihood across different batch sizes.

    Parameters
    ----------
    log_likelihood : callable
        The log likelihood function that takes parameter vector(s)
    param_initialiser_settings : dict
        Settings for parameter initialization
    pdf_model : pdf_model.PDFModel
        The PDF model to fit
    output_path : pathlib.PosixPath
        Path to the output folder where log_likelihood_times.csv will be saved
    """

    # Create vectorized version
    log_likelihood_vec = jax.vmap(log_likelihood, in_axes=(0,), out_axes=0)

    # Batch sizes to test
    sizes = [1, 10, 100, 1000, 5000, 10000, 20000, 50000, 100000]

    # Pre-generate samples for the largest size only
    log.info("Generating samples for log likelihood timing...")
    max_size = max(sizes)

    all_samples = []
    for replica_idx in range(max_size):
        params = pdf_initial_parameters(
            pdf_model, param_initialiser_settings, replica_idx
        )
        all_samples.append(params)

    # Stack all samples into one large batch
    all_samples_batch = jnp.stack(all_samples)

    # Create subsets for each size
    samples_list = []
    for size in sizes:
        samples_list.append(all_samples_batch[:size])

    # Warm-up: compile the function by calling it a couple times
    log.info("Warming up (JIT compilation)...")
    _ = log_likelihood_vec(samples_list[0])
    _ = log_likelihood_vec(samples_list[1])
    jax.block_until_ready(_)  # Wait for compilation to finish

    # Now time each batch size
    log.info("Timing different batch sizes...")
    times = []
    n_repeats = 100  # Number of times to repeat for averaging

    for i, size in enumerate(sizes):
        t0 = time.time()
        for _ in range(n_repeats):
            result = log_likelihood_vec(samples_list[i])
            jax.block_until_ready(result)  # Critical: wait for GPU to finish
        t1 = time.time()

        avg_time = (t1 - t0) / n_repeats
        times.append(avg_time)
        log.info(f"Size: {size:6d}, Time: {avg_time:.6f} s")

    # Save to CSV in the output_path
    save_path = output_path / "log_likelihood_times.csv"
    with open(save_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["batch_size", "avg_time_seconds", "relative_time"])

        # Compute relative times
        relative_times = np.array(times) / times[0]

        for size, time_val, rel_time in zip(sizes, times, relative_times):
            writer.writerow([size, time_val, rel_time])

    log.info(f"Results saved to {save_path}")

    return sizes, times
