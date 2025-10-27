"""
colibri.time_likelihood.py

This module times the execution time of the vectorised likelihood.

"""

import logging
import time
import csv

import jax
import numpy as np

log = logging.getLogger(__name__)


def time_log_likelihood(log_likelihood, bayesian_prior, pdf_model, output_path):
    """
    Time the vectorized log likelihood across different batch sizes.
    Replicates Figure 1 from the paper.

    Parameters
    ----------
    log_likelihood : callable
        The log likelihood function that takes parameter vector(s)
    bayesian_prior : dict
        Dictionary containing prior_transform and other prior functions
    pdf_model : pdf_model.PDFModel
        The PDF model to fit
    output_path : pathlib.PosixPath
        Path to the output folder where log_likelihood_times.csv will be saved
    """
    # Extract prior_transform from bayesian_prior
    prior_transform = bayesian_prior["prior_transform"]

    # Get number of parameters from pdf_model
    n_params = len(pdf_model.param_names)

    # Create vectorized version
    log_likelihood_vec = jax.vmap(log_likelihood, in_axes=(0,), out_axes=0)

    # Batch sizes to test
    sizes = [1, 10, 100, 1000, 5000, 10000, 20000, 50000, 100000]

    # Pre-generate samples for each size
    log.info("Generating samples...")
    samples_list = []
    rng_key = jax.random.PRNGKey(0)

    for i, size in enumerate(sizes):
        # Generate uniform samples in [0,1]
        key = jax.random.fold_in(rng_key, i)
        cube = jax.random.uniform(key, shape=(size, n_params))
        # Transform through prior
        samples = prior_transform(cube)
        samples_list.append(samples)

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
