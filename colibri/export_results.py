"""
colibri.export_results.py

This module contains the functions to export the results of the fit.

"""

import os
from dataclasses import dataclass

import jax.numpy as jnp
import pandas as pd
from mpi4py import MPI

from colibri.lhapdf import write_exportgrid

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import logging

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BayesianFit:
    """
    Dataclass containing the results and specs of a Bayesian fit.

    Attributes
    ----------
    param_names: list
        List of the names of the parameters.
    resampled_posterior: jnp.array
        Array containing the resampled posterior samples.
    full_posterior_samples: jnp.array
        Array containing the full posterior samples.
    bayes_complexity: float
        The Bayesian complexity of the model.
    avg_chi2: float
        The average chi2 of the model.
    min_chi2: float
        The minimum chi2 of the model.
    logz: float
        The log evidence of the model.
    """

    param_names: list
    resampled_posterior: jnp.array
    full_posterior_samples: jnp.array
    bayes_complexity: float
    avg_chi2: float
    min_chi2: float
    logz: float


def export_bayes_results(
    bayes_fit,
    output_path,
    results_name,
):
    """
    Export the results of a Bayesian fit to a csv file.

    Parameters
    ----------
    bayes_fit: BayesianFit
        The results of the Bayesian fit.
    output_path: pathlib.PosixPath
        Path to the output folder.
    """

    # Write full sample to csv
    full_samples_df = pd.DataFrame(
        bayes_fit.full_posterior_samples, columns=bayes_fit.param_names
    )
    full_samples_df.to_csv(
        str(output_path) + "/full_posterior_sample.csv", float_format="%.5e"
    )

    # Save the resampled results
    df = pd.DataFrame(bayes_fit.resampled_posterior, columns=bayes_fit.param_names)
    df.to_csv(str(output_path) + f"/{results_name}.csv", float_format="%.5e")

    # Write the results to file
    with open(str(output_path) + "/bayes_metrics.csv", "w") as f:
        f.write(
            f"logz,min_chi2,avg_chi2,Cb\n{bayes_fit.logz},{bayes_fit.min_chi2},{bayes_fit.avg_chi2},{bayes_fit.bayes_complexity}\n"
        )


def write_replicas(bayes_fit, output_path, pdf_model):
    if rank == 0:
        # create replicas folder if it does not exist
        replicas_path = str(output_path) + "/replicas"
        if not os.path.exists(replicas_path):
            os.mkdir(replicas_path)

    # Synchronize to ensure all processes are ready to write replicas
    comm.Barrier()

    n_posterior_samples = bayes_fit.resampled_posterior.shape[0]

    # Distribute indices among processes using scatter
    indices_per_process = list(range(rank, n_posterior_samples, size))

    # Finish by writing the replicas to export grids, ready for evolution
    for i in indices_per_process:
        log.info(f"Writing exportgrid for replica {i+1}")
        write_exportgrid(
            jnp.array(bayes_fit.resampled_posterior[i, :]),
            pdf_model,
            i + 1,
            output_path,
        )
