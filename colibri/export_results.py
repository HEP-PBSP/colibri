"""
colibri.export_results.py

This module contains the functions to export the results of the fit.

"""

import os
from dataclasses import dataclass
import yaml
import numpy as np

import jax.numpy as jnp
import pandas as pd
from mpi4py import MPI

from colibri.constants import LHAPDF_XGRID, evolution_to_flavour_matrix, EXPORT_LABELS

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
    results_name: str
        Name of the results file.
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


def write_exportgrid(
    grid_for_writing,
    grid_name,
    replica_index,
    Q=1.65,
    xgrid=LHAPDF_XGRID,
    export_labels=EXPORT_LABELS,
):
    """
    Writes an exportgrid file to the output path.
    The exportgrids are written in the format required by EKO, but are not yet
    evolved.

    Parameters
    ----------
    grid_for_writing: jnp.array
        An array of shape (14,Nx) containing the PDF values in the evolution basis.

    grid_name: str
        The name of the grid to write.

    replica_index: int
        The replica number which will be written.
    """

    grid_for_writing = evolution_to_flavour_matrix @ grid_for_writing
    grid_for_writing = grid_for_writing.T.tolist()

    # Prepare a dictionary for the exportgrid
    export_grid = {}

    # Set the initial Q2 value, which should always be (1.65)**2.
    export_grid["q20"] = (Q) ** 2
    export_grid["xgrid"] = xgrid
    export_grid["replica"] = int(replica_index)
    export_grid["labels"] = export_labels

    export_grid["pdfgrid"] = grid_for_writing

    with open(f"{grid_name}.exportgrid", "w") as outfile:
        yaml.dump(export_grid, outfile)


def write_replicas(
    bayes_fit,
    output_path,
    pdf_model,
    monte_carlo=False,
    Q=1.65,
    xgrid=LHAPDF_XGRID,
    export_labels=EXPORT_LABELS,
):
    """
    Write the replicas of the Bayesian fit to export grids.

    Parameters
    ----------
    bayes_fit: BayesianFit
        The results of the Bayesian fit.
    output_path: pathlib.PosixPath
        Path to the output folder.
    pdf_model: pdf_model.PDFModel
        The PDF model used in the fit.
    monte_carlo: bool
        Whether the fit is a Monte Carlo fit. If True, the exportgrids are written
        to a folder called "fit_replicas" in the output_path.
    """
    if rank == 0:
        # create replicas folder if it does not exist
        if monte_carlo:
            replicas_path = str(output_path) + "/fit_replicas"
        else:
            replicas_path = str(output_path) + "/replicas"

        if not os.path.exists(replicas_path):
            os.mkdir(replicas_path)

    # Synchronize to ensure all processes are ready to write replicas
    comm.Barrier()

    n_posterior_samples = bayes_fit.resampled_posterior.shape[0]

    # Distribute indices among processes using scatter
    indices_per_process = list(range(rank, n_posterior_samples, size))

    fit_name = str(output_path).split("/")[-1]

    # Create the exportgrid
    lhapdf_interpolator = pdf_model.grid_values_func(LHAPDF_XGRID)

    # Finish by writing the replicas to export grids, ready for evolution
    for i in indices_per_process:

        # Get the PDF grid in the evolution basis
        parameters = jnp.array(bayes_fit.resampled_posterior[i, :])
        grid_for_writing = np.array(lhapdf_interpolator(parameters))

        replica_index = i + 1
        rep_path = replicas_path + f"/replica_{replica_index}"
        if not os.path.exists(rep_path):
            os.mkdir(rep_path)
        grid_name = rep_path + "/" + fit_name

        log.info(f"Writing exportgrid for replica {replica_index}")
        write_exportgrid(
            grid_for_writing=grid_for_writing,
            grid_name=grid_name,
            replica_index=replica_index,
            Q=Q,
            xgrid=xgrid,
            export_labels=export_labels,
        )
