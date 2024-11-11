"""
colibri.model_average.py

Module containing functions for performing Bayesian model average.
"""

import os
import logging
import dill

import numpy as np
import pandas as pd

from colibri.utils import resample_from_ns_posterior
from colibri.export_results import write_exportgrid
from colibri.constants import LHAPDF_XGRID, EXPORT_LABELS

from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

log = logging.getLogger()


def uniform_ns_fit_resampler(
    fit_path,
    resampled_fit_path,
    resampled_fit_name,
    nreplicas,
    resampling_seed,
    parametrisation_scale,
):
    """ """

    # check whether fit path exists
    log.info(f"Loading pdf model from {fit_path}")
    # load pdf_model from fit using dill
    with open(fit_path / "pdf_model.pkl", "rb") as file:
        pdf_model = dill.load(file)

    # Check that the .txt file with posterior samples exists
    if not os.path.exists(fit_path / "ultranest_logs/chains/equal_weighted_post.txt"):
        raise FileNotFoundError(
            f"{fit_path}/ultranest_logs/chains/equal_weighted_post.txt does not exist;"
            "please run the bayesian fit first."
        )

    equal_weight_post_path = fit_path / "ultranest_logs/chains/equal_weighted_post.txt"

    samples = pd.read_csv(equal_weight_post_path, sep="\s+", dtype=float).values

    if nreplicas > samples.shape[0]:
        nreplicas = samples.shape[0]
        log.warning(
            f"The chosen number of posterior samples exceeds the number of posterior"
            "samples computed by ultranest. Setting the number of resampled posterior"
            f"samples to {nreplicas}"
        )

    resampled_posterior = resample_from_ns_posterior(
        samples,
        nreplicas,
        resampling_seed,
    )

    if rank == 0:
        # copy old fit to resampled fit
        os.system(f"cp -r {fit_path} {resampled_fit_path}")

        # remove old replicas from resampled fit
        os.system(f"rm -r {resampled_fit_path}/replicas/*")

        # overwrite old ns_result.csv with resampled posterior
        parameters = pdf_model.param_names
        df = pd.DataFrame(resampled_posterior, columns=parameters)
        df.to_csv(str(resampled_fit_path) + "/ns_result.csv", float_format="%.5e")

    comm.Barrier()

    # Distribute indices among processes using scatter
    indices_per_process = list(range(rank, nreplicas, size))

    new_rep_path = resampled_fit_path / "replicas"

    if not os.path.exists(new_rep_path):
        os.mkdir(new_rep_path)

    # Finish by writing the replicas to export grids, ready for evolution
    for i in indices_per_process:

        # Get the PDF grid in the evolution basis
        parameters = resampled_posterior[i]
        lhapdf_interpolator = pdf_model.grid_values_func(LHAPDF_XGRID)
        grid_for_writing = np.array(lhapdf_interpolator(parameters))

        replica_index = i + 1

        replica_index_path = new_rep_path / f"replica_{replica_index}"
        if not os.path.exists(replica_index_path):
            os.mkdir(replica_index_path)

        grid_name = replica_index_path / resampled_fit_name

        log.info(f"Writing exportgrid for replica {replica_index}")
        write_exportgrid(
            grid_for_writing=grid_for_writing,
            grid_name=grid_name,
            replica_index=replica_index,
            Q=parametrisation_scale,
            xgrid=LHAPDF_XGRID,
            export_labels=EXPORT_LABELS,
        )
        log.info(f"Writing exportgrid for replica {replica_index} on rank {rank}")

    # Synchronize to ensure all processes have finished
    comm.Barrier()
    if rank == 0:
        log.info(f"Resampling completed. Resampled fit stored in {resampled_fit_path}")


def selected_fits(fits, delta_logz=26.6):
    """
    Select a subset, A, of the models the set of models is defined as

    A = {M_k : log(Zk) >= log(Zmax) - delta_logz}

    where delta_logz can be chosen, for instance, to be a value from Jeffreys scales.

    Parameters
    ----------
    fits: list
        list of ColibriFitSpecs

    delta_logz: float
        Model selection tolerance constant

    Returns
    -------
    list
        a list of ColibriFitSpecs with len <= of the original
    """
    log_z_max = np.max([fit.bayesian_metrics["logz"] for fit in fits])
    selected_models = [
        fit for fit in fits if (fit.bayesian_metrics["logz"] >= log_z_max - delta_logz)
    ]
    return selected_models


def selected_fits_with_weights(selected_fits):
    """
    If ∆ is the quantity of interest, or parameters of interest, then when performing model
    average we can compute its posterior distribution given data D as

    p(∆|D) = sum_k p(D|∆,M_k) p(M_k|D)

    - M_k are the different models over which we average.
    - p(M_k|D) is the posterior model probability

    Assuming that the prior probability is the same for each model we get

    p(M_k|D) = p(D|M_k) / (sum_l p(D|M_l))

    hence multiplying by 1 = exp(-log(Z_avg))/exp(-log(Z_avg)) we get

    p(D|M_k) = exp(log(Z_k) - log(Z_avg)) / (1 + sum_{l != 1} exp(log(Z_l)-log(Z_avg)))

    """
    logz_values = np.array([fit.bayesian_metrics["logz"] for fit in selected_fits])
    mean_logZ = np.mean(logz_values)

    for fit, logz in zip(selected_fits, logz_values):
        p_k = np.exp(logz - mean_logZ) / (np.sum(np.exp(logz_values - mean_logZ)))

        fit.bayesian_metrics["bayesian_weight"] = p_k
    import IPython

    IPython.embed()
    return selected_fits


def model_combination(selected_fits_with_weights, n_samples):
    """ """
    # get fraction of number of replicas for each fit
    for fit in selected_fits_with_weights:
        n_frac_samples = int(fit.bayesian_metrics["bayesian_weight"] * n_samples)
        fit.bayesian_metrics["n_frac_samples"] = n_frac_samples
