"""
super_net.mc_utils.py

Module containing functions for the Monte Carlo fit.

Date: 17.1.2023
"""

import jax
import jax.numpy as jnp

from dataclasses import dataclass, asdict

from super_net.utils import training_validation_split
import pathlib
import pandas as pd
import os
import shutil

import logging

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPseudodata:
    pseudodata: jnp.array
    training_indices: jnp.array
    validation_indices: jnp.array
    trval_split: bool = False

    def to_dict(self):
        return asdict(self)


def mc_pseudodata(
    pseudodata_central_covmat_index,
    replica_index,
    trval_seed,
    shuffle_indices=True,
    mc_validation_fraction=0.2,
):
    """Produces Monte Carlo pseudodata for the replica with index replica_index.
    The pseudodata is returned with a set of training indices, which account for
    a fraction mc_validation_fraction of the data.
    """

    central_values = pseudodata_central_covmat_index.central_values
    covmat = pseudodata_central_covmat_index.covmat
    all_indices = pseudodata_central_covmat_index.central_values_idx

    # Generate pseudodata according to a multivariate Gaussian centred on
    # central_values and with covariance matrix covmat.
    key = jax.random.PRNGKey(replica_index)
    pseudodata = jax.random.multivariate_normal(
        key,
        central_values,
        covmat,
    )

    # Now select a subset of 1 - mc_validation_fraction indices to be the
    # training indices.
    if not mc_validation_fraction:
        return MCPseudodata(
            pseudodata=pseudodata,
            training_indices=all_indices,
            validation_indices=jnp.array([]),
            trval_split=False,
        )

    trval_obj = training_validation_split(
        all_indices,
        mc_validation_fraction,
        trval_seed,
        shuffle_indices,
    )

    training_indices = trval_obj.training
    validation_indices = trval_obj.validation

    return MCPseudodata(
        pseudodata=pseudodata,
        training_indices=training_indices,
        validation_indices=validation_indices,
        trval_split=True,
    )


def len_trval_data(mc_pseudodata):
    """Returns the number of training data points."""
    return len(mc_pseudodata.training_indices), len(mc_pseudodata.validation_indices)


def mc_postfit(fit_path, chi2_threshold=3.0, n_replica_target=100):
    """Postfit function for the Monte Carlo fit.
    It filters out the replicas with final training loss
    above the threshold and copy the remaining ones
    to the replicas directory.

    Parameters
    ----------
    fit_path : str
        Path to the fit directory.

    chi2_threshold : float, optional
        Threshold for the final training loss, by default 3.0.

    n_replica_target : int, optional
        Target number of replicas, by default 100.
    """

    # Convert fit_path to a pathlib.Path object
    fit_path = pathlib.Path(fit_path)

    # Check that the folder fit_replicas exists
    if not os.path.exists(fit_path / "fit_replicas"):
        log.error(
            f"The folder {fit_path}/fit_replicas does not exist.\n"
            f"Please run the Monte Carlo fit first."
        )
        raise FileNotFoundError(f"{fit_path}/fit_replicas does not exist")

    log.info("Running postfit for the Monte Carlo fit")
    log.debug(f"Threshold for final training loss: {chi2_threshold}")

    # Filter out only the directories
    replicas_path = fit_path / "fit_replicas"
    # Create the directory for the replicas if it does not exist
    # else delete it and create it again
    if not os.path.exists(fit_path / "replicas"):
        os.mkdir(fit_path / "replicas")
    else:
        shutil.rmtree(fit_path / "replicas")
        os.mkdir(fit_path / "replicas")

    replicas_list = sorted(list(replicas_path.iterdir()))

    # List of replicas to keep
    good_replicas = []

    # We will copy the replicas and order them starting with 0
    # and increasing the index for each good replica we find
    i = 0
    for replica in replicas_list:
        # Get last iteration from the mc_loss.csv file
        final_loss = pd.read_csv(replica / "mc_loss.csv").iloc[-1]["training_loss"]

        index = int(replica.name.split("_")[1])

        # Check if final loss is above the threshold
        if final_loss > chi2_threshold:
            log.warning(
                f"Discarding replica {index}, it has final training loss {final_loss:.3f}"
            )

            continue

        else:
            # We found a good replica
            good_replicas.append(index)
            # Increase replica index
            i += 1
            # Copy the replica to the fit directory
            shutil.copytree(replica, fit_path / f"replicas/replica_{i}")

        if i == n_replica_target:
            break

    log.info(f"{i} replicas pass postfit selection")

    if i < n_replica_target:
        log.critical(
            f"You asked for {n_replica_target} replicas, but only {i} replicas pass postfit selection.\n"
            f"You could consider increasing the threshold for the final training loss.",
        )

    else:
        log.info(
            f"Target number of replicas reached, {i} replicas pass postfit selection"
        )

    fit_df = pd.read_csv(fit_path / "fit_mc_result.csv", index_col=0)

    # Keep only the replicas with index in good_replicas
    postfit_df = fit_df.loc[good_replicas]

    # Save the postfit dataframe
    postfit_df.to_csv(fit_path / "mc_result.csv")

    log.info("Postfit completed")
