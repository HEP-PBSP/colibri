from functools import lru_cache
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import pandas as pd

from validphys.core import FitSpec

from colibri.constants import XGRID
from colibri.utils import get_pdf_model
from colibri.ntk.ntkutils import (
    get_replica_idx_list,
    get_parameters_all_epochs,
    compute_ntk,
    NTKStats,
)

log = logging.getLogger(__name__)


# Bounded to the current epoch only. Each entry holds the full per-epoch NTK ensemble;
# the old unbounded cache retained one per epoch -> unbounded growth over the epoch
# trajectory (OOM). See eigenvectors_ensemble_at_epoch for the same fix/rationale.
@lru_cache(maxsize=1)
def ntk_ensemble_at_epoch(
    fit: FitSpec,
    replicas_path: Path,
    epoch: int,
    replica_index_list: Optional[tuple] = None,
    max_workers: Optional[int] = None,
    kwargs: frozenset = frozenset({}),
):
    """
    Compute the NTK ensemble at a specific epoch across multiple replicas.

    Parameters
      ----------
      fit : FitSpec
          The fit object containing fit information
      replicas_path : Path
          Path to the replicas directory
      epoch : int
          The epoch at which to compute eigenvectors
      replica_index_list : tuple, optional
          Specific replica indices to compute. If None, computes all. Mainly for
          testing.
      max_workers : int, optional
          Maximum number of parallel workers. If None, defaults to min(10,
          n_replicas).
      kwargs : dict, optional
          Additional kwargs to pass to compute_eigenvectors_at_epoch_for_replica
    """

    # Determine which replicas to compute
    if replica_index_list is None:
        replica_index_list = get_replica_idx_list(replicas_path)
    n_replicas = len(replica_index_list)

    if max_workers is None:
        max_workers = min(10, n_replicas)
    log.info(f"Using max_workers={max_workers} for parallel computation")

    ntk_data = {}
    shape = None

    def compute_ntk_at_epoch_for_replica(replica_idx):
        pdf_model = get_pdf_model(fit.name, replica_idx=replica_idx)
        param_files = get_parameters_all_epochs(replicas_path, replica_idx)
        params_at_epoch = param_files.get(epoch, None)
        if params_at_epoch is None:
            raise ValueError(
                f"Epoch {epoch} not found for replica {replica_idx} in fit {fit.name}"
            )

        params = jnp.load(params_at_epoch)["params"]
        ntk, shape = compute_ntk(pdf_model, params, **dict(kwargs))
        return (ntk, shape)

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_replica = {
            executor.submit(compute_ntk_at_epoch_for_replica, replica_idx): replica_idx
            for replica_idx in replica_index_list
        }

        futures_iter = as_completed(future_to_replica)
        for future in tqdm(futures_iter, total=n_replicas, desc="Computing NTKs"):
            replica_idx = future_to_replica[future]
            try:
                result = future.result()
                if result is None:
                    log.warning(f"Replica {replica_idx} failed")
                ntk_data[replica_idx], _shape = result
                with lock:
                    if shape is None:
                        shape = _shape
            except Exception as e:
                log.warning(f"Error computing the NTK for replica {replica_idx}: {e}")

        if not ntk_data:
            raise ValueError("No NTK data computed successfully")

        # Convert the final dict into a ndarray
        ntk_data_array = np.array(
            [ntk_data[replica_idx] for replica_idx in sorted(ntk_data.keys())]
        )  # Shape: (n_repl, nflav*ngrid, nflav*ngrid)

        return {
            "ntk_data": ntk_data_array,
            "epoch": epoch,
            "shape": shape,
            "replica_indices": replica_index_list,
        }


def ntk_at_epoch(ntk_ensemble_at_epoch, flavours):
    """
    Returns dataframe with the NTK at a specific epoch.
    """
    ntk_ensemble = ntk_ensemble_at_epoch["ntk_data"]
    shape = ntk_ensemble_at_epoch["shape"]
    n_replicas = len(ntk_ensemble_at_epoch["replica_indices"])

    # Index follows NTK_ORDERING: for each flavour, all x-points in sequence
    index = pd.MultiIndex.from_tuples(
        [(fl, i + 1) for fl in flavours for i in range(len(XGRID))],
        names=["flavour", "x"],
    )
    if len(flavours) > shape[0]:
        raise ValueError(f"flavour_indices {flavours} out of range [0, {shape[0]})")

    dfs = [
        pd.DataFrame(data=ntk_ensemble[k], index=index, columns=index)
        for k in range(n_replicas)
    ]

    return NTKStats(dfs)
