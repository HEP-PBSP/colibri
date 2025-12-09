"""
colibri.ntk.py

This module contains the routine that computes the Neural Tangent Kernel (NTK)
for a given PDF model.

"""

import jax
import jax.numpy as jnp
import logging

from colibri.constants import XGRID

log = logging.getLogger(__name__)

def compute_ntk(ntk_replicas_path, pdf_model, replicas_path, replica_index):
  log.info("Computing Neural Tangent Kernel (NTK)...")

  # Ensure parameter folder exists and not empty
  params_folder = replicas_path / f"replica_{replica_index}/parameters"
  if not params_folder.exists() or not any(params_folder.glob("*.npz")):
    raise FileNotFoundError(f"Parameters folder {params_folder} does not exist or is empty.")

  pdf_func = pdf_model.grid_values_func(XGRID)
  jacobian_func = jax.jacfwd(pdf_func)

  ntk_replica_folder = ntk_replicas_path / f"replica_{replica_index}"
  ntk_replica_folder.mkdir(parents=True, exist_ok=True)


  # Loop through the parameters and compute the NTK
  ntk = jnp.zeros((len(pdf_model.param_names), len(pdf_model.param_names)))
  for param_file in params_folder.glob("*.npz"):
    epoch = int(param_file.stem.split("_")[-1])
    log.info(f"Computing NTK for epoch {epoch}")

    params = jnp.load(param_file)["params"]
    jacobian = jacobian_func(params)
    ntk = jnp.einsum('ijk,lmk->ijlm', jacobian, jacobian)
    jnp.savez(ntk_replica_folder / f"ntk_epoch_{epoch}.npz", ntk=ntk)

