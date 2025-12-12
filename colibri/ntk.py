"""
colibri.ntk.py

This module contains the routine that computes the Neural Tangent Kernel (NTK)
for a given PDF model.

"""

import jax
import jax.numpy as jnp
import logging
import numpy as np
import matplotlib.pyplot as plt

from colibri.constants import XGRID

log = logging.getLogger(__name__)


def ntk_computation(pdf_model, replicas_path, replica_index, ntk_plots_settings):

    log.info("Computing Neural Tangent Kernel (NTK)...")

    # Ensure parameter folder exists and not empty
    params_folder = replicas_path / f"replica_{replica_index}/parameters"
    if not params_folder.exists() or not any(params_folder.glob("*.npz")):
        raise FileNotFoundError(
            f"Parameters folder {params_folder} does not exist or is empty."
        )

    pdf_func = pdf_model.grid_values_func(XGRID)
    jacobian_func = jax.jacfwd(pdf_func)

    # ------------------------------------------------------------
    # STEP 0: Define helpers locally (flatten, reshape, eigendecomp)
    # ------------------------------------------------------------

    def flatten_ntk_tensor(ntk):
        """Flatten 4D NTK tensor [14, 50, 14, 50] → [700, 700]."""
        d1, d2, d3, d4 = ntk.shape
        return ntk.reshape(d1 * d2, d3 * d4)

    def reshape_eigenvector(eigenvector, original_shape=(14, 50)):
        """
        Reshape eigenvector from [14*50] back to [14, 50]
        """
        return eigenvector.reshape(original_shape)

    def compute_eigendecomposition(ntk_matrix):
        """Compute eigenvalues and eigenvectors."""
        ntk_np = np.asarray(ntk_matrix)
        if not np.allclose(ntk_np, ntk_np.T, rtol=1e-10):
            log.warning("NTK matrix is not symmetric")

        eigvals, eigvecs = np.linalg.eigh(ntk_np)

        idx = np.argsort(np.abs(eigvals))[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        return eigvals, eigvecs

    # ------------------------------------------------------------
    # STEP 1: Compute NTK for each epoch
    # ------------------------------------------------------------

    epochs = []
    eigenvalues_all_epochs = []

    # Loop through the parameters and compute the NTK
    ntk = jnp.zeros((len(pdf_model.param_names), len(pdf_model.param_names)))

    param_files = list(params_folder.glob("*.npz"))
    param_files.sort(key=lambda f: int(f.stem.split("_")[-1]))

    for param_file in param_files:
        epoch = int(param_file.stem.split("_")[-1])
        log.info(f"Computing NTK for epoch {epoch}")

        params = jnp.load(param_file)["params"]
        jacobian = jacobian_func(params)

        # Compute NTK (14,50,14,50)
        ntk = jnp.einsum("ijk,lmk->ijlm", jacobian, jacobian)

        # Flatten to 700×700
        ntk_matrix = flatten_ntk_tensor(ntk)

        # Compute eigenvalues
        eigvals, eigvecs = compute_eigendecomposition(ntk_matrix)

        epochs.append(epoch)
        eigenvalues_all_epochs.append(eigvals)

    # Convert to numpy arrays
    epochs = np.array(epochs)
    eigenvalues_all_epochs = np.array(eigenvalues_all_epochs)

    log.info(f"Computed NTK eigenvalues for {len(epochs)} epochs")
    log.info(f"Eigenvalues shape: {eigenvalues_all_epochs.shape}")

    # =====================================================================
    #  OPTIONAL PLOTTING
    # =====================================================================

    if not ntk_plots_settings.get("ntk_plots", False):
        log.info("NTK plotting disabled")
        return

    if ntk_plots_settings.get("ntk_plots", False):
        log.info("Plotting NTK plots...")

        n_top_eigenvalues = ntk_plots_settings.get("n_top_eigenvalues", 5)
        print(n_top_eigenvalues)

        plt.figure(figsize=(10, 6))

        for i in range(min(n_top_eigenvalues, eigenvalues_all_epochs.shape[1])):
            plt.plot(
                epochs,
                eigenvalues_all_epochs[:, i],
                "o-",
                label=f"Eigenvalue {i+1}",
                linewidth=2,
                markersize=6,
            )
            if ntk_plots_settings.get("x_scale") == "log":
                plt.xscale("log")

            if ntk_plots_settings.get("y_scale") == "log":
                plt.yscale("log")

            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Eigenvalue", fontsize=12)
            plt.title("NTK Eigenvalue Evolution")
            plt.legend()
            plt.grid(True)

    return ntk, epoch


def compute_ntk(
    ntk_replicas_path,
    pdf_model,
    replicas_path,
    replica_index,
    ntk_plots_settings,
    ntk_plots_path,
):
    """
    Runs ntk_computation and saves the results.

    Plots NTK plots when the user sets ntk_plots: True in the runcard.
    """

    log.info(f"Running NTK computation for replica {replica_index}")

    ntk, epoch = ntk_computation(
        pdf_model, replicas_path, replica_index, ntk_plots_settings
    )

    # Save NTK results

    ntk_replica_folder = ntk_replicas_path / f"replica_{replica_index}"
    jnp.savez(ntk_replica_folder / f"ntk_epoch_{epoch}.npz", ntk=ntk)

    log.info("NTK computation finished.")

    # Save plots

    ntk_plots_folder = ntk_plots_path
    ntk_plots_folder.mkdir(parents=True, exist_ok=True)

    if ntk_plots_settings.get("ntk_plots", False):
        plot_path = ntk_plots_folder / f"eigenvalue_evolution.pdf"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        log.info(f"Saved plot: {plot_path}")
