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


def ntk_computation(
    pdf_model, replicas_path, replica_index, ntk_plots_settings, ntk_plots_path=None
):

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

    def plot_eigenvalue_evolution(epochs, eigenvalues_all_epochs, ntk_plots_settings):
        """
        Generate eigenvalue evolution plot.

        Parameters
        ----------
        epochs : np.ndarray
            Array of epoch numbers
        eigenvalues_all_epochs : np.ndarray
            2D array of eigenvalues for each epoch
        ntk_plots_settings : dict
            Dictionary containing plot settings
        """
        log.info("Plotting NTK eigenvalue evolution...")

        n_top_eigenvalues = ntk_plots_settings.get("n_top_eigenvalues", 5)

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

        # Save immediately if path provided
        if ntk_plots_path is not None:
            plot_path = ntk_plots_path / "eigenvalue_evolution.pdf"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            log.info(f"Saved plot: {plot_path}")

    def plot_eigenvector_evolution(
        epochs, eigenvectors_all_epochs, eigenvector_idx, ntk_plots_settings
    ):
        """
        Plot the evolution of a specific eigenvector across epochs.

        Parameters
        ----------
        epochs : np.ndarray
            Array of epoch numbers
        eigenvectors_all_epochs : np.ndarray
            3D array of eigenvectors for each epoch
        eigenvector_idx : int
            Index of the eigenvector to plot
        ntk_plots_settings : dict
            Dictionary containing plot settings
        """
        log.info(f"Plotting eigenvector {eigenvector_idx + 1} evolution...")

        # Flavour labels
        flavour_labels = [
            "γ",
            "Σ",
            "g",
            "V",
            "V3",
            "V8",
            "V15",
            "V24",
            "V35",
            "T3",
            "T8",
            "T15",
            "T24",
            "T35",
        ]

        # Get number of epochs to plot
        plot_n_epochs = ntk_plots_settings.get("plot_n_epochs")

        # Select equally spaced epochs
        n_available = len(epochs)
        if plot_n_epochs >= n_available:
            epoch_indices = range(n_available)
        else:
            epoch_indices = np.linspace(0, n_available - 1, plot_n_epochs, dtype=int)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for plot_idx, epoch_idx in enumerate(epoch_indices[:6]):
            if plot_idx >= 6:
                break

            # Get eigenvector and reshape from [700] to [14, 50]
            eigenvector_flat = eigenvectors_all_epochs[epoch_idx, :, eigenvector_idx]
            eigenvector = reshape_eigenvector(eigenvector_flat)
            epoch_num = epochs[epoch_idx]

            # Plot each of the 14 flavours
            for flavour in range(14):
                (line,) = axes[plot_idx].plot(
                    XGRID,
                    eigenvector[flavour, :],
                    alpha=0.7,
                    linewidth=1.5,
                    label=flavour_labels[flavour],
                )

            axes[plot_idx].set_title(f"Epoch {epoch_num}", fontsize=12)
            axes[plot_idx].set_xlabel("x", fontsize=10)
            axes[plot_idx].set_ylabel("Eigenvector Value", fontsize=10)
            axes[plot_idx].grid(True, alpha=0.3)

            # Add legend only to top right plot (index 2)
            if plot_idx == 2:
                axes[plot_idx].legend(loc="best", fontsize=8, ncol=2)

        # Remove empty subplots
        for plot_idx in range(len(list(epoch_indices[:6])), 6):
            axes[plot_idx].remove()

        plt.suptitle(
            f"Evolution of Eigenvector {eigenvector_idx + 1} Across Epochs",
            fontsize=14,
            y=1.02,
        )
        plt.tight_layout()

        # Save immediately if path provided
        if ntk_plots_path is not None:
            plot_path = (
                ntk_plots_path / f"eigenvector_{eigenvector_idx + 1}_evolution.pdf"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            log.info(f"Saved plot: {plot_path}")

    def plot_eigenvector_heatmap(
        epochs, eigenvectors_all_epochs, eigenvector_idx, ntk_plots_settings
    ):

        log.info("Plotting eigenvector heatmaps...")

        plot_n_epochs = ntk_plots_settings.get("plot_n_epochs")
        n_available = len(epochs)

        # Select epochs (equally spaced)
        if plot_n_epochs >= n_available:
            epoch_indices = list(range(n_available))
        else:
            epoch_indices = list(
                np.linspace(0, n_available - 1, plot_n_epochs, dtype=int)
            )

        # eigenvector_idx should be a list like [1, 2] (1-based from runcard)
        if isinstance(eigenvector_idx, int):
            eigenvector_idx = [eigenvector_idx]

        # convert to 0-based
        eigvec_indices = [i - 1 for i in eigenvector_idx]
        n_cols = len(eigvec_indices)
        n_rows = len(epoch_indices)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(10, 10),
            squeeze=False,
        )

        for ax in axes.flat:
            ax.set_box_aspect(1)

        # Compute global color scale so all panels are comparable
        vmin = np.inf
        vmax = -np.inf
        for epoch_idx in epoch_indices:
            for ev in eigvec_indices:
                ev_flat = eigenvectors_all_epochs[epoch_idx, :, ev]
                ev_mat = reshape_eigenvector(ev_flat)  # (14, 50)
                vmin = min(vmin, float(np.min(ev_mat)))
                vmax = max(vmax, float(np.max(ev_mat)))

        for r, epoch_idx in enumerate(epoch_indices):
            epoch_num = epochs[epoch_idx]
            for c, ev in enumerate(eigvec_indices):
                ax = axes[r, c]

                ev_flat = eigenvectors_all_epochs[epoch_idx, :, ev]  # (700,)
                ev_mat = reshape_eigenvector(ev_flat)  # (14, 50)

                im = ax.imshow(
                    ev_mat,
                    aspect="auto",
                    origin="lower",
                    interpolation="nearest",
                    vmin=vmin,
                    vmax=vmax,
                )

                # Column titles = eigenvector number (1-based)
                if r == 0:
                    lambda_ev = eigenvalues_all_epochs[-1, ev]
                    ax.set_title(
                        f"Eigenvector {ev + 1}\n" rf"$\lambda = {lambda_ev:.3e}$",
                        fontsize=10,
                        pad=8,
                    )

                # Row labels = epoch
                # Y axis: flavour index (14 → 0)
                ax.set_yticks(np.arange(0, 15, 2))

                if c == 0:
                    ax.set_yticklabels(list(range(14, -1, -2)))  # labels: 14,12,...,0
                    ax.set_ylabel(f"Epoch {epoch_num}\nFlavour", fontsize=11)
                else:
                    ax.set_yticklabels([])

                ax.set_xticks(np.arange(0, ev_mat.shape[1], 10))
                if r == n_rows - 1:
                    ax.set_xlabel("x-grid", fontsize=10)
                else:
                    ax.set_xticklabels([])

        # One shared colorbar for whole figure
        cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label("Eigenvector value")

        if ntk_plots_settings.get("ntk_plots") and ntk_plots_path is not None:
            plot_path = ntk_plots_path / "eigenvector_heatmap.pdf"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            log.info(f"Saved plot: {plot_path}")

    # ------------------------------------------------------------
    # STEP 1: Compute NTK for each epoch
    # ------------------------------------------------------------

    epochs = []
    eigenvalues_all_epochs = []
    eigenvectors_all_epochs = []

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

        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = compute_eigendecomposition(ntk_matrix)

        epochs.append(epoch)
        eigenvalues_all_epochs.append(eigvals)
        eigenvectors_all_epochs.append(eigvecs)

    # Convert to numpy arrays
    epochs = np.array(epochs)
    eigenvalues_all_epochs = np.array(eigenvalues_all_epochs)
    eigenvectors_all_epochs = np.array(eigenvectors_all_epochs)

    log.info(f"Computed NTK eigenvalues for {len(epochs)} epochs")
    log.info(f"Eigenvalues shape: {eigenvalues_all_epochs.shape}")

    # =====================================================================
    #  OPTIONAL PLOTTING
    # =====================================================================

    if ntk_plots_settings.get("ntk_plots"):
        # Create plots folder
        if ntk_plots_path is not None:
            ntk_plots_path.mkdir(parents=True, exist_ok=True)

        # Plot and save eigenvalues
        plot_eigenvalue_evolution(epochs, eigenvalues_all_epochs, ntk_plots_settings)

        # Plot and save eigenvector evolution for specified eigenvectors
        plot_n_eigenvectors = ntk_plots_settings.get("plot_n_eigenvectors")

        for eigvec_idx in plot_n_eigenvectors:
            plot_eigenvector_evolution(
                epochs, eigenvectors_all_epochs, eigvec_idx - 1, ntk_plots_settings
            )

        plot_eigenvector_heatmap(
            epochs, eigenvectors_all_epochs, plot_n_eigenvectors, ntk_plots_settings
        )
    else:
        log.info("NTK plotting disabled")

    # =====================================================================
    #  PRINT STATISTICS
    # =====================================================================

    log.info("=== NTK Eigenvalue Analysis Summary ===")
    log.info(f"Number of epochs analyzed: {len(epochs)}")
    log.info(f"Epoch range: {epochs[0]} to {epochs[-1]}")
    log.info(f"NTK matrix size: {eigenvalues_all_epochs.shape[1]} x {14 * 50}")

    log.info("Top 5 eigenvalues at selected epochs:")

    # Representative epoch indices
    key_epoch_indices = [
        0,
        len(epochs) // 4,
        len(epochs) // 2,
        3 * len(epochs) // 4,
        -1,
    ]

    for idx in key_epoch_indices:
        epoch_num = epochs[idx]
        evals = eigenvalues_all_epochs[idx, :5]  # top 5 eigenvalues only

        log.info(f"Epoch {epoch_num}:")
        for i, ev in enumerate(evals, start=1):
            log.info(f"  λ_{i} = {ev:.6e}")

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
        pdf_model, replicas_path, replica_index, ntk_plots_settings, ntk_plots_path
    )

    # Save NTK results
    ntk_replica_folder = ntk_replicas_path / f"replica_{replica_index}"
    jnp.savez(ntk_replica_folder / f"ntk_epoch_{epoch}.npz", ntk=ntk)

    log.info("NTK computation finished.")
