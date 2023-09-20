from reportengine.figure import figure, figuregen
from validphys import plotutils


@figure
def plot_training_validation(weight_minimization_fit):
    training_loss = weight_minimization_fit["training_loss"]
    validation_loss = weight_minimization_fit["validation_loss"]

    fig, ax = plotutils.subplots()
    ax.plot(training_loss, alpha=0.5, label="training loss")
    ax.plot(validation_loss, alpha=0.5, label="validation loss")
    ax.legend()
    return fig


@figuregen
def plot_wmin_PDFs(weight_minimization_fit, wminpdfset):
    INPUT_GRID = weight_minimization_fit["INPUT_GRID"]
    wmin_INPUT_GRID = weight_minimization_fit["wmin_INPUT_GRID"]
    weights = weight_minimization_fit["weights"]

    import jax.numpy as jnp

    for fl in range(2, 14):
        fig, ax = plotutils.subplots()
        ax.plot(
            jnp.einsum(
                "i,ijk->jk", jnp.concatenate((jnp.array([1]), weights)), wmin_INPUT_GRID
            )[fl, :],
            label="wmin PDF",
        )
        ax.plot(INPUT_GRID[0, fl, :], label=f"wmin pdf set: {wminpdfset}")
        ax.legend()
        yield fig
