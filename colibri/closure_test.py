import jax.numpy as jnp
import logging

from validphys import convolution

from colibri.utils import produce_pdf_model_from_colibri_model

log = logging.getLogger(__name__)


def closure_test_pdf_grid(
    closure_test_pdf, FIT_XGRID, Q0=1.65, closure_test_model_settings={}
):
    """
    Computes the closure_test_pdf grid in the evolution basis.

    Parameters
    ----------
    closure_test_pdf: validphys.core.PDF

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    Q0: float, default is 1.65

    Returns
    -------
    grid: jnp.array
        grid, is N_rep x N_fl x N_x
    """
    if isinstance(closure_test_pdf, str):
        if closure_test_pdf == "colibri_model":
            pdf = closure_test_colibri_model_pdf(closure_test_model_settings, FIT_XGRID)
            return [pdf]
        else:
            raise ValueError(
                f"Unknown closure test pdf '{closure_test_pdf}'. "
                "Supported values are 'colibri_model' or LHAPDF sets."
            )
    else:
        grid = jnp.array(
            convolution.evolution.grid_values(
                closure_test_pdf, convolution.FK_FLAVOURS, FIT_XGRID, [Q0]
            ).squeeze(-1)
        )
    return grid


def closure_test_central_pdf_grid(closure_test_pdf_grid):
    """
    Returns the central replica of the closure test pdf grid.
    """
    return closure_test_pdf_grid[0]


def closure_test_colibri_model_pdf(closure_test_model_settings, FIT_XGRID):
    """
    Computes the closure test pdf grid from a colibri model.

    Parameters
    ----------
    closure_test_model_settings: dict
        Settings for the closure test model.

    FIT_XGRID: jnp.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    Returns
    -------
    jnp.array
        The closure test pdf grid.
    """
    # Produce the pdf model
    model_name = closure_test_model_settings["model"]
    pdf_model = produce_pdf_model_from_colibri_model(
        model_name, closure_test_model_settings
    )

    # Compute the pdf grid
    pdf_grid_func = pdf_model.grid_values_func(FIT_XGRID)
    params = jnp.array(closure_test_model_settings["parameters"])
    pdf_grid = pdf_grid_func(params)

    return pdf_grid
