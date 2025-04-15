import jax.numpy as jnp
import logging

from validphys import convolution

from colibri.utils import pdf_model_from_colibri_model

log = logging.getLogger(__name__)


def closure_test_pdf_grid(
    closure_test_pdf, FIT_XGRID, Q0=1.65, closure_test_model_settings={}
):
    """
    Computes the closure_test_pdf grid in the evolution basis.

    Parameters
    ----------
    closure_test_pdf: validphys.core.PDF or str
        PDF object or string colibri_model.

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
        pdf = closure_test_colibri_model_pdf(closure_test_model_settings, FIT_XGRID)
        # return the pdf by adding a dimension to simulate the number of replicas
        return jnp.array([pdf])
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
    pdf_model = pdf_model_from_colibri_model(closure_test_model_settings)

    # Compute the pdf grid
    pdf_grid_func = pdf_model.grid_values_func(FIT_XGRID)
    # params = jnp.array(closure_test_model_settings["parameters"])
    params = jnp.array(list(closure_test_model_settings["parameters"].values()))
    pdf_grid = pdf_grid_func(params)

    return pdf_grid
