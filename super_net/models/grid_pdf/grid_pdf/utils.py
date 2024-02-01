"""
grid_pdf.utils.py

Module containing util functions for grid PDF fits.
"""

from validphys import convolution
from super_net.constants import XGRID

def closure_test_central_pdf_grid(
    closure_test_pdf,
    pdf_model,
    reduced_xgrid_data=False,
):
    """
    Computes the central member of the closure_test_pdf grid in the
    evolution basis and only on x points that are specified in xgrids.
    The grid is then interpolated to the full XGRID.

    NOTE: when reduced_xgrid_data=True, this function overrides the one in super_net.utils
    otherwise the one in super_net.utils is used.

    Parameters
    ----------
    closure_test_pdf: validphys.core.PDF

    pdf_model: PDFModel
        Specifically, this is the GridPDFModel for this provider.

    reduced_xgrid_data: bool, default is True
        When True the closure_test_central_pdf_grid is overriden.
        When False the closure_test_pdf_grid from super_net.utils is used.

    Q0: float, default is 1.65

    Returns
    -------
    grid: jnp.array
        grid, is N_fl x N_x
    """

    if not reduced_xgrid_data:
        return super_net.utils.closure_test_pdf_grid(closure_test_pdf, Q0=1.65)[0]

    # Obtain the PDF values as parameters, then use the model interpolation function
    interpolator = pdf_model.grid_values_func(XGRID)

    parameters = []
    for fl in pdf_model.xgrids:
        x_vals = pdf_model.xgrids[flavour]
        parameters += [convolution.evolution.grid_values(closure_test_pdf, [fl], x_vals, [1.65])]

    reduced_pdfgrid = interpolator(parameters)

    return reduced_pdfgrid
