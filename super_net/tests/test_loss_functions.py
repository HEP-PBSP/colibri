from super_net.api import API as SuperNetAPI

from super_net.loss_functions import make_chi2_training_data

from super_net.tests.conftest import TEST_DATASETS, T0_PDFSET, TRVAL_INDEX, TEST_PDFSET

from numpy.testing import assert_allclose

import jaxlib
import jax.numpy as jnp

from super_net.constants import XGRID
from validphys.core import PDF
from validphys import convolution

def test_make_chi2_training_data():
    """
    Test the function produced by make_chi2_training_data works as expected. In particular tests:
    - make_chi2_training_data returns a jit-compiled function
    -  
    """

    data = SuperNetAPI._data_values(**{**TEST_DATASETS, **T0_PDFSET, **TRVAL_INDEX})
    pred_data = SuperNetAPI._pred_data(**TEST_DATASETS)
    result = make_chi2_training_data(data, pred_data)

    # Check that make_chi2_training_data is a jit-compiled function
    assert isinstance(result, jaxlib.xla_extension.PjitFunction)

    # Check that make_chi2_training_data returns a single float
    # TODO
    #tr_idx = data.training_data.central_values_idx
    #pdf = PDF(TEST_PDFSET)
    #pdf_grid = jnp.array(
    #    convolution.evolution.grid_values(
    #        pdf, convolution.FK_FLAVOURS, XGRID, [1.65]
    #    ).squeeze(-1)
    #)[0,:,:].squeeze()
    #chi2 = result(pdf_grid, tr_idx)
    #assert isinstance(chi2, float)
