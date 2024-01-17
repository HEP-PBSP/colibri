from super_net.api import API as SuperNetAPI

from super_net.mc_loss_functions import make_chi2_training_data

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

    result = SuperNetAPI.make_chi2_training_data(
        **{**TEST_DATASETS, **T0_PDFSET, **TRVAL_INDEX}
    )

    # Check that make_chi2_training_data is a jit-compiled function
    assert isinstance(result, jaxlib.xla_extension.PjitFunction)
