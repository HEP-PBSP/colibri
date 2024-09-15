"""
wmin.tests.test_wmin_basis.py

Test the wmin.basis module.
"""

import pytest
from wmin.basis import wmin_pdfbasis_normalization
import numpy as np
from colibri.constants import FLAVOUR_TO_ID_MAPPING
from numpy.testing import assert_allclose


@pytest.mark.parametrize("pdf_basis", ["intrinsic_charm", "perturbative_charm"])
def test_wmin_pdfbasis_normalization(pdf_basis):
    """
    Test the normalization of the PDF basis.
    """
    pdf_grid = np.random.rand(100, 14, 50)

    pdf_grid_normalized = wmin_pdfbasis_normalization(
        pdf_grid=pdf_grid, pdf_basis=pdf_basis
    )

    sigma = pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["\Sigma"], :]
    valence = pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["V"], :]

    if pdf_basis == "intrinsic_charm":
        assert_allclose(
            pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["V15"], :], valence
        )
        assert_allclose(
            pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["V24"], :], valence
        )
        assert_allclose(
            pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["V35"], :], valence
        )
        assert_allclose(pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["T24"], :], sigma)
        assert_allclose(pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["T35"], :], sigma)

    elif pdf_basis == "perturbative_charm":
        assert_allclose(
            pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["V15"], :], valence
        )
        assert_allclose(
            pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["V24"], :], valence
        )
        assert_allclose(
            pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["V35"], :], valence
        )
        assert_allclose(pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["T15"], :], sigma)
        assert_allclose(pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["T24"], :], sigma)
        assert_allclose(pdf_grid_normalized[:, FLAVOUR_TO_ID_MAPPING["T35"], :], sigma)
