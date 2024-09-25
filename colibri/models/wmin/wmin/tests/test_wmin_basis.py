"""
wmin.tests.test_wmin_basis.py

Test the wmin.basis module.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from wmin.basis import (
    wmin_pdfbasis_normalization,
    wmin_basis_sum_rules_normalization,
    sum_rules_dict,
)
from colibri.constants import FLAVOUR_TO_ID_MAPPING, LHAPDF_XGRID
from colibri.tests.conftest import TEST_PDFSET


from validphys.core import PDF
from validphys import convolution


@pytest.mark.parametrize("pdf_basis", ["intrinsic_charm", "perturbative_charm"])
def test_wmin_pdfbasis_normalization(pdf_basis):
    """
    Test that the PDF-basis, intrinsic and perturbative charm, normalisation
    works as expected.
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


def test_wmin_basis_sum_rules_normalization():
    """
    Test that the PDF basis sum-rules normalisation works as expected.
    """
    pdf = PDF(TEST_PDFSET)
    sr = sum_rules_dict(pdf)[TEST_PDFSET]

    pdf_grid = convolution.evolution.grid_values(
        pdf, convolution.FK_FLAVOURS, LHAPDF_XGRID, [1.65]
    ).squeeze(-1)

    pdf_grid_sr_norm = wmin_basis_sum_rules_normalization(pdf_grid, sum_rule_dict=sr)

    assert_allclose(
        np.trapz(
            pdf_grid_sr_norm[:, FLAVOUR_TO_ID_MAPPING["V"], :] / np.array(LHAPDF_XGRID),
            x=LHAPDF_XGRID,
            axis=-1,
        ),
        3,
        rtol=1e-2,
    )
    assert_allclose(
        np.trapz(
            pdf_grid_sr_norm[:, FLAVOUR_TO_ID_MAPPING["V3"], :]
            / np.array(LHAPDF_XGRID),
            x=LHAPDF_XGRID,
            axis=-1,
        ),
        1,
        rtol=1e-2,
    )
    assert_allclose(
        np.trapz(
            pdf_grid_sr_norm[:, FLAVOUR_TO_ID_MAPPING["V8"], :]
            / np.array(LHAPDF_XGRID),
            x=LHAPDF_XGRID,
            axis=-1,
        ),
        3,
        rtol=1e-2,
    )
    assert_allclose(
        np.trapz(
            (
                pdf_grid_sr_norm[:, FLAVOUR_TO_ID_MAPPING["\Sigma"], :]
                + pdf_grid_sr_norm[:, FLAVOUR_TO_ID_MAPPING["g"], :]
            ),
            x=LHAPDF_XGRID,
            axis=-1,
        ),
        1,
        rtol=1e-2,
    )
