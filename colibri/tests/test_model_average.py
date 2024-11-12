"""
colibri.tests.test_model_average.py

Module containing tests for the model_average module.
"""

from colibri.model_average import selected_fits
from colibri.core import ColibriFitSpec


def test_selected_fits():
    """
    Test that the models are selected correctly based on the delta_logz parameter

    A = {M_k : log(Zk) >= log(Zmax) - delta_logz}
    """
    fits = [
        ColibriFitSpec(
            bayesian_metrics={"logz": 1.0},
            fit_path=None,
        ),
        ColibriFitSpec(
            bayesian_metrics={"logz": 2.0},
            fit_path=None,
        ),
    ]

    delta_logz_1 = 0.5
    expected_1 = [fits[1]]

    delta_logz_2 = 1.5
    expected_2 = fits

    selected_fits_1 = selected_fits(fits, delta_logz=delta_logz_1)
    selected_fits_2 = selected_fits(fits, delta_logz=delta_logz_2)

    assert selected_fits_1 == expected_1
    assert selected_fits_2 == expected_2
