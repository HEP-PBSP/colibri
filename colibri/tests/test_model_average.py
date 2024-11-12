"""
colibri.tests.test_model_average.py

Module containing tests for the model_average module.
"""

import numpy as np

from colibri.model_average import selected_fits, selected_fits_with_weights
from colibri.core import ColibriFitSpec


LOGZ1 = 1.0
LOGZ2 = 2.0


def test_selected_fits():
    """
    Test that the models are selected correctly based on the delta_logz parameter

    A = {M_k : log(Zk) >= log(Zmax) - delta_logz}
    """
    fits = [
        ColibriFitSpec(
            bayesian_metrics={"logz": LOGZ1},
            fit_path=None,
        ),
        ColibriFitSpec(
            bayesian_metrics={"logz": LOGZ2},
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
