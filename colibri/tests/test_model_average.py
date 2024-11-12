"""
colibri.tests.test_model_average.py

Module containing tests for the model_average module.
"""

from unittest.mock import MagicMock, patch

import numpy as np
from colibri.core import ColibriFitSpec
from colibri.model_average import (
    bayesian_model_combination,
    selected_fits,
    selected_fits_with_weights,
)

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


def test_selected_fits_with_weights():
    """
    If ∆ is the quantity of interest, or parameters of interest, then when performing model
    average we can compute its posterior distribution given data D as

    p(∆|D) = sum_k p(D|∆,M_k) p(M_k|D)

    - M_k are the different models over which we average.
    - p(M_k|D) is the posterior model probability

    Assuming that the prior probability is the same for each model we get

    p(M_k|D) = p(D|M_k) / (sum_l p(D|M_l))

    hence multiplying by 1 = exp(-log(Z_avg))/exp(-log(Z_avg)) we get

    p(D|M_k) = exp(log(Z_k) - log(Z_avg)) / (1 + sum_{l != 1} exp(log(Z_l)-log(Z_avg)))

    """

    # test that formula works correctly for a single model
    selected_fit_1 = [
        ColibriFitSpec(
            bayesian_metrics={"logz": LOGZ1},
            fit_path=None,
        ),
    ]

    sel_fit_w_1s = selected_fits_with_weights(selected_fit_1)

    assert sel_fit_w_1s[0].bayesian_metrics["bayesian_weight"] == 1.0

    # test that formula works correctly for multiple models
    selected_fit_2 = [
        ColibriFitSpec(
            bayesian_metrics={"logz": LOGZ1},
            fit_path=None,
        ),
        ColibriFitSpec(
            bayesian_metrics={"logz": LOGZ2},
            fit_path=None,
        ),
    ]
    expected = [
        np.exp(LOGZ1) / (np.exp(LOGZ1) + np.exp(LOGZ2)),
        np.exp(LOGZ2) / (np.exp(LOGZ1) + np.exp(LOGZ2)),
    ]
    sel_fit_w_2s = selected_fits_with_weights(selected_fit_2)

    for fit, exp in zip(sel_fit_w_2s, expected):
        assert np.isclose(fit.bayesian_metrics["bayesian_weight"], exp)
