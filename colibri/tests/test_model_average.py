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


@patch("colibri.model_average.ns_fit_resampler")
@patch("colibri.model_average.write_resampled_ns_fit")
def test_bayesian_model_combination(mock_write_resampled_ns_fit, mock_ns_fit_resampler):
    # Mocked parameters for the test
    n_samples = 10
    model_avg_fit_path = "path/to/model_avg_fit"
    model_avg_fit_name = "model_avg_fit_name"
    parametrisation_scale = 1.65
    resampling_seed = 1

    # Sample input data with mock bayesian weights and fit paths
    selected_fits_with_weights = [
        MagicMock(bayesian_metrics={"bayesian_weight": 0.4}, fit_path="path/to/fit1"),
        MagicMock(bayesian_metrics={"bayesian_weight": 0.6}, fit_path="path/to/fit2"),
    ]

    # Mock the ns_fit_resampler return value
    mock_ns_fit_resampler.side_effect = [
        ["sample_1_1", "sample_1_2", "sample_1_3", "sample_1_4"],
        [
            "sample_2_1",
            "sample_2_2",
            "sample_2_3",
            "sample_2_4",
            "sample_2_5",
            "sample_2_6",
        ],
    ]

    # Call the function under test
    bayesian_model_combination(
        selected_fits_with_weights=selected_fits_with_weights,
        n_samples=n_samples,
        model_avg_fit_path=model_avg_fit_path,
        model_avg_fit_name=model_avg_fit_name,
        parametrisation_scale=parametrisation_scale,
        resampling_seed=resampling_seed,
    )

    # Verify ns_fit_resampler was called correctly
    mock_ns_fit_resampler.assert_any_call(
        selected_fits_with_weights[0].fit_path,
        n_replicas=4,
        resampling_seed=resampling_seed,
    )
    mock_ns_fit_resampler.assert_any_call(
        selected_fits_with_weights[1].fit_path,
        n_replicas=6,
        resampling_seed=resampling_seed,
    )

    # Verify write_resampled_ns_fit was called with the expected arguments
    mock_write_resampled_ns_fit.assert_any_call(
        resampled_posterior=["sample_1_1", "sample_1_2", "sample_1_3", "sample_1_4"],
        fit_path=selected_fits_with_weights[0].fit_path,
        resampled_fit_path=model_avg_fit_path,
        n_replicas=4,
        resampled_fit_name=model_avg_fit_name,
        parametrisation_scale=parametrisation_scale,
        copy_fit_dir=True,
        write_ns_results=True,
        replica_range=None,
    )
    mock_write_resampled_ns_fit.assert_any_call(
        resampled_posterior=[
            "sample_2_1",
            "sample_2_2",
            "sample_2_3",
            "sample_2_4",
            "sample_2_5",
            "sample_2_6",
        ],
        fit_path=selected_fits_with_weights[1].fit_path,
        resampled_fit_path=model_avg_fit_path,
        n_replicas=6,
        resampled_fit_name=model_avg_fit_name,
        parametrisation_scale=parametrisation_scale,
        copy_fit_dir=False,
        write_ns_results=False,
        replica_range=range(4, 10),
    )
