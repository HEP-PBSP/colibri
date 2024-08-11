import os
from unittest.mock import Mock, patch

import jax.numpy as jnp
import numpy as np
from colibri.monte_carlo_fit import MonteCarloFit, monte_carlo_fit, run_monte_carlo_fit
from colibri.tests.conftest import MOCK_PDF_MODEL
from numpy.testing import assert_allclose

mock_pdf_model = MOCK_PDF_MODEL
N_PARAMS = len(MOCK_PDF_MODEL.param_names)


class MockOptimizerProvider:

    @staticmethod
    def init(parameters):
        return jnp.array([0.0 for _ in range(N_PARAMS)])

    @staticmethod
    def update(grads, opt_state, params):
        return jnp.array([0.0 for _ in range(N_PARAMS)]), opt_state


class MockEarlyStopper:
    def __init__(self):
        self.should_stop = False

    def update(self, epoch_val_loss):
        self.should_stop = epoch_val_loss < 0.1  # Mock early stopping criteria
        return self


def test_monte_carlo_fit_runs_without_errors():
    # Provide necessary inputs for the function

    result = monte_carlo_fit(
        _chi2_training_data_with_positivity=lambda *args: 0.0,
        _chi2_validation_data_with_positivity=lambda *args: 0.0,
        _pred_data=lambda *args: (np.zeros((N_PARAMS,)), np.zeros((N_PARAMS,))),
        fast_kernel_arrays=(np.zeros((10, 10)), np.zeros((10, 10))),
        positivity_fast_kernel_arrays=(np.zeros((10, 10)), np.zeros((10, 10))),
        len_trval_data=(100, 50),
        pdf_model=mock_pdf_model,
        mc_initial_parameters=np.zeros((N_PARAMS,)),
        optimizer_provider=MockOptimizerProvider(),
        early_stopper=MockEarlyStopper(),
        max_epochs=100,
        FIT_XGRID=np.zeros((10,)),
    )

    # Assert that the function returns an instance of MonteCarloFit
    assert isinstance(result, MonteCarloFit)
    assert result.monte_carlo_specs["max_epochs"] == 100
    assert result.monte_carlo_specs["batch_size"] == 100
    assert result.monte_carlo_specs["batch_seed"] == 1
    assert result.monte_carlo_specs["alpha"] == 1e-07
    assert result.monte_carlo_specs["lambda_positivity"] == 1000

    assert_allclose(result.optimized_parameters, jnp.array([0.0, 0.0]))
    assert_allclose(result.training_loss, jnp.array([]))
    assert_allclose(result.validation_loss, jnp.array([]))


@patch("colibri.monte_carlo_fit.write_exportgrid_mc")
def test_run_monte_carlo_fit(mock_write_exportgrid, tmp_path):

    # add side effect to the mock function to create dir
    def create_directory_side_effect(*args, **kwargs):
        os.makedirs(os.path.join(tmp_path, "fit_replicas/replica_1"), exist_ok=True)

    # Assign the side effect function to the mock object
    mock_write_exportgrid.side_effect = create_directory_side_effect

    # Define mock ultranest fit
    mock_monte_carlo_fit = Mock()
    mock_monte_carlo_fit.monte_carlo_specs = {}
    mock_monte_carlo_fit.training_loss = jnp.array([0.1, 0.2, 0.3])
    mock_monte_carlo_fit.validation_loss = jnp.array([0.2, 0.3, 0.4])
    mock_monte_carlo_fit.optimized_parameters = jnp.array([0.0, 0.0])

    # Run the run_monte_carlo_fit function
    output_path = str(tmp_path)

    run_monte_carlo_fit(
        mock_monte_carlo_fit, mock_pdf_model, output_path, replica_index=1
    )

    # Check if the write_exportgrid function was called once as expected
    assert mock_write_exportgrid.call_count == 1

    # Assertions - check if files are created in the output path
    assert (tmp_path / "fit_replicas/replica_1/mc_loss.csv").exists()
    assert (tmp_path / "fit_replicas/replica_1/mc_result_replica_1.csv").exists()
