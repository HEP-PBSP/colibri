import numpy as np
from numpy.testing import assert_allclose
import jax.numpy as jnp
from colibri.monte_carlo_fit import monte_carlo_fit, MonteCarloFit
from colibri.tests.conftest import MOCK_PDF_MODEL

pdf_model = MOCK_PDF_MODEL
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
        len_trval_data=(100, 50),
        pdf_model=pdf_model,
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
