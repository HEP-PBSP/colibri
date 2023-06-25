import logging

import jax
import jax.numpy as jnp
import optax

import numpy as np

from super_net.loss_utils import central_covmat_index, train_validation_split, data_training, data_validation
from super_net.loss_functions import make_chi2_training_data, make_chi2_validation_data

log = logging.getLogger(__name__)


def central_covmat_index_monte_carlo(data, dataset_inputs_t0_predictions, monte_carlo_replicas=2):
    """
    For each Monte Carlo replica 
    """
    res = []
    for _ in range(monte_carlo_replicas):
        filterseed = np.random.randint(10000)
        res.append(central_covmat_index(data, dataset_inputs_t0_predictions, pseudodata=True, filterseed=filterseed))
    return res


def train_validation_split_monte_carlo(central_covmat_index_monte_carlo, test_size=0.2):
    res = []
    for cv_cov_idx in central_covmat_index_monte_carlo:
        trval_seed = np.random.randint(10000)
        res.append(train_validation_split(
                    cv_cov_idx,
                    test_size=test_size,
                    trval_seed=trval_seed,
                    ))
    return res


def data_training_monte_carlo(train_validation_split_monte_carlo):
    res = []
    for trval_split in train_validation_split_monte_carlo:
        res.append(data_training(trval_split))
    return res


def data_validation_monte_carlo(train_validation_split_monte_carlo):
    res = []
    for trval_split in train_validation_split_monte_carlo:
        res.append(data_validation(trval_split))
    return res


def make_chi2_training_data_monte_carlo(data, data_training_monte_carlo):
    
    res=[]
    for data_training in data_training_monte_carlo:
        res.append(make_chi2_training_data(data, data_training))
    return res


def make_chi2_validation_data_monte_carlo(data, data_validation_monte_carlo):
    
    res=[]
    for data_validation in data_validation_monte_carlo:
        res.append(make_chi2_validation_data(data, data_validation))
    return res
   

def weight_minimization_fit(
    make_chi2_training_data,
    make_chi2_validation_data,
    weight_minimization_grid,
    optimizer_provider,
    early_stopper,
    max_epochs,
    data_batch_info,
    nr_validation_points,
):
    """
    TODO
    """

    INPUT_GRID, wmin_INPUT_GRID, init_weights = weight_minimization_grid

    @jax.jit
    def loss_training(weights, batch_idx):
        wmin_weights = jnp.concatenate((jnp.array([1.0]), weights))
        pdf = jnp.einsum("i,ijk", wmin_weights, wmin_INPUT_GRID)
        return make_chi2_training_data(pdf, batch_idx)

    @jax.jit
    def loss_validation(weights):
        wmin_weights = jnp.concatenate((jnp.array([1.0]), weights))
        pdf = jnp.einsum("i,ijk", wmin_weights, wmin_INPUT_GRID)
        return make_chi2_validation_data(pdf)

    @jax.jit
    def step(params, opt_state, batch_idx):
        loss_value, grads = jax.value_and_grad(loss_training)(params, batch_idx)
        updates, opt_state = optimizer_provider.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    loss = []
    val_loss = []

    opt_state = optimizer_provider.init(init_weights)
    weights = init_weights

    batches = data_batch_info["data_batch_stream_index"]
    num_batches = data_batch_info["num_batches"]
    batch_size = data_batch_info["batch_size"]

    for i in range(max_epochs):
        epoch_loss = 0
        epoch_val_loss = 0

        for _ in range(num_batches):
            batch = next(batches)

            weights, opt_state, loss_value = step(weights, opt_state, batch)

            epoch_loss += loss_training(weights, batch) / batch_size

        epoch_val_loss += loss_validation(weights) / nr_validation_points
        epoch_loss /= num_batches

        loss.append(epoch_loss)
        val_loss.append(epoch_val_loss)

        _, early_stopper = early_stopper.update(epoch_val_loss)
        if early_stopper.should_stop:
            log.info("Met early stopping criteria, breaking...")
            break

        if i % 100 == 0:
            log.info(
                f"step {i}, loss: {epoch_loss:.3f}, validation_loss: {epoch_val_loss:.3f}"
            )
            log.info(f"epoch:{i}, early_stopper: {early_stopper}")

    return {
        "weights": weights,
        "training_loss": loss,
        "validation_loss": val_loss,
        "INPUT_GRID": INPUT_GRID,
        "wmin_INPUT_GRID": wmin_INPUT_GRID,
    }





def monte_carlo_fit(
    make_chi2_training_data_monte_carlo,
    make_chi2_validation_data_monte_carlo,
    weight_minimization_grid,
    optimizer_provider,
    early_stopper,
    max_epochs,
    data_batch_info,
    nr_validation_points,
):
    """
    """
    fit_data = []
    for make_chi2_tr, make_chi2_val in zip(make_chi2_training_data_monte_carlo, make_chi2_validation_data_monte_carlo):
        fit_data.append(weight_minimization_fit(
                    make_chi2_tr,
                    make_chi2_val,
                    weight_minimization_grid,
                    optimizer_provider,
                    early_stopper,
                    max_epochs,
                    data_batch_info,
                    nr_validation_points,
            ))
    return fit_data