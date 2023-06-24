import logging

import jax
import jax.numpy as jnp
import numpy as np

import optax
from flax.training.early_stopping import EarlyStopping

from validphys.convolution import evolution
from validphys import convolution

from super_net.constants import XGRID

log = logging.getLogger(__name__)


# here a consistent use of jnp or np should be made
# seed for init weights should be passed
# @check_pdf_is_mc(wminpdfset)
def weight_minimization_grid(wminpdfset, n_replicas_wmin=50, Q0=1.65):
    """
    TODO

    add something about sum rules / wmin parametrization
    """

    INPUT_GRID = evolution.grid_values(
        wminpdfset, convolution.FK_FLAVOURS, XGRID, [Q0]
    ).squeeze(-1)

    if n_replicas_wmin > INPUT_GRID.shape[0]:
        raise (
            f"n_replicas_wmin should be <= than the number of replicas contained in the PDF set {wminpdfset}"
        )

    # reduce INPUT_GRID to only keep n_replicas_wmin PDF replicas
    random_replicas = np.random.choice(
        INPUT_GRID.shape[0], n_replicas_wmin, replace=False
    )
    # include central replica
    if not np.any(random_replicas == 0):
        random_replicas = np.append(random_replicas, 0)
        # shuffle random replicas
        np.random.shuffle(random_replicas)

    INPUT_GRID = INPUT_GRID[random_replicas]

    # generate weight minimization grid so that sum rules are automatically fulfilled

    rep1_idx = np.random.choice(INPUT_GRID.shape[0])
    wmin_INPUT_GRID = (
        np.delete(INPUT_GRID, rep1_idx, axis=0) - INPUT_GRID[np.newaxis, rep1_idx]
    )
    wmin_INPUT_GRID = np.vstack((INPUT_GRID[np.newaxis, rep1_idx], wmin_INPUT_GRID))

    # initial weights for weight minimization
    rng = jax.random.PRNGKey(0xDEADBEEF)
    weight_base_num = jax.random.normal(rng, shape=(INPUT_GRID.shape[0] - 1,))

    return INPUT_GRID, wmin_INPUT_GRID, weight_base_num


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
