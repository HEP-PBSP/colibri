from dataclasses import dataclass



@dataclass(frozen=True)
class WeightMinimizationFit:
    pass


def weight_minimization_fit(
    make_chi2_training_data_test,
    make_chi2_validation_data_test,
    weight_minimization_grid,
    optimizer_provider,
    early_stopper,
    max_epochs,
    data_batch_info,
    nr_validation_points,
    alpha=1e-7,
    lambda_positivity=1000,
):
    """
    TODO
    """

    (
        INPUT_GRID,
        wmin_INPUT_GRID,
        init_weights,
        wmin_basis_idx,
        rep1_idx,
    ) = weight_minimization_grid

    @jax.jit
    def loss_training(weights, batch_idx):
        wmin_weights = jnp.concatenate((jnp.array([1.0]), weights))
        pdf = jnp.einsum("i,ijk", wmin_weights, wmin_INPUT_GRID)
        return make_chi2_training_data(pdf, batch_idx, alpha, lambda_positivity)

    @jax.jit
    def loss_validation(weights):
        wmin_weights = jnp.concatenate((jnp.array([1.0]), weights))
        pdf = jnp.einsum("i,ijk", wmin_weights, wmin_INPUT_GRID)
        return make_chi2_validation_data(pdf, alpha, lambda_positivity)

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
    
    batches = data_batch_info["data_batch_stream_index"]()
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

        if i % 5 == 0:
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
        "wmin_basis_idx": wmin_basis_idx,
        "rep1_idx": rep1_idx,
    }
