import logging

import jax
import jax.numpy as jnp
import optax

import numpy as np

from super_net.loss_utils import (
    central_covmat_index,
    train_validation_split,
    data_training,
    data_validation,
)
from super_net.loss_functions import make_chi2_training_data, make_chi2_validation_data
from super_net.wmin_utils import lhapdf_from_weights

log = logging.getLogger(__name__)


def central_covmat_index_monte_carlo(
    data, dataset_inputs_t0_predictions, monte_carlo_replicas=2
):
    """
    This function is responsible for the pseudodata generation
    needed in a Monte Carlo fit.

    Parameters
    ----------
    data : super_net.core.SuperNetDataGroupSpec

    dataset_inputs_t0_predictions : list
            list of t0 predictions each element corresponds to t0
            prediction for a dataset

    monte_carlo_replicas : int, default is 2
            number of wmin replicas to be fitted

    Returns
    -------
    list
    list of len monte_carlo_replicas, each element is central_covmat_index
    computed with a random filterseed.

    """

    res = []
    for _ in range(monte_carlo_replicas):
        filterseed = np.random.randint(1000000)
        res.append(
            central_covmat_index(
                data,
                dataset_inputs_t0_predictions,
                pseudodata=True,
                filterseed=filterseed,
            )
        )
    return res


def train_validation_split_monte_carlo(central_covmat_index_monte_carlo, test_size=0.2):
    """
    This function is responsible for splitting the monte carlo pseudo data into
    training and validation set

    Parameters
    ----------
    central_covmat_index_monte_carlo : list
                output of function super_net.monte_carlo_utils.central_covmat_index_monte_carlo

    test_size : float, default is 0.2

    Returns
    -------
    list
    list of len monte_carlo_replicas.

    """
    res = []
    for cv_cov_idx in central_covmat_index_monte_carlo:
        trval_seed = np.random.randint(1000000)
        res.append(
            train_validation_split(
                cv_cov_idx,
                test_size=test_size,
                trval_seed=trval_seed,
            )
        )
    return res


def data_training_monte_carlo(train_validation_split_monte_carlo):
    """
    Returns a list of the training data. Each element corresponds
    to training data for a separate replica.

    Parameters
    ----------
    train_validation_split_monte_carlo: list
        output of the function train_validation_split_monte_carlo

    Returns
    -------
    list

    """
    res = []
    for trval_split in train_validation_split_monte_carlo:
        res.append(data_training(trval_split))
    return res


def data_validation_monte_carlo(train_validation_split_monte_carlo):
    """
    Same as `data_training_monte_carlo` but for validation data.

    Parameters
    ----------
    train_validation_split_monte_carlo : list
        output of the function train_validation_split_monte_carlo

    Returns
    -------
    list

    """
    res = []
    for trval_split in train_validation_split_monte_carlo:
        res.append(data_validation(trval_split))
    return res


def make_chi2_training_data_monte_carlo(
    data, data_training_monte_carlo, posdatasets, posdata_training_index
):
    """
    Same as `make_chi2_training_data` but for a list of pseudodata.
    """
    res = []
    for data_training in data_training_monte_carlo:
        res.append(
            make_chi2_training_data(
                data, data_training, posdatasets, posdata_training_index
            )
        )
    return res


def make_chi2_validation_data_monte_carlo(
    data, data_validation_monte_carlo, posdatasets, posdata_validation_index
):
    """
    Same as `make_chi2_validation_data` but for a list of pseudodata.
    """
    res = []
    for data_validation in data_validation_monte_carlo:
        res.append(
            make_chi2_validation_data(
                data, data_validation, posdatasets, posdata_validation_index
            )
        )
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


def monte_carlo_fit(
    make_chi2_training_data_monte_carlo,
    make_chi2_validation_data_monte_carlo,
    weight_minimization_grid,
    weights_initializer_provider,
    n_replicas_wmin,
    wminpdfset,
    optimizer_provider,
    early_stopper,
    max_epochs,
    data_batch_info,
    nr_validation_points,
    alpha=1e-7,
    lambda_positivity=1000,
    random_parametrisation=True,
):
    """
    Fits Monte Carlo pseudodata using weight minimisation parametrisation.
    If random_parametrisation is True, then for each Monte Carlo replica a 
    different 'weight minimisation' parametrisation is used.
    
    Parameters
    ----------
    make_chi2_training_data_monte_carlo: list, list of jit compiled jax functions

    make_chi2_validation_data_monte_carlo: list, list of jit compiled jax functions

    weight_minimization_grid: tuple, contains the weight minimisation grid at Q0=1.65Gev

    weights_initializer_provider: function 
        takes shape=integer in input and returns array of shape = (shape, )
    
    n_replicas_wmin: int
        number of replicas from wminpdfset to be used in the weight
        minimization parametrization

    wminpdfset: validphys.core.PDF
    
    optimizer_provider: optax._src.base.GradientTransformationExtraArgs

    early_stopper: int

    max_epochs: int
    
    data_batch_info: dict, contains info about data batches

    nr_validation_points: int

    alpha: float, default is 1e-7
        alpha parameter of Elu function for positivity constraint

    lambda_positivity: integer or float, default is 1000
        positivity penalty parameter

    random_parametrisation: bool, default is True
        whether to use a different (randomly picked) central replica in the weight min parametrisation

    Returns
    -------
    list

    """
    if random_parametrisation:
        fit_data = []
        for make_chi2_tr, make_chi2_val in zip(
            make_chi2_training_data_monte_carlo, make_chi2_validation_data_monte_carlo
        ):
            from super_net.wmin_model import weight_minimization_grid

            rng_seed = np.random.randint(1000000)
            # for each wmin replica fit, pick a random replica from the basis to be
            # used as the central replica of the wmin parametrisation
            wmin_grid = weight_minimization_grid(
                wminpdfset,
                weights_initializer_provider,
                n_replicas_wmin,
                rng_jax=rng_seed,
            )
            fit_data.append(
                weight_minimization_fit(
                    make_chi2_tr,
                    make_chi2_val,
                    wmin_grid,
                    optimizer_provider,
                    early_stopper,
                    max_epochs,
                    data_batch_info,
                    nr_validation_points,
                    alpha,
                    lambda_positivity,
                )
            )
        return fit_data

    else:
        return [
            weight_minimization_fit(
                make_chi2_tr,
                make_chi2_val,
                weight_minimization_grid,
                optimizer_provider,
                early_stopper,
                max_epochs,
                data_batch_info,
                nr_validation_points,
                alpha,
                lambda_positivity,
            )
            for make_chi2_tr, make_chi2_val in zip(
                make_chi2_training_data_monte_carlo,
                make_chi2_validation_data_monte_carlo,
            )
        ]


def snmc_fit(monte_carlo_fit, wminpdfset, folder="", set_name=None):
    """
    TODO
    """

    weights = []
    wmin_basis_idxs = []
    rep1_idxs = []

    for replica_fit in monte_carlo_fit:
        weights.append(replica_fit["weights"])
        wmin_basis_idxs.append(replica_fit["wmin_basis_idx"])
        rep1_idxs.append(replica_fit["rep1_idx"])

    weights = np.array(weights)

    lhapdf_from_weights(
        wminpdfset,
        weights,
        folder=folder,
        set_name=set_name,
        errortype="replicas",
        wmin_basis_idxs=wmin_basis_idxs,
        rep1_idxs=rep1_idxs,
    )
