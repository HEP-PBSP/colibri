import jax.numpy as jnp

import pandas as pd
from sklearn.model_selection import train_test_split

from validphys import covmats


# @check_data_is_super_net(data)
def central_covmat_index(
    data, dataset_inputs_t0_predictions, pseudo=False, filterseed=1
):
    """
    Used to get data values, t0 covariance matrix,
    and indices of data values.

    If pseudo is False, then the experimental data values are
    returned, otherwise pseudo data is generated with random
    seed given by `filterseed`

    Parameters
    ----------
    data : super_net.core.SuperNetDataGroupSpec instance

    dataset_inputs_t0_predictions : list[jnp.array]
        The t0 predictions for all datasets.

    pseudo : bool, default is False
            if True pseudo data generated with `make_level1_data` function
            is used instead of experimental central values

    filterseed : int, default is 1
            seed used to generate pseudo data with `make_level1_data` function

    Returns
    -------
    tuple
        3D tuple containing
        - jnp.ndarray of central values of data
        - jnp.ndarray for t0 covariance matrix of data
        - jnp.ndarray of indices of data values
    """

    cd_list = data.load_pseudo_commondata(pseudo=pseudo, filterseed=filterseed)

    central_values = [cd.central_values for cd in cd_list]

    central_values = jnp.array(pd.concat(central_values, axis=0))

    covmat = jnp.array(
        covmats.dataset_inputs_t0_covmat_from_systematics(
            cd_list,
            data_input=data.dsinputs,
            use_weights_in_covmat=False,
            norm_threshold=None,
            dataset_inputs_t0_predictions=dataset_inputs_t0_predictions,
        )
    )

    indices = jnp.arange(central_values.shape[0])

    return central_values, covmat, indices


def train_validation_split(
    central_covmat_index,
    test_size=0.2,
    random_state=42,
):
    """
    Get training validation split for the data values.

    Parameters
    ----------
    central_covmat_index : super_net.loss_utils.central_covmat_index function/provider

    test_size : float, default is 0.2
            size of the test/validation set, float between 0 and 1

    random_state : int, default is 42
                integer specifiying the random state of the training
                test split

    Returns
    -------
    tuple
        6D tuple containing:
        - jnp.ndarray of training central values
        - jnp.ndarray of training covmat
        - jnp.ndarray of training indices
        - jnp.ndarray of validation central values
        - jnp.ndarray of validation covmat
        - jnp.ndarray of validation indices
    """

    central_values, covmat, indices = central_covmat_index

    # Perform train-test split on indices
    indices_train, indices_val = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )

    # Use indices to split central values, covariance matrix and predictions
    central_values_train, central_values_val = (
        central_values[indices_train],
        central_values[indices_val],
    )
    covmat_train, covmat_val = (
        covmat[indices_train][:, indices_train],
        covmat[indices_val][:, indices_val],
    )

    return (
        central_values_train,
        covmat_train,
        indices_train,
        central_values_val,
        covmat_val,
        indices_val,
    )


def data_training(train_validation_split):
    """
    Used to get training data values, t0 training covariance matrix,
    and indices of training data values.


    Returns
    -------
    dict
        - central_values_train: jnp.ndarray of central values of training data
        - t0covmat_train: jnp.ndarray for t0 training  covariance matrix of data
        - central_values_train_index: jnp.ndarray of indices of training data values
        - nr_training_points: float, number of training points
    """
    (
        central_values_train,
        covmat_train,
        indices_train,
        _,
        _,
        _,
    ) = train_validation_split

    return {
        "central_values_train": central_values_train,
        "t0covmat_train": covmat_train,
        "central_values_train_index": indices_train,
        "nr_training_points": central_values_train.shape[0],
    }


# this is a bit weird and maybe not optimal, is needed by data_batch_stream_index provider
def nr_training_points(data_training):
    """ """
    return data_training["nr_training_points"]


def data_validation(train_validation_split):
    """
    Used to get validation data values, t0 validation covariance matrix,
    and indices of validation data values.


    Returns
    -------
    dict
        - central_values_val: jnp.ndarray of central values of validation data
        - t0covmat_val: jnp.ndarray for t0 validation  covariance matrix of data
        - central_values_val_index: jnp.ndarray of indices of validation data values
        - nr_validation_points: float, number of validation points
    """
    (
        _,
        _,
        _,
        central_values_val,
        covmat_val,
        indices_val,
    ) = train_validation_split

    return {
        "central_values_val": central_values_val,
        "t0covmat_val": covmat_val,
        "central_values_val_index": indices_val,
        "nr_validation_points": central_values_val.shape[0],
    }


def nr_validation_points(data_validation):
    """ """
    return data_validation["nr_validation_points"]
