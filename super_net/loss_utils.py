import jax
import jax.numpy as jnp

import numpy as np
import pandas as pd

from validphys import covmats

from reportengine import collect
from reportengine.configparser import ConfigError

from dataclasses import dataclass

from super_net.commondata_utils import CentralCovmatIndex


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


def posdata_train_validation_split(
    posdatasets,
    pos_test_size=0.5,
    trval_seed=42,
):
    """
    Get training validation split for the positivity data values.

    Parameters
    ----------
    posdatasets : list
            list of PositivitySetSpec

    test_size : float, default is 0.5
            size of the test/validation set, float between 0 and 1

    trval_seed : int, default is 42
                integer specifiying the random state of the training
                test split

    Returns
    -------
    tuple

    """

    ndata_pos = np.sum(
        [
            pos_ds.load_commondata().with_cuts(pos_ds.cuts).ndata
            for pos_ds in posdatasets
        ]
    )
    indices = np.arange(ndata_pos)

    indices_tr, indices_val = train_test_split(
        indices, test_size=pos_test_size, random_state=trval_seed
    )

    return indices_tr, indices_val


def posdata_training_index(posdata_train_validation_split):
    idx_tr, _ = posdata_train_validation_split
    return idx_tr


def posdata_validation_index(posdata_train_validation_split):
    _, idx_val = posdata_train_validation_split
    return idx_val


def pos_test(posdatasets):
    from validphys.fkparser import load_fktable

    for ds in posdatasets:
        for fkspec in ds.fkspecs:
            fk = load_fktable(fkspec)
            print(f"ds = {ds}, fk.hadronic = {fk.hadronic}")

    # pos_ds = posdatasets[0]
    # print(dir(posdatasets))
    # print(type(posdatasets))
    # print(type(posdatasets.data))
    # nposdata = np.sum([pos_ds.load_commondata().with_cuts(pos_ds.cuts).ndata for pos_ds in posdatasets])
    # print(nposdata)
    # print([ds for ds in posdatasets])
