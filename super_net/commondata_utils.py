"""
TODO
"""
import pandas as pd
from dataclasses import dataclass

import jax.numpy as jnp

from validphys.pseudodata import make_level1_data

from reportengine import collect


def pseudodata_commondata_tuple(data, experimental_commondata_tuple, replica_seed):
    """
    Returns a tuple (validphys nodes should be immutable)
    of commondata instances with experimental central values
    fluctuated with random noise sampled from experimental
    covariance matrix

    Parameters
    ----------
    data: super_net.core.SuperNetDataGroupSpec

    experimental_commondata_tuple: tuple
        tuple of commondata with experimental central values

    replica_seed: int
        seed used for the sampling of random noise

    Returns
    -------
    tuple
        tuple of validphys.coredata.CommonData instances
    """

    index = data.data_index()
    dataset_order = [cd.setname for cd in experimental_commondata_tuple]
    pseudodata_list = make_level1_data(
        data, experimental_commondata_tuple, replica_seed, index, sep_mult=True
    )
    pseudodata_list = sorted(
        pseudodata_list, key=lambda obj: dataset_order.index(obj.setname)
    )
    return tuple(pseudodata_list)


def closuretest_pseudodata_commondata_tuple(
    data, closuretest_commondata_tuple, replica_seed
):
    """
    Like `pseudodata_commondata_tuple` but with closure test (fake-data) central values.

    Returns
    -------
    tuple
        tuple of validphys.coredata.CommonData instances
    """
    return pseudodata_commondata_tuple(data, closuretest_commondata_tuple, replica_seed)


"""
Collect over multiple random seeds so as to generate multiple commondata instances.
To be used in a Monte Carlo fit to experimental data.
"""
mc_replicas_pseudodata_commondata_tuple = collect(
    "pseudodata_commondata_tuple", ("replica_indices",)
)

"""
Collect over multiple random seeds so as to generate multiple commondata instances.
To be used in a Monte Carlo closure test fit.
"""
mc_replicas_closuretest_pseudodata_commondata_tuple = collect(
    "closuretest_pseudodata_commondata_tuple",
    ("replica_indices",),
)

"""
validphys actions should return immutable objects
"""


@dataclass(frozen=True)
class CentralCovmatIndex:
    central_values: jnp.array
    covmat: jnp.array
    central_values_idx: jnp.array


def central_covmat_index(commondata_tuple, covariance_matrix):
    """
    Given a commondata_tuple and a covariance_matrix, generated
    according to respective explicit node in config.py, store
    relevant data into CentralCovmatIndex dataclass.

    Parameters
    ----------
    commondata_tuple: tuple
        tuple of commondata instances, is generated as explicit node
        (see config.produce_commondata_tuple) and accordingly to the
        specified options (pseudodata, fakedata).

    covariance_matrix: jnp.array
        covariance matrix, is generated as explicit node
        (see config.covariance_matrix) can be either experimental
        or t0 covariance matrix depending on whether `use_t0` is
        True or False

    Returns
    -------
    CentralCovmatIndex dataclass
        dataclass containing central values, covariance matrix and
        index of central values
    """
    central_values = jnp.array(
        pd.concat([cd.central_values for cd in commondata_tuple], axis=0)
    )
    central_values_idx = jnp.arange(central_values.shape[0])

    return CentralCovmatIndex(
        central_values=central_values,
        central_values_idx=central_values_idx,
        covmat=covariance_matrix,
    )


"""
Collect central_covmat_index over multiple replicas
"""
mc_central_covmat_index = collect("central_covmat_index", ("replica_indices",))
