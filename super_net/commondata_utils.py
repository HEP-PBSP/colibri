"""
super_net.commondata_utils.py

Module containing commondata and central covmat index functions.

Author: Mark N. Costantini
Date: 11.11.2023
"""

import pandas as pd
from dataclasses import dataclass, asdict

import jax
import jax.numpy as jnp

from super_net.theory_predictions import make_pred_dataset


from reportengine import collect


def experimental_commondata_tuple(data):
    """
    returns a tuple (validphys nodes should be immutable)
    of commondata instances with experimental central values

    Parameters
    ----------
    data: super_net.core.SuperNetDataGroupSpec

    Returns
    -------
    tuple
        tuple of validphys.coredata.CommonData instances
    """
    return tuple(data.load_commondata_instance())


def level_0_commondata_tuple(
    data,
    experimental_commondata_tuple,
    closure_test_central_pdf_grid,
    flavour_indices=None,
):
    """
    returns a tuple (validphys nodes should be immutable)
    of commondata instances with experimental central values
    replaced with theory predictions computed from a PDF `closure_test_pdf`
    and fktables corresponding to datasets within data

    Parameters
    ----------
    data: super_net.core.SuperNetDataGroupSpec

    experimental_commondata_tuple: tuple
        tuple of commondata with experimental central values

    closure_test_central_pdf_grid: jnp.array
        grid is of shape N_fl x N_x

    Returns
    -------
    tuple
        tuple of validphys.coredata.CommonData instances
    """

    fake_data = []
    for cd, ds in zip(experimental_commondata_tuple, data.datasets):
        if cd.setname != ds.name:
            raise RuntimeError(f"commondata {cd} does not correspond to dataset {ds}")
        # replace central values with theory prediction from `closure_test_pdf`
        fake_data.append(
            cd.with_central_value(
                make_pred_dataset(ds, flavour_indices=flavour_indices)(
                    closure_test_central_pdf_grid
                )
            )
        )
    return tuple(fake_data)


def level_1_commondata_tuple(
    level_0_commondata_tuple,
    data_generation_covariance_matrix,
    level_1_seed=123456,
):
    """
    returns a tuple (validphys nodes should be immutable)
    of commondata instances with experimental central values
    replaced with theory predictions computed from a PDF `closure_test_pdf`
    and fktables corresponding to datasets within data

    Noise is added on top of the central values according to a
    multivariate Gaussian with covariance data_generation_covariance_matrix

    Parameters
    ----------
    level_0_commondata_tuple: tuple of validphys.coredata.CommonData instances
        A tuple of level_0 closure test data.

    data_generation_covariance_matrix: jnp.array
        The covariance matrix used for data generation.

    level_1_seed: int
        The random seed from which the level_1 data is drawn.

    Returns
    -------
    tuple
        tuple of validphys.coredata.CommonData instances
    """

    # First, construct a jax array from the level_0_commondata_tuple
    central_values = jnp.array(
        pd.concat([cd.central_values for cd in level_0_commondata_tuple], axis=0)
    )

    # Now, sample from the multivariate Gaussian with central values central_values
    # and covariance matrix data_generation_covariance_matrix. This produces the
    # level_1 data.
    rng = jax.random.PRNGKey(level_1_seed)
    sample = jax.random.multivariate_normal(
        rng, central_values, data_generation_covariance_matrix
    )

    # Now, reconstruct the commondata tuple, by modifying the original commondata
    # tuple's central values.
    sample_list = []
    for cd in level_0_commondata_tuple:
        sample_list.append(cd.with_central_value(sample[: cd.ndata]))
        sample = sample[cd.ndata :]

    return tuple(sample_list)


@dataclass(frozen=True)
class CentralCovmatIndex:
    central_values: jnp.array
    covmat: jnp.array
    central_values_idx: jnp.array

    def to_dict(self):
        return asdict(self)


def central_covmat_index(commondata_tuple, fit_covariance_matrix):
    """
    Given a commondata_tuple and a covariance_matrix, generated
    according to respective explicit node in config.py, store
    relevant data into CentralCovmatIndex dataclass.

    Parameters
    ----------
    commondata_tuple: tuple
        tuple of commondata instances, is generated as explicit node
        (see config.produce_commondata_tuple) and accordingly to the
        specified options.

    fit_covariance_matrix: jnp.array
        covariance matrix, is generated as explicit node
        (see config.fit_covariance_matrix) can be either experimental
        or t0 covariance matrix depending on whether `use_fit_t0` is
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
        covmat=fit_covariance_matrix,
    )


def pseudodata_central_covmat_index(
    commondata_tuple, data_generation_covariance_matrix
):
    """Same as central_covmat_index, but with the pseudodata generation
    covariance matrix for a Monte Carlo fit.
    """
    return central_covmat_index(commondata_tuple, data_generation_covariance_matrix)
