"""
colibri.commondata_utils.py

Module containing commondata and central covmat index functions.
"""

import pandas as pd
from dataclasses import dataclass, asdict

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from validphys.fkparser import load_fktable

from colibri.theory_predictions import make_pred_dataset


def experimental_commondata_tuple(data):
    """
    Returns a tuple (validphys nodes should be immutable)
    of commondata instances with experimental central values.

    Parameters
    ----------
    data: validphys.core.DataGroupSpec

    Returns
    -------
    tuple
        tuple of nnpdf_data.coredata.CommonData instances
    """
    return tuple(data.load_commondata_instance())


def level_0_commondata_tuple(
    data,
    experimental_commondata_tuple,
    closure_test_central_pdf_grid,
    FIT_XGRID,
    fast_kernel_arrays,
    flavour_indices=None,
    fill_fk_xgrid_with_zeros=False,
):
    """
    Returns a tuple (validphys nodes should be immutable)
    of commondata instances with experimental central values
    replaced with theory predictions computed from a PDF `closure_test_pdf`
    and fktables corresponding to datasets within data.

    Parameters
    ----------
    data: validphys.core.DataGroupSpec

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    experimental_commondata_tuple: tuple
        tuple of commondata with experimental central values

    closure_test_central_pdf_grid: jnp.array
        grid is of shape N_fl x N_x

    fast_kernel_arrays: tuple
        tuple of jnp.array of shape (Ndat, Nfl, Nfk_xgrid)
        containing the fast kernel arrays for each dataset in data.

    flavour_indices: list, default is None
        Subset of flavour (evolution basis) indices to be used.

    fill_fk_xgrid_with_zeros: bool, default is False
        If True, then the missing xgrid points in the FK table
        will be filled with zeros. This is useful when the FK table
        is needed as tensor of shape (Ndat, Nfl, Nfk_xgrid) with Nfk_xgrid and Nfl fixed
        for all datasets.


    Returns
    -------
    tuple
        tuple of nnpdf_data.coredata.CommonData instances
    """

    fake_data = []
    for cd, ds, fk_dataset in zip(
        experimental_commondata_tuple, data.datasets, fast_kernel_arrays
    ):
        if cd.setname != ds.name:
            raise RuntimeError(f"commondata {cd} does not correspond to dataset {ds}")
        # replace central values with theory prediction from `closure_test_pdf`
        fake_data.append(
            cd.with_central_value(
                make_pred_dataset(
                    ds,
                    FIT_XGRID,
                    flavour_indices=flavour_indices,
                    fill_fk_xgrid_with_zeros=fill_fk_xgrid_with_zeros,
                )(closure_test_central_pdf_grid, fk_dataset)
            )
        )
    return tuple(fake_data)


def level_1_commondata_tuple(
    level_0_commondata_tuple,
    data_generation_covariance_matrix,
    level_1_seed=123456,
):
    """
    Returns a tuple (validphys nodes should be immutable)
    of level 1 commondata instances.
    Noise is added to the level_0_commondata_tuple central values
    according to a multivariate Gaussian with covariance data_generation_covariance_matrix

    Parameters
    ----------
    level_0_commondata_tuple: tuple of nnpdf_data.coredata.CommonData instances
        A tuple of level_0 closure test data.

    data_generation_covariance_matrix: jnp.array
        The covariance matrix used for data generation.

    level_1_seed: int
        The random seed from which the level_1 data is drawn.

    Returns
    -------
    tuple
        tuple of nnpdf_data.coredata.CommonData instances
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


@dataclass(frozen=True)
class CentralInvCovmatIndex:
    central_values: jnp.array
    inv_covmat: jnp.array
    central_values_idx: jnp.array

    def to_dict(self):
        return asdict(self)


def central_inv_covmat_index(central_covmat_index):
    """
    Given a CentralCovmatIndex dataclass, compute the inverse
    of the covariance matrix and store the relevant data into
    CentralInvCovmatIndex dataclass.
    """
    inv_covmat = jla.inv(central_covmat_index.covmat)
    return CentralInvCovmatIndex(
        central_values=central_covmat_index.central_values,
        central_values_idx=central_covmat_index.central_values_idx,
        inv_covmat=inv_covmat,
    )
