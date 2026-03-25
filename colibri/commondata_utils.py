"""
colibri.commondata_utils.py

Module containing commondata and central covmat index functions.
"""

import pandas as pd

import jax
import jax.lax.linalg as jlinalg
import jax.numpy as jnp

from colibri.theory_predictions import make_pred_dataset
from colibri.core import CentralCovmatIndex


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
        Tuple of nnpdf_data.coredata.CommonData instances.
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

    closure_test_central_pdf_grid: jnp.ndarray
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
        Tuple of nnpdf_data.coredata.CommonData instances.
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
    general_covariance_matrix,
    level_1_seed=123456,
):
    """
    Returns a tuple (validphys nodes should be immutable)
    of level 1 commondata instances.
    Noise is added to the level_0_commondata_tuple central values
    according to a multivariate Gaussian with covariance general_covariance_matrix

    Parameters
    ----------
    level_0_commondata_tuple: tuple of nnpdf_data.coredata.CommonData instances
        A tuple of level_0 closure test data.

    general_covariance_matrix: jnp.ndarray
        The covariance matrix used for data generation.

    level_1_seed: int
        The random seed from which the level_1 data is drawn.

    Returns
    -------
    tuple
        Tuple of nnpdf_data.coredata.CommonData instances.
    """

    # First, construct a jax array from the level_0_commondata_tuple
    central_values = jnp.array(
        pd.concat([cd.central_values for cd in level_0_commondata_tuple], axis=0)
    )

    # Now, sample from the multivariate Gaussian with central values central_values
    # and general_covariance_matrix. This produces the
    # level_1 data.
    rng = jax.random.PRNGKey(level_1_seed)
    sample = jax.random.multivariate_normal(
        rng, central_values, general_covariance_matrix
    )

    # Now, reconstruct the commondata tuple, by modifying the original commondata
    # tuple's central values.
    sample_list = []
    for cd in level_0_commondata_tuple:
        sample_list.append(cd.with_central_value(sample[: cd.ndata]))
        sample = sample[cd.ndata :]

    return tuple(sample_list)


def central_covmat_index(commondata_tuple, general_sqrt_covariance_matrix):
    """
    Given a commondata_tuple and a general_sqrt_covariance_matrix, whiten the
    central values and store the inverse Cholesky factor in CentralCovmatIndex.

    The data are transformed to the whitened basis d_w = L^{-1} d (where L is
    ``general_sqrt_covariance_matrix``), making them i.i.d. before any
    training/validation split is applied.

    Parameters
    ----------
    commondata_tuple: tuple
        tuple of commondata instances, is generated as explicit node
        (see config.produce_commondata_tuple) and accordingly to the
        specified options.

    general_sqrt_covariance_matrix: jnp.ndarray
        lower triangular Cholesky factor L of the covariance matrix, generated
        as explicit node (see covmats.general_sqrt_covariance_matrix). Satisfies:
        ``general_sqrt_covariance_matrix @ general_sqrt_covariance_matrix.T == covmat``

    Returns
    -------
    CentralCovmatIndex
        Dataclass containing whitened central values, the inverse Cholesky
        factor L^{-1} (``inv_sqrt_covmat``), and the index of central values.
    """
    central_values = jnp.array(
        pd.concat([cd.central_values for cd in commondata_tuple], axis=0)
    )
    n = general_sqrt_covariance_matrix.shape[0]
    L_inv = jlinalg.triangular_solve(
        general_sqrt_covariance_matrix, jnp.eye(n), left_side=True, lower=True
    )
    whitened_central_values = L_inv @ central_values
    central_values_idx = jnp.arange(n)

    return CentralCovmatIndex(
        central_values=whitened_central_values,
        central_values_idx=central_values_idx,
        inv_sqrt_covmat=L_inv,
    )
