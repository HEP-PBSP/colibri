"""
colibri.theory_predictions.py

This module contains the functions necessary for the computation of
theory predictions by means of fast-kernel (FK) tables.
"""

import jax
import jax.numpy as jnp
import numpy as np

from validphys import convolution
from validphys.fkparser import load_fktable
from colibri.utils import mask_fktable_array, mask_luminosity_mapping, closest_indices

# Is this needed? -> probably no need to jit compile
OP = {key: jax.jit(val) for key, val in convolution.OP.items()}


def fktable_xgrid_indices(fktable, FIT_XGRID, fill_fk_xgrid_with_zeros=False):
    """
    Given an FKTableData instance and the xgrid used in the fit returns
    the indices of the xgrid of the FK table in the xgrid of the fit.

    If fill_fk_xgrid_with_zeros is True, then the all indices of the fit xgrid
    are returned. This is useful when the FK table is needed as tensor
    of shape (Ndat, Nfl, Nfk_xgrid) with Nfk_xgrid and Nfl fixed for all datasets.

    Parameters
    ----------
    fktable : validphys.coredata.FKTableData

    FIT_XGRID: jnp.ndarray
        array of xgrid points of the theory entering the fit

    fill_fk_xgrid_with_zeros: bool, default is False

    Returns
    -------
    jnp.ndarray of indices
    """
    if fill_fk_xgrid_with_zeros:
        return jnp.arange(len(FIT_XGRID))

    # Extract xgrid of the FK table and find the indices
    fk_xgrid = fktable.xgrid
    # atol is chosen to be 1e-8 as this is the order of magnitude of the difference between the smallest entries of the XGRID
    fk_xgrid_indices = closest_indices(jnp.array(FIT_XGRID), fk_xgrid, atol=1e-8)

    return fk_xgrid_indices


def fast_kernel_arrays(
    data, FIT_XGRID, flavour_indices=None, fill_fk_xgrid_with_zeros=False
):
    """
    Returns a tuple of tuples of jax.numpy arrays.

    Parameters
    ----------
    data : validphys.core.DataGroupSpec

    FIT_XGRID: np.ndarray

    flavour_indices: list, default is None
        if not None, the function will return fk arrays
        that allow to compute the prediction for a subset
        of flavours. The list must contain the flavour indices.
        The indices correspond to the flavours in convolution.FK_FLAVOURS
        e.g.: [1,2] -> ['\\Sigma', 'g']

    fill_fk_xgrid_with_zeros: bool, default is False
        If True, then the missing xgrid points in the FK table
        will be filled with zeros. This is useful when the FK table
        is needed as tensor of shape (Ndat, Nfl, Nfk_xgrid) with Nfk_xgrid and Nfl fixed
        for all datasets.

    Returns
    -------
    tuple
        tuple of tuples of jax.numpy arrays
    """
    fk_arrays = []

    for ds in data.datasets:
        fk_dataset_arr = []
        for fkspec in ds.fkspecs:
            # load fktable
            fk = load_fktable(fkspec).with_cuts(ds.cuts)

            # get FK-array with masked flavours
            fk_arr = mask_fktable_array(fk, flavour_indices)

            if fill_fk_xgrid_with_zeros:
                # fill with zeros the Xgrid dimension of the FK table so as to have tensor of shape (Ndat, Nfl, Nfk_xgrid)
                fk_xgrid = fk.xgrid
                non_zero_indices = closest_indices(FIT_XGRID, fk_xgrid, atol=1e-8)
                new_fk_arr = np.zeros(
                    (fk_arr.shape[0], fk_arr.shape[1], len(FIT_XGRID))
                )
                new_fk_arr[:, :, non_zero_indices] = fk_arr
                fk_arr = jnp.array(new_fk_arr)

            fk_dataset_arr.append(fk_arr)
        fk_arrays.append(tuple(fk_dataset_arr))

    return tuple(fk_arrays)


def make_dis_prediction(
    fktable, FIT_XGRID, flavour_indices=None, fill_fk_xgrid_with_zeros=False
):
    """
    Closure to compute the theory prediction for a DIS observable.

    Parameters
    ----------
    fktable : validphys.coredata.FKTableData
            The fktable should be a validphys.coredata.FKTableData instance
            and with cuts and masked flavours already applied.

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    flavour_indices: list, default is None

    fill_fk_xgrid_with_zeros: bool, default is False
        If True, then the missing xgrid points in the FK table
        will be filled with zeros. This is useful when the FK table
        is needed as tensor of shape (Ndat, Nfl, Nfk_xgrid) with Nfk_xgrid and Nfl fixed
        for all datasets.

    Returns
    -------
    Callable
    """
    lumi_indices = mask_luminosity_mapping(fktable, flavour_indices)

    fk_xgrid_indices = fktable_xgrid_indices(
        fktable, FIT_XGRID, fill_fk_xgrid_with_zeros=fill_fk_xgrid_with_zeros
    )

    def dis_prediction(pdf, fk_arr):
        """
        Function to compute the theory prediction for a DIS observable.

        Note that when running an ultranest fit this function gets compiled by jax.jit,
        hence, ideally, this function should be pure.
        However, luminosity indices and fk_xgrid_indices don't take much memory
        and hence are left as global variables.
        For more details on jax.jit issues with non pure functions see e.g.
        https://github.com/google/jax/issues/5071

        Parameters
        ----------
        pdf: jnp.ndarray
            pdf grid (shape is 14,50)

        fk_arr: jnp.ndarray
            fktable array

        Returns
        -------
        jnp.ndarray
            theory prediction for a hadronic observable (shape is Ndata, )
        """
        return jnp.einsum(
            "ijk, jk ->i", fk_arr, pdf[lumi_indices, :][:, fk_xgrid_indices]
        )

    return dis_prediction


def make_had_prediction(
    fktable, FIT_XGRID, flavour_indices=None, fill_fk_xgrid_with_zeros=False
):
    """
    Closure to compute the theory prediction for a Hadronic observable.

    Parameters
    ----------
    fktable : validphys.coredata.FKTableData

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    flavour_indices: list, default is None

    fill_fk_xgrid_with_zeros: bool, default is False
        If True, then the missing xgrid points in the FK table
        will be filled with zeros. This is useful when the FK table
        is needed as tensor of shape (Ndat, Nfl, Nfk_xgrid) with Nfk_xgrid and Nfl fixed
        for all datasets.

    Returns
    -------
    Callable
    """
    lumi_indices = mask_luminosity_mapping(fktable, flavour_indices)
    first_lumi_indices = lumi_indices[0::2]
    second_lumi_indices = lumi_indices[1::2]

    fk_xgrid_indices = fktable_xgrid_indices(
        fktable, FIT_XGRID, fill_fk_xgrid_with_zeros=fill_fk_xgrid_with_zeros
    )

    def had_prediction(pdf, fk_arr):
        """
        Function to compute the theory prediction for a Hadronic observable.

        Note that when running an ultranest fit this function gets compiled by jax.jit,
        hence, ideally, this function should be pure.
        However, luminosity indices and fk_xgrid_indices don't take much memory
        and hence are left as global variables.
        For more details on jax.jit issues with non pure functions see e.g.
        https://github.com/google/jax/issues/5071

        Parameters
        ----------
        pdf: jnp.ndarray
            pdf grid (shape is 14,50)

        fk_arr: jnp.ndarray
            fktable array

        Returns
        -------
        jnp.ndarray
            theory prediction for a hadronic observable (shape is Ndata, )
        """
        return jnp.einsum(
            "ijkl,jk,jl->i",
            fk_arr,
            pdf[first_lumi_indices, :][:, fk_xgrid_indices],
            pdf[second_lumi_indices, :][:, fk_xgrid_indices],
        )

    return had_prediction


def pred_funcs_from_dataset(
    dataset, FIT_XGRID, flavour_indices, fill_fk_xgrid_with_zeros=False
):
    """
    Returns a list containing the forward maps associated with the fkspecs of a dataset.

    Parameters
    ----------
    dataset: validphys.core.DataGroupSpec

    FIT_XGRID: array

    flavour_indices: list, default is None

    fill_fk_xgrid_with_zeros: bool, default is False

    Returns
    -------
    list of Mappings
    """
    pred_funcs = []

    for fkspec in dataset.fkspecs:
        fk = load_fktable(fkspec).with_cuts(dataset.cuts)

        if fk.hadronic:
            pred = make_had_prediction(
                fk, FIT_XGRID, flavour_indices, fill_fk_xgrid_with_zeros
            )
        else:
            pred = make_dis_prediction(
                fk, FIT_XGRID, flavour_indices, fill_fk_xgrid_with_zeros
            )
        pred_funcs.append(pred)

    return pred_funcs


def make_pred_dataset(
    dataset, FIT_XGRID, flavour_indices=None, fill_fk_xgrid_with_zeros=False
):
    """
    Compute theory prediction for a DataSetSpec

    Parameters
    ----------
    dataset : validphys.core.DataSetSpec

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    flavour_indices: list, default is None

    fill_fk_xgrid_with_zeros: bool, default is False

    Returns
    -------
    Callable
    """

    pred_funcs = pred_funcs_from_dataset(
        dataset, FIT_XGRID, flavour_indices, fill_fk_xgrid_with_zeros
    )

    def prediction(pdf, fk_dataset):
        return OP[dataset.op](
            *[f(pdf, fk_arr) for (f, fk_arr) in zip(pred_funcs, fk_dataset)]
        )

    return prediction


def make_pred_data(
    data, FIT_XGRID, flavour_indices=None, fill_fk_xgrid_with_zeros=False
):
    """
    Compute theory prediction for entire DataGroupSpec

    Parameters
    ----------
    data: DataGroupSpec instance

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    flavour_indices: list, default is None

    fill_fk_xgrid_with_zeros: bool, default is False

    Returns
    -------
    Callable
    """

    predictions = []

    for ds in data.datasets:
        predictions.append(
            make_pred_dataset(
                ds,
                FIT_XGRID,
                flavour_indices,
                fill_fk_xgrid_with_zeros=fill_fk_xgrid_with_zeros,
            )
        )

    def eval_preds(pdf, fast_kernel_arrays):
        return jnp.concatenate(
            [
                f(pdf, fk_dataset)
                for f, fk_dataset in zip(predictions, fast_kernel_arrays)
            ],
            axis=-1,
        )

    return eval_preds


def make_pred_t0data(
    data, FIT_XGRID, flavour_indices=None, fill_fk_xgrid_with_zeros=False
):
    """
    Compute theory prediction for entire DataGroupSpec.
    It is specifically meant for t0 predictions, i.e. it
    is similar to dataset_t0_predictions in validphys.covmats.

    Parameters
    ----------
    data: DataGroupSpec instance

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    flavour_indices: list, default is None

    fill_fk_xgrid_with_zeros: bool, default is False

    Returns
    -------
    Callable
    """

    predictions = []

    for ds in data.datasets:
        predictions.append(
            make_pred_dataset(
                ds,
                FIT_XGRID,
                flavour_indices=flavour_indices,
                fill_fk_xgrid_with_zeros=fill_fk_xgrid_with_zeros,
            )
        )

    def eval_preds(pdf, fast_kernel_arrays):
        return [
            f(pdf, fk_dataset) for f, fk_dataset in zip(predictions, fast_kernel_arrays)
        ]

    return eval_preds
