"""
colibri.theory_predictions.py

This module contains the functions necessary for the computation of
theory predictions by means of fast-kernel (FK) tables.
"""

import jax
import jax.numpy as jnp

from validphys import convolution
from validphys.fkparser import load_fktable
from colibri.utils import mask_fktable_array, mask_luminosity_mapping, closest_indices

# Is this needed? -> probably no need to jit compile
OP = {key: jax.jit(val) for key, val in convolution.OP.items()}


def fast_kernel_arrays(data, flavour_indices=None):
    """
    Returns a tuple of tuples of jax.numpy arrays.

    Parameters
    ----------
    data : validphys.core.DataGroupSpec

    flavour_indices: list, default is None
        if not None, the function will return fk arrays
        that allow to compute the prediction for a subset
        of flavours. The list must contain the flavour indices.
        The indices correspond to the flavours in convolution.FK_FLAVOURS
        e.g.: [1,2] -> ['\\Sigma', 'g']

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

            fk_dataset_arr.append(fk_arr)
        fk_arrays.append(tuple(fk_dataset_arr))

    return tuple(fk_arrays)


def make_dis_prediction(fktable, FIT_XGRID, flavour_indices=None):
    """
    Given an FKTableData instance returns a jax.jit
    compiled function taking a pdf grid as input
    and returning a theory prediction for a DIS
    observable.

    Parameters
    ----------
    fktable : colibri.coredata.FKTableData
            The fktable should be a colibri.coredata.FKTableData instance
            and with cuts and masked flavours already applied.

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.


    Returns
    -------
    @jax.jit CompiledFunction
    """
    lumi_indices = mask_luminosity_mapping(fktable, flavour_indices)

    # Extract xgrid of the FK table and find the indices
    fk_xgrid = fktable.xgrid
    # atol is chosen to be default = 1e-8 as this is the order of magnitude of the difference between the smallest entries of the XGRID
    fk_xgrid_indices = closest_indices(FIT_XGRID, fk_xgrid)

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


def make_had_prediction(fktable, FIT_XGRID, flavour_indices=None):
    """
    Given an FKTableData instance returns a jax.jit
    compiled function taking a pdf grid as input
    and returning a theory prediction for a hadronic
    observable.

    Parameters
    ----------
    fktable : validphys.coredata.FKTableData

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    flavour_indices: list, default is None

    Returns
    -------
    @jax.jit CompiledFunction
    """
    lumi_indices = mask_luminosity_mapping(fktable, flavour_indices)
    first_lumi_indices = lumi_indices[0::2]
    second_lumi_indices = lumi_indices[1::2]

    # Extract xgrid of the FK table and find the indices
    fk_xgrid = fktable.xgrid
    # atol is chosen to be default = 1e-8 as this is the order of magnitude of the difference between the smallest entries of the XGRID
    fk_xgrid_indices = closest_indices(FIT_XGRID, fk_xgrid)

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


def pred_funcs_from_dataset(dataset, FIT_XGRID, flavour_indices):
    """
    Returns a list containing the forward maps associated with the fkspecs of a dataset.

    Parameters
    ----------
    dataset: validphys.core.DataGroupSpec

    FIT_XGRID: array

    flavour_indices: list, default is None

    Returns
    -------
    list of Mappings
    """
    pred_funcs = []

    for fkspec in dataset.fkspecs:
        fk = load_fktable(fkspec).with_cuts(dataset.cuts)

        if fk.hadronic:
            pred = make_had_prediction(fk, FIT_XGRID, flavour_indices)
        else:
            pred = make_dis_prediction(fk, FIT_XGRID, flavour_indices)
        pred_funcs.append(pred)

    return pred_funcs


def make_pred_dataset(dataset, FIT_XGRID, flavour_indices=None):
    """
    Compute theory prediction for a DataSetSpec

    Parameters
    ----------
    dataset : validphys.core.DataSetSpec

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    flavour_indices: list, default is None

    Returns
    -------
    @jax.jit CompiledFunction
        Compiled function taking pdf grid in input
        and returning theory prediction for one
        dataset
    """

    pred_funcs = pred_funcs_from_dataset(dataset, FIT_XGRID, flavour_indices)

    def prediction(pdf, fk_dataset):
        return OP[dataset.op](
            *[f(pdf, fk_arr) for (f, fk_arr) in zip(pred_funcs, fk_dataset)]
        )

    return prediction


def make_pred_data(data, FIT_XGRID, flavour_indices=None):
    """
    Compute theory prediction for entire DataGroupSpec

    Parameters
    ----------
    data: DataGroupSpec instance

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    flavour_indices: list, default is None

    Returns
    -------
    @jax.jit CompiledFunction
        Compiled function taking pdf grid in input
        and returning theory prediction for one
        data group
    """

    predictions = []

    for ds in data.datasets:
        predictions.append(make_pred_dataset(ds, FIT_XGRID, flavour_indices))

    def eval_preds(pdf, fast_kernel_arrays):
        return jnp.concatenate(
            [
                f(pdf, fk_dataset)
                for f, fk_dataset in zip(predictions, fast_kernel_arrays)
            ],
            axis=-1,
        )

    return eval_preds


def make_pred_t0data(data, FIT_XGRID, flavour_indices=None):
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

    Returns
    -------
    @jax.jit CompiledFunction
        Compiled function taking pdf grid in input
        and returning theory prediction for one
        data group
    """

    predictions = []

    for ds in data.datasets:
        predictions.append(
            make_pred_dataset(ds, FIT_XGRID, flavour_indices=flavour_indices)
        )

    def eval_preds(pdf, fast_kernel_arrays):
        return [
            f(pdf, fk_dataset) for f, fk_dataset in zip(predictions, fast_kernel_arrays)
        ]

    return eval_preds
