"""
colibri.theory_predictions.py

This module contains the functions necessary for the computation of
theory predictions by means of fast-kernel (FK) tables.

Author: Mark N. Costantini
Date: 11.11.2023
"""

import jax
import jax.numpy as jnp

from validphys import convolution
from validphys.fkparser import load_fktable

from colibri.utils import fill_dis_fkarr_with_zeros, fill_had_fkarr_with_zeros

# Is this needed? -> probably no need to jit compile
OP = {key: jax.jit(val) for key, val in convolution.OP.items()}


def make_dis_prediction(fktable, FIT_XGRID, vectorized=False, flavour_indices=None):
    """
    Given an FKTableData instance returns a jax.jit
    compiled function taking a pdf grid as input
    and returning a theory prediction for a DIS
    observable.

    Parameters
    ----------
    fktable : validphys.coredata.FKTableData

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    vectorized: bool, default is False
        if True, the function will be compiled in a way
        that allows to compute the prediction for multiple
        prior samples at once.

    flavour_indices: list, default is None
        if not None, the function will be compiled in a way
        that allows to compute the prediction for a subset
        of flavours. The list must contain the flavour indices.
        The indices correspond to the flavours in convolution.FK_FLAVOURS
        e.g.: [1,2] -> ['\\Sigma', 'g']


    Returns
    -------
    @jax.jit CompiledFunction
    """

    if flavour_indices is not None:
        indices = fktable.luminosity_mapping
        mask = jnp.isin(indices, jnp.array(flavour_indices))
        indices = indices[mask]
        fk_arr = jnp.array(fill_dis_fkarr_with_zeros(fktable, FIT_XGRID))[:, mask, :]

    else:
        indices = fktable.luminosity_mapping
        fk_arr = jnp.array(fill_dis_fkarr_with_zeros(fktable, FIT_XGRID))

    @jax.jit
    def dis_prediction(pdf):
        return jnp.einsum("ijk, jk ->i", fk_arr, pdf[indices, :])

    if vectorized:
        return jnp.vectorize(dis_prediction, signature="(m,n)->(k)")
    return dis_prediction


def make_had_prediction(fktable, FIT_XGRID, vectorized=False, flavour_indices=None):
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

    vectorized: bool, default is False
        if True, the function will be compiled in a way
        that allows to compute the prediction for multiple
        prior samples at once.

    flavour_indices: list, default is None
        if not None, the function will be compiled in a way
        that allows to compute the prediction for a subset
        of flavours. The list must contain the flavour indices.
        The indices correspond to the flavours in convolution.FK_FLAVOURS
        e.g.: [1,2] -> ['\\Sigma', 'g']

    Returns
    -------
    @jax.jit CompiledFunction
    """

    if flavour_indices is not None:
        indices = fktable.luminosity_mapping
        mask_even = jnp.isin(indices[0::2], jnp.array(flavour_indices))
        mask_odd = jnp.isin(indices[1::2], jnp.array(flavour_indices))

        # for hadronic predictions pdfs enter in pair, hence product of two
        # boolean arrays and repeat by 2
        mask = jnp.repeat(mask_even * mask_odd, repeats=2)
        indices = indices[mask]

        first_indices = indices[0::2]
        second_indices = indices[1::2]

        fk_arr = jnp.array(fill_had_fkarr_with_zeros(fktable, FIT_XGRID))[
            :, mask_even * mask_odd, :, :
        ]

    else:
        indices = fktable.luminosity_mapping

        first_indices = indices[0::2]
        second_indices = indices[1::2]

        fk_arr = jnp.array(fill_had_fkarr_with_zeros(fktable, FIT_XGRID))

    @jax.jit
    def had_prediction(pdf):
        return jnp.einsum(
            "ijkl,jk,jl->i", fk_arr, pdf[first_indices, :], pdf[second_indices, :]
        )

    if vectorized:
        return jnp.vectorize(had_prediction, signature="(m,n)->(k)")
    return had_prediction


def make_pred_dataset(dataset, FIT_XGRID, vectorized=False, flavour_indices=None):
    """
    Compute theory prediction for a DataSetSpec

    Parameters
    ----------
    dataset : validphys.core.DataSetSpec

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    vectorized: bool, default is False

    Returns
    -------
    @jax.jit CompiledFunction
        Compiled function taking pdf grid in input
        and returning theory prediction for one
        dataset
    """

    pred_funcs = []

    for fkspec in dataset.fkspecs:
        fk = load_fktable(fkspec).with_cuts(dataset.cuts)
        if fk.hadronic:
            pred = make_had_prediction(fk, FIT_XGRID, vectorized, flavour_indices)
        else:
            pred = make_dis_prediction(fk, FIT_XGRID, vectorized, flavour_indices)
        pred_funcs.append(pred)

    @jax.jit
    def prediction(pdf):
        return OP[dataset.op](*[f(pdf) for f in pred_funcs])

    return prediction


def make_pred_data(data, FIT_XGRID, vectorized=False, flavour_indices=None):
    """
    Compute theory prediction for entire DataGroupSpec

    Parameters
    ----------
    data: DataGroupSpec instance

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.
    
    vectorized: bool, default is False

    Returns
    -------
    @jax.jit CompiledFunction
        Compiled function taking pdf grid in input
        and returning theory prediction for one
        data group
    """

    predictions = []

    for ds in data.datasets:
        predictions.append(make_pred_dataset(ds, FIT_XGRID, vectorized, flavour_indices))

    @jax.jit
    def eval_preds(pdf):
        return jnp.concatenate([f(pdf) for f in predictions], axis=-1)

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
            make_pred_dataset(ds, FIT_XGRID, vectorized=False, flavour_indices=flavour_indices)
        )

    @jax.jit
    def eval_preds(pdf):
        return [f(pdf) for f in predictions]

    return eval_preds


def make_pred_data_non_vectorized(data, FIT_XGRID):
    """
    Same as make_pred_data but with vectorized=False
    """
    return make_pred_data(data, FIT_XGRID, vectorized=False)


def make_penalty_posdataset(posdataset, FIT_XGRID, vectorized=False, flavour_indices=None):
    """
    Given a PositivitySetSpec compute the positivity penalty
    as a lagrange multiplier times elu of minus the theory prediction

    Parameters
    ----------
    posdataset : validphys.core.PositivitySetSpec

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    vectorized: bool, default is False

    Returns
    -------
    @jax.jit CompiledFunction
        Compiled function taking pdf grid and alpha parameter
        of jax.nn.elu function in input and returning
        elu function evaluated on minus the theory prediction

        Note: this is needed in order to compute the positivity
        loss function. Elu function is used to avoid a big discontinuity
        in the derivative at 0 when the lagrange multiplier is very big.

        In practice this function can produce results in the range (-alpha, inf)

        see also nnpdf.n3fit.src.layers.losses.LossPositivity

    """

    pred_funcs = []

    for fkspec in posdataset.fkspecs:
        fk = load_fktable(fkspec).with_cuts(posdataset.cuts)
        if fk.hadronic:
            pred = make_had_prediction(fk, FIT_XGRID, vectorized, flavour_indices)
        else:
            pred = make_dis_prediction(fk, FIT_XGRID, vectorized, flavour_indices)
        pred_funcs.append(pred)

    @jax.jit
    def pos_penalty(pdf, alpha, lambda_positivity):
        return lambda_positivity * jax.nn.elu(
            -OP[posdataset.op](*[f(pdf) for f in pred_funcs]), alpha
        )

    return pos_penalty


def make_penalty_posdata(posdatasets, FIT_XGRID, vectorized=False):
    """
    Compute positivity penalty for list of PositivitySetSpec

    Parameters
    ----------
    posdatasets: list
            list of PositivitySetSpec

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    vectorized: bool, default is False

    Returns
    -------
    @jax.jit CompiledFunction

    """

    predictions = []

    for posdataset in posdatasets:
        predictions.append(make_penalty_posdataset(posdataset, FIT_XGRID, vectorized))

    @jax.jit
    def pos_penalties(pdf, alpha, lambda_positivity):
        return jnp.concatenate(
            [f(pdf, alpha, lambda_positivity) for f in predictions], axis=-1
        )

    return pos_penalties
