"""
super_net.theory_predictions.py

This module contains the functions necessary for the computation of
theory predictions by means of fast-kernel (FK) tables.

Author: Mark N. Costantini
Date: 11.11.2023
"""

import jax
import jax.numpy as jnp

from validphys import convolution
from validphys.fkparser import load_fktable


# Is this needed? -> probably no need to jit compile
OP = {key: jax.jit(val) for key, val in convolution.OP.items()}


def make_dis_prediction(fktable, vectorized=False):
    """
    given an FKTableData instance returns a jax.jit
    compiled function taking a pdf grid as input
    and returning a theory prediction for a DIS
    observable.

    Parameters
    ----------
    fktable : validphys.coredata.FKTableData

    vectorized: bool, default is False
        if True, the function will be compiled in a way
        that allows to compute the prediction for multiple
        prior samples at once.

    Returns
    -------
    @jax.jit CompiledFunction
    """
    indices = fktable.luminosity_mapping
    fk_arr = jnp.array(fktable.get_np_fktable())

    if vectorized:

        @jax.jit
        def dis_prediction(pdf):
            return jnp.einsum("ijk, rjk ->ri", fk_arr, pdf[:, indices, :])

    else:

        @jax.jit
        def dis_prediction(pdf):
            return jnp.einsum("ijk, jk ->i", fk_arr, pdf[indices, :])

    return dis_prediction


def make_had_prediction(fktable, vectorized=False):
    """
    given an FKTableData instance returns a jax.jit
    compiled function taking a pdf grid as input
    and returning a theory prediction for a hadronic
    observable.

    Parameters
    ----------
    fktable : validphys.coredata.FKTableData

    vectorized: bool, default is False
        if True, the function will be compiled in a way
        that allows to compute the prediction for multiple
        prior samples at once.

    Returns
    -------
    @jax.jit CompiledFunction
    """

    indices = fktable.luminosity_mapping
    first_indices = indices[0::2]
    second_indices = indices[1::2]
    fk_arr = jnp.array(fktable.get_np_fktable())

    if vectorized:

        @jax.jit
        def had_prediction(pdf):
            return jnp.einsum(
                "ijkl,rjk,rjl->ri",
                fk_arr,
                pdf[:, first_indices, :],
                pdf[:, second_indices, :],
            )

    else:

        @jax.jit
        def had_prediction(pdf):
            return jnp.einsum(
                "ijkl,jk,jl->i", fk_arr, pdf[first_indices, :], pdf[second_indices, :]
            )

    return had_prediction


def make_pred_dataset(dataset, vectorized=False):
    """
    Compute theory prediction for a DataSetSpec

    Parameters
    ----------
    dataset : validphys.core.DataSetSpec

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
            pred = make_had_prediction(fk, vectorized)
        else:
            pred = make_dis_prediction(fk, vectorized)
        pred_funcs.append(pred)

    @jax.jit
    def prediction(pdf):
        return OP[dataset.op](*[f(pdf) for f in pred_funcs])

    return prediction


def make_pred_data(data, vectorized=False):
    """
    Compute theory prediction for entire DataGroupSpec

    Parameters
    ----------
    data: DataGroupSpec instance

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
        predictions.append(make_pred_dataset(ds, vectorized))

    @jax.jit
    def eval_preds(pdf):
        return jnp.concatenate([f(pdf) for f in predictions], axis=-1)

    return eval_preds


def make_pred_data_non_vectorized(data):
    """
    Same as make_pred_data but with vectorized=False
    """
    return make_pred_data(data, vectorized=False)


def make_penalty_posdataset(posdataset, vectorized=False):
    """
    Given a PositivitySetSpec compute the positivity penalty
    as a lagrange multiplier times elu of minus the theory prediction

    Parameters
    ----------
    posdataset : validphys.core.PositivitySetSpec

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
            pred = make_had_prediction(fk, vectorized)
        else:
            pred = make_dis_prediction(fk, vectorized)
        pred_funcs.append(pred)

    @jax.jit
    def pos_penalty(pdf, alpha, lambda_positivity):
        return lambda_positivity * jax.nn.elu(
            -OP[posdataset.op](*[f(pdf) for f in pred_funcs]), alpha
        )

    return pos_penalty


def make_penalty_posdata(posdatasets, vectorized=False):
    """
    Compute positivity penalty for list of PositivitySetSpec

    Parameters
    ----------
    posdatasets: list
            list of PositivitySetSpec

    vectorized: bool, default is False

    Returns
    -------
    @jax.jit CompiledFunction

    """

    predictions = []

    for posdataset in posdatasets:
        predictions.append(make_penalty_posdataset(posdataset, vectorized))

    @jax.jit
    def pos_penalties(pdf, alpha, lambda_positivity):
        return jnp.concatenate(
            [f(pdf, alpha, lambda_positivity) for f in predictions], axis=-1
        )

    return pos_penalties
