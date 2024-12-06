"""
colibri.theory_penalties.py

Module for the Lagrange penalties terms added to the Likelihood / Loss function.
"""

import jax
import jax.numpy as jnp

from colibri.theory_predictions import pred_funcs_from_dataset, OP
from colibri.utils import mask_fktable_array, closest_indices
from colibri.constants import XGRID

from validphys.fkparser import load_fktable


def positivity_fast_kernel_arrays(posdatasets, flavour_indices=None):
    """
    Similar to fast_kernel_arrays but for Positivity datasets.
    """
    pos_fk_arrays = []

    for posdataset in posdatasets:
        fk_dataset_arr = []
        for fkspec in posdataset.fkspecs:
            # load fktable and apply flavour mask
            fk = load_fktable(fkspec).with_cuts(posdataset.cuts)

            # get FK-array with masked flavours
            fk_arr = mask_fktable_array(fk, flavour_indices)

            fk_dataset_arr.append(fk_arr)
        pos_fk_arrays.append(tuple(fk_dataset_arr))

    return tuple(pos_fk_arrays)


def make_penalty_posdataset(posdataset, FIT_XGRID, flavour_indices=None):
    """
    Given a PositivitySetSpec compute the positivity penalty
    as a lagrange multiplier times elu of minus the theory prediction

    Parameters
    ----------
    posdataset : validphys.core.PositivitySetSpec

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

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

    pred_funcs = pred_funcs_from_dataset(posdataset, FIT_XGRID, flavour_indices)

    def pos_penalty(pdf, alpha, lambda_positivity, fk_dataset):
        return lambda_positivity * jax.nn.elu(
            -OP[posdataset.op](
                *[f(pdf, fk_arr) for (f, fk_arr) in zip(pred_funcs, fk_dataset)]
            ),
            alpha,
        )

    return pos_penalty


def make_penalty_posdata(posdatasets, FIT_XGRID, flavour_indices=None):
    """
    Compute positivity penalty for list of PositivitySetSpec

    Parameters
    ----------
    posdatasets: list
            list of PositivitySetSpec

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    Returns
    -------
    @jax.jit CompiledFunction

    """

    predictions = []

    for posdataset in posdatasets:
        predictions.append(
            make_penalty_posdataset(posdataset, FIT_XGRID, flavour_indices)
        )

    def pos_penalties(pdf, alpha, lambda_positivity, fast_kernel_arrays):
        return jnp.concatenate(
            [
                f(pdf, alpha, lambda_positivity, fk_dataset)
                for (f, fk_dataset) in zip(predictions, fast_kernel_arrays)
            ],
            axis=-1,
        )

    return pos_penalties


def integrability_penalty(pdf_grid, integrability_settings, FIT_XGRID):
    """
    Compute the integrability penalty to be added to the loss function.

    Parameters
    ----------
    pdf_grid: jnp.array of shape (14, 50)

    integrability_settings: colibri.core.IntegrabilitySettings dataclass

    FIT_XGRID: jnp array
        contains the xgrid used in the fit.

    Returns
    -------
    """
    # only select a subset of flavours
    integ_flavours = jnp.array(
        integrability_settings.integrability_specs["evolution_flavours"]
    )
    integ_pdf_grid = pdf_grid[integ_flavours]

    # only select the smallest xgrid point of FIT_XGRID to impose Integrability on
    x_idx = closest_indices(jnp.array(XGRID), FIT_XGRID[0])
    integ_pdf_grid = integ_pdf_grid[:, x_idx]

    # compute integrability penalty term
    penalty = integrability_settings.integrability_specs[
        "lambda_integrability"
    ] * jnp.sum((FIT_XGRID[0] * (integ_pdf_grid)) ** 2)

    return penalty
