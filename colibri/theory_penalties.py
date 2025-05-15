"""
colibri.theory_penalties.py

Module for the Lagrange penalties terms added to the Likelihood / Loss function.
"""

import jax
import jax.numpy as jnp

from colibri.theory_predictions import pred_funcs_from_dataset, OP
from colibri.utils import mask_fktable_array, closest_indices

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

    flavour_indices: list
        list of indices of the flavours to be considered in the fit.

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

    pred_funcs = pred_funcs_from_dataset(
        posdataset, FIT_XGRID, flavour_indices, fill_fk_xgrid_with_zeros=False
    )

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

    flavour_indices: list
        list of indices of the flavours to be considered in
        the fit.

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


def integrability_penalty(integrability_settings, FIT_XGRID):
    """
    Compute the integrability penalty to be added to the loss function.

    Parameters
    ----------
    integrability_settings: colibri.core.IntegrabilitySettings dataclass

    Returns
    -------
    Callable
        A function that takes the pdf grid (x f(x)) and returns the integrability penalty term.
    """

    if not integrability_settings.integrability:
        return lambda pdf: jnp.array([0])

    # only select the subset of specified flavours
    integ_flavours = jnp.array(
        integrability_settings.integrability_specs["evolution_flavours"]
    )

    integ_xgrid = integrability_settings.integrability_specs["integrability_xgrid"]
    # ensure that the integrability xgrid points are included in the range of the fit xgrid
    if any([x > FIT_XGRID[-1] or x < FIT_XGRID[0] for x in integ_xgrid]):
        raise ValueError(
            f"Integrability xgrid points are not included in the range of the fit xgrid, choose xgrid points within {FIT_XGRID[0]} and {FIT_XGRID[-1]}."
        )

    # select
    x_idxs = closest_indices(jnp.array(FIT_XGRID), jnp.array(integ_xgrid))

    lambda_integrability = integrability_settings.integrability_specs[
        "lambda_integrability"
    ]

    def integ_penalty(
        pdf,
    ):
        """
        Computes the penalty due to integrability.

        Parameters
        ----------
        pdf: jnp.array of shape (Nfl, Nx)

        Returns
        -------
        jnp.array of shape (len(integ_flavours), )
        """
        integ_pdf_grid = pdf[integ_flavours, :][:, x_idxs]
        # compute integrability penalty term and sum over xgrid points
        penalty = lambda_integrability * jnp.sum((integ_pdf_grid) ** 2, axis=1)

        return penalty

    return integ_penalty
