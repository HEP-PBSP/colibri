"""
Note: so far this only works when the reduced grids are of the 
same shape for each flavour.

The problem in using different sizes for different flavours is in the generation
of the PDF covariance matrix:

replicas_grid = convolution.evolution.grid_values(
        pdf_prior,
        [convolution.FK_FLAVOURS[fl] for fl in flavour_mapping],
        REDUCED_XGRIDS[fl],
        [Q0],
    )
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass

from validphys import convolution

from super_net.constants import XGRID


FLAVOURS_ID_MAPPINGS = {
    0: "photon",
    1: "\Sigma",
    2: "g",
    3: "V",
    4: "V3",
    5: "V8",
    6: "V15",
    7: "V24",
    8: "V35",
    9: "T3",
    10: "T8",
    11: "T15",
    12: "T24",
    13: "T35",
}

FLAVOUR_TO_ID_MAPPING = {val: key for (key, val) in FLAVOURS_ID_MAPPINGS.items()}

"""
Specifies which flavours to include in a fit.
"""
FLAVOUR_MAPPING = [1, 2, 3]


def interpolate_grid(
    reduced_xgrids,
    length_reduced_xgrids,
    flavour_mapping=FLAVOUR_MAPPING,
    vectorised=False,
):
    """
    Produces the function which produces the grid interpolation.
    """

    fit_xgrids = jnp.array([jnp.array(reduced_xgrids[fl]) for fl in flavour_mapping])

    if vectorised:
        # Function to perform interpolation for a single grid
        @jax.jit
        def interpolate_flavors(y):
            reshaped_y = y.reshape(
                (
                    len(flavour_mapping),
                    length_reduced_xgrids,
                )
            )
            out = jnp.array(
                            [
                                jnp.interp(jnp.array(XGRID), xgrid, reshaped_y[i, :])
                                for i, xgrid in enumerate(fit_xgrids)
                            ]
                        )
            return out

        @jax.jit
        def interp_func(stacked_pdf_grid):
            # generate an empty matrix of shape (:, valipdhys.convolution.NFK, len(super_net.constants.XGRID),)
            input_grid = jnp.zeros(
                (
                    stacked_pdf_grid.shape[0],
                    convolution.NFK,
                    len(XGRID),
                )
            )

            pdf_interp = jnp.apply_along_axis(
                interpolate_flavors,
                axis=-1,
                arr=stacked_pdf_grid,
            )

            input_grid = input_grid.at[:, flavour_mapping, :].set(pdf_interp)

            return input_grid

    else:

        @jax.jit
        def interp_func(stacked_pdf_grid):
            reshaped_stacked_pdf_grid = stacked_pdf_grid.reshape(
                (
                    len(flavour_mapping),
                    length_reduced_xgrids,
                ),
            )

            # generate an empty matrix of shape (valipdhys.convolution.NFK, len(super_net.constants.XGRID),)
            input_grid = jnp.zeros(
                (
                    convolution.NFK,
                    len(XGRID),
                )
            )

            # Loop to perform interpolation of the 2D arrays
            # The JIT compilation flattens the loop, good efficiency
            pdf_interp = []
            for i, xgrid in enumerate(fit_xgrids):
                pdf_interp.append(
                    jnp.interp(jnp.array(XGRID), xgrid, reshaped_stacked_pdf_grid[i, :])
                )

            pdf_interp = jnp.array(pdf_interp)

            input_grid = input_grid.at[flavour_mapping, :].set(pdf_interp)

            return input_grid

    return interp_func


@dataclass(frozen=True)
class PdfPriorGrid:
    """
    TODO
    """

    stacked_pdf_grid_prior: jnp.array
    pdf_covmat_prior: jnp.array
    error68_up: jnp.array
    error68_down: jnp.array


def pdf_prior_grid(pdf_prior, reduced_xgrids, flavour_mapping=FLAVOUR_MAPPING, Q0=1.65):
    """
    Get PDF grid prior values (x*f(x)) from a PDF set.

    Parameters
    ----------
    pdf_prior: validphys.core.PDF
        pdf set from which to get prior values.

    flavour_mapping: list, default is FLAVOUR_MAPPING
        specifies the ids of the flavours to include in a fit.

    Q0: float, default is 1.65
        specifies the scale at which PDFs are parameterised.

    Returns
    -------
    PdfPriorGrid dataclass with the following attributes:
        stacked_pdf_grid_prior: jnp.array
        pdf_covmat_prior: jnp.array
        error68_up: jnp.array
        error68_down: jnp.array
    """

    stacked_pdf_grid_prior = jnp.array([])

    # note: different flavours might have different REDUCED_XGRID
    for fl in flavour_mapping:
        # Save central value of pdf_prior at Q0
        stacked_pdf_grid_prior = jnp.append(
            stacked_pdf_grid_prior,
            jnp.array(
                convolution.evolution.grid_values(
                    pdf_prior, [convolution.FK_FLAVOURS[fl]], reduced_xgrids[fl], [Q0]
                )
            ).squeeze(-1)[0],
        )

    # generate PDF covariance matrix of size Nx * Nfl x Nx * Nfl
    replicas_grid = convolution.evolution.grid_values(
        pdf_prior,
        [convolution.FK_FLAVOURS[fl] for fl in flavour_mapping],
        reduced_xgrids[fl],
        [Q0],
    )

    error68_up = jnp.nanpercentile(replicas_grid, 84.13, axis=0).reshape(-1)
    error68_down = jnp.nanpercentile(replicas_grid, 15.87, axis=0).reshape(-1)

    pdf_covmat_prior = jnp.cov(replicas_grid.reshape((replicas_grid.shape[0], -1)).T)

    return PdfPriorGrid(
        stacked_pdf_grid_prior=stacked_pdf_grid_prior,
        pdf_covmat_prior=pdf_covmat_prior,
        error68_up=error68_up,
        error68_down=error68_down,
    )


def grid_pdf_model_prior(
    pdf_prior_grid,
    uniform_pdf_prior=True,
    gaussian_pdf_prior=False,
    sigma_pdf_prior=1,
):
    """
    This function returns a prior transform for the ultranest sampler.

    1. if uniform_pdf_prior is True, return a grid whose values are uniformly distribued
       in the 68% confidence interval of the central value of pdf_prior.

    2. if gaussian_pdf_prior is True, return a grid centered at the central value of the
       pdf_prior and with a covariance matrix given by the diagonal entries of covariance matrix
       of the pdf_prior


    Note:
        - The 68% confidence interval is computed from the replicas of the pdf_prior
        - TODO: the 68% confidence interval should be modifiedable by the user


    Parameters
    ----------
    pdf_prior_grid: PdfPriorGrid dataclass
        contains the prior pdf grid values and the upper and lower 68% confidence intervals

    """
    # This is needed to avoid ultranest crashing with
    # ValueError: Buffer dtype mismatch, expected 'float_t' but got 'float'
    jax.config.update("jax_enable_x64", True)

    stacked_pdf_grid_prior = pdf_prior_grid.stacked_pdf_grid_prior

    if gaussian_pdf_prior:
        pdf_covmat_prior = pdf_prior_grid.pdf_covmat_prior

        # note: this matrix could be ill-conditioned when too many xgrids are used
        # this is because xgrid values are not independent
        # in order to avoid this we  only keep the diagonal of the covariance matrix
        # and regularise it by cutting off small values
        pdf_diag_covmat_prior = jnp.where(
            jnp.diag(pdf_covmat_prior) < 0, 0, jnp.diag(pdf_covmat_prior)
        )
        cholesky_pdf_covmat = jnp.diag(jnp.sqrt(pdf_diag_covmat_prior))

        @jax.jit
        def prior_transform(cube):
            """
            This currently does not support vectorisation.
            """
            # generate independent gaussian with mean 0 and std 1
            independent_gaussian = jax.scipy.stats.norm.ppf(cube)[:, jnp.newaxis]

            # generate random samples from a multivariate normal distribution
            prior = stacked_pdf_grid_prior + jnp.einsum(
                "ij,kj->ki", independent_gaussian.T, cholesky_pdf_covmat
            ).squeeze(-1)

            return prior

    elif uniform_pdf_prior:
        error68_up = pdf_prior_grid.error68_up
        error68_down = pdf_prior_grid.error68_down

        @jax.jit
        def prior_transform(cube):
            """
            TODO
            """
            params = error68_down + (error68_up - error68_down) * cube
            return params

    return prior_transform
