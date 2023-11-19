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

"""
Dict, key = flavour id (see FLAVOURS_ID_MAPPINGS), val = reduced x grid.
"""
REDUCED_XGRIDS = {
    "length": 50,
    0: [],
    1: XGRID,
    2: XGRID,
    3: XGRID,
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
    10: [],
    11: [],
    12: [],
    13: [],
}

"""
Specifies which flavours to include in a fit.
"""
FLAVOUR_MAPPING = [1, 2, 3]


def interpolate_grid(stacked_pdf_grid, flavour_mapping=FLAVOUR_MAPPING):
    """
    TODO

    Parameters
    ----------
    stacked_pdf_grid: jnp.array

    flavour_mapping: list, default is ppdf.ppdf_model.FLAVOUR_MAPPING
        specifies the ids of the flavours to include in a fit.

    """

    # all reduced x grids should have the same length
    length_grid = REDUCED_XGRIDS["length"]

    # reshape stacked_pdf_grid to (len(REDUCED_XGRID), len(flavour_mapping))
    reshaped_stacked_pdf_grid = stacked_pdf_grid.reshape(
        (length_grid, len(flavour_mapping)), order="F"
    )

    # generate an empty matrix of shape (len(super_net.constants.XGRID),valipdhys.convolution.NFK)
    input_grid = jnp.zeros((len(XGRID), convolution.NFK))

    # interpolate columns of reshaped_stacked_pdf_grid
    for i, fl in enumerate(flavour_mapping):
        input_grid = input_grid.at[:, fl].set(
            jnp.interp(
                jnp.array(XGRID),
                jnp.array(REDUCED_XGRIDS[fl]),
                reshaped_stacked_pdf_grid[:, i],
            )
        )

    return input_grid


@dataclass(frozen=True)
class PdfPriorGrid:
    """
    TODO
    """

    stacked_pdf_grid_prior: jnp.array
    pdf_covmat_prior: jnp.array
    error68_up: jnp.array
    error68_down: jnp.array


def pdf_prior_grid(pdf_prior, flavour_mapping=FLAVOUR_MAPPING, Q0=1.65):
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
                    pdf_prior, [convolution.FK_FLAVOURS[fl]], REDUCED_XGRIDS[fl], [Q0]
                )
            ).squeeze(-1)[0],
        )

    # generate PDF covariance matrix of size Nx * Nfl x Nx * Nfl
    replicas_grid = convolution.evolution.grid_values(
        pdf_prior,
        [convolution.FK_FLAVOURS[fl] for fl in flavour_mapping],
        REDUCED_XGRIDS[fl],
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


def ppdf_model_prior(
    pdf_prior_grid, uniform_pdf_prior=True, gaussian_pdf_prior=False, sigma_pdf_prior=1
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

        def prior_transform(cube):
            """
            TODO
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

        def prior_transform(cube):
            """
            TODO
            """
            params = cube.copy()
            params = error68_down + (error68_up - error68_down) * cube
            return params

    return prior_transform
