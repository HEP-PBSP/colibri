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
    0: [],
    1: [
        2.00000000e-07,
        2.43894329e-06,
        1.96025050e-05,
        2.38787829e-04,
        2.87386758e-03,
        6.04800288e-02,
        2.65113704e-01,
        7.29586844e-01,
        9.29586844e-01,
    ],
    2: [
        2.00000000e-07,
        2.43894329e-06,
        1.96025050e-05,
        2.38787829e-04,
        2.87386758e-03,
        6.04800288e-02,
        2.65113704e-01,
        7.29586844e-01,
        9.29586844e-01,
    ],
    3: [
        2.00000000e-07,
        2.43894329e-06,
        1.96025050e-05,
        2.38787829e-04,
        2.87386758e-03,
        6.04800288e-02,
        2.65113704e-01,
        7.29586844e-01,
        9.29586844e-01,
    ],
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

    lenght_grid = jnp.sum(
        jnp.array([len(REDUCED_XGRIDS[fl]) for fl in flavour_mapping])
    )

    # reshape stacked_pdf_grid to (len(REDUCED_XGRID), len(flavour_mapping))
    reshaped_stacked_pdf_grid = stacked_pdf_grid.reshape(
        (lenght_grid, len(flavour_mapping)), order="F"
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
    stacked_pdf_grid_prior: jnp.array
    pdf_covmat_prior: jnp.array


def pdf_prior_grid(pdf_prior, flavour_mapping=FLAVOUR_MAPPING, Q0=1.65):
    """
    TODO 
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
    jnp.array
        1-D array of central values of prior PDF for the reduced
        grid used in the fit. Flavours included in the fit are stacked.
    TODO
    """

    stacked_pdf_grid_prior = jnp.array([])

    # note: different flavours might have different len REDUCED_XGRID
    for fl in flavour_mapping:
        # save central value of pdf_prior

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

    pdf_covmat_prior = jnp.cov(replicas_grid.reshape((replicas_grid.shape[0], -1)).T)
    
    return PdfPriorGrid(
        stacked_pdf_grid_prior=stacked_pdf_grid_prior, pdf_covmat_prior=pdf_covmat_prior
    )


def ppdf_model_prior(pdf_prior_grid, sigma):
    """
    General Idea: given pdf_prior_grid return a grid
    whose values are random gaussians centred at the prior
    grid and with certain sigma.

    Use Cholesky decomposition to generate a multivariate Gaussian

    Parameters
    ----------
    pdf_prior:

    """
    stacked_pdf_grid_prior = pdf_prior_grid.stacked_pdf_grid_prior
    pdf_covmat_prior = sigma * pdf_prior_grid.pdf_covmat_prior

    cholesky_pdf_covmat = jnp.linalg.cholesky(pdf_covmat_prior)

    # a = jnp.linalg.inv(pdf_covmat_prior)
    # l, v = jnp.linalg.eigh(a)
    # rotation_matrix = jnp.dot(v, jnp.diag(1. / jnp.sqrt(l)) )

    def prior_transform(cube):
        """
        TODO
        """
        # sample an independent multivariate gaussian
        independent_gaussian = jax.scipy.stats.norm.ppf(cube)
        
        # rotate and shift
        return stacked_pdf_grid_prior + jnp.einsum("ij,kj->ki", independent_gaussian.T, cholesky_pdf_covmat).squeeze(-1)
        
        # return stacked_pdf_grid_prior + jnp.einsum("ij,kj->ki", independent_gaussian.T, rotation_matrix).squeeze(-1)
        

    return prior_transform
