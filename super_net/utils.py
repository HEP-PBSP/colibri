"""
super_net.utils.py

Module containing several utils for PDF fits.

Author: Mark N. Costantini
Date: 11.11.2023
"""

import jax
import jax.numpy as jnp

from dataclasses import dataclass, asdict

from super_net.constants import XGRID
from validphys import convolution

def compute_maximal_curvature(
    _data_values,
    _pred_data,
    pdf_model,
):
    """
    Given a PDF model and predictions, computes the maximal
    """

    interp_func = pdf_model.grid_values_func(XGRID)

    @jax.jit
    def pred_from_params(params):
        return _pred_data(interp_func(params))

    # The matrix dt/dc from (6.40) of James' thesis
    jacobian = jax.jacfwd(pred_from_params)
    covmat = _data_values.training_data.covmat
    ndat, _ = covmat.shape
    nparams = len(pdf_model.param_names)
    inv_covmat = jnp.linalg.inv(covmat)

    # @jax.jit
    def curvature_prefactor(params):
        dt_dc = jnp.array(jacobian(params)).reshape((ndat, nparams))
        Mprime = dt_dc.T @ inv_covmat
        # Construct the M matrix from the Jacobian. This is effected by finding
        # the non-zero solutions of Mprime x = 0. Construct the singular value
        # decomposition of Mprime.
        _, _, Vh = jnp.linalg.svd(Mprime)
        M = Vh[:, nparams:]
        # Now take the determinant from equation (6.40) of James' thesis
        return abs(jnp.linalg.det(jnp.concatenate([dt_dc, M], axis=1)))

    print(curvature_prefactor([-0.001]*len(pdf_model.param_names)))
    print(curvature_prefactor([0.001]*len(pdf_model.param_names)))
    print(curvature_prefactor([0.01]*len(pdf_model.param_names)))
    print(curvature_prefactor([0.1]*len(pdf_model.param_names)))

    @jax.jit
    def integral_prefactor(params):
        # The integral prefactor from James' thesis
        # This is probably ridiculously difficult to compute
        pass

def replica_seed(replica_index):
    """
    Generate a random integer given a replica_index.
    Note that each replica index has a unique key.
    """
    key = jax.random.PRNGKey(replica_index)
    randint = jax.random.randint(key, shape=(1,), minval=0, maxval=1e10)
    return int(randint)


def trval_seed(trval_index):
    """
    Returns a PRNGKey key given `trval_index` seed.
    """
    key = jax.random.PRNGKey(trval_index)
    return key


@dataclass(frozen=True)
class TrainValidationSplit:
    training: jnp.array
    validation: jnp.array

    def to_dict(self):
        return asdict(self)


def training_validation_split(
    indices, mc_validation_fraction, random_seed, shuffle_indices=True
):
    """
    Performs training validation split on an array.

    Parameters
    ----------
    indices: jaxlib.xla_extension.Array

    mc_validation_fraction: float

    random_seed: jaxlib.xla_extension.Array
        PRNGKey, obtained as jax.random.PRNGKey(random_number)

    shuffle_indices: bool

    Returns
    -------
    dataclass
    """

    if shuffle_indices:
        # shuffle indices
        permuted_indices = jax.random.permutation(random_seed, indices)
    else:
        permuted_indices = indices

    # determine split point
    split_point = int(indices.shape[0] * (1 - mc_validation_fraction))

    # split indices
    indices_train = permuted_indices[:split_point]
    indices_validation = permuted_indices[split_point:]

    return TrainValidationSplit(training=indices_train, validation=indices_validation)


def t0_pdf_grid(t0pdfset, Q0=1.65):
    """
    Computes the t0 pdf grid in the evolution basis.

    Parameters
    ----------
    t0pdfset: validphys.core.PDF

    Q0: float, default is 1.65

    Returns
    -------
    t0grid: jnp.array
        t0 grid, is N_rep x N_fl x N_x
    """

    t0grid = jnp.array(
        convolution.evolution.grid_values(
            t0pdfset, convolution.FK_FLAVOURS, XGRID, [Q0]
        ).squeeze(-1)
    )
    return t0grid


def closure_test_pdf_grid(closure_test_pdf, Q0=1.65):
    """
    Computes the closure_test_pdf grid in the evolution basis.

    Parameters
    ----------
    closure_test_pdf: validphys.core.PDF

    Q0: float, default is 1.65

    Returns
    -------
    grid: jnp.array
        grid, is N_rep x N_fl x N_x
    """

    grid = jnp.array(
        convolution.evolution.grid_values(
            closure_test_pdf, convolution.FK_FLAVOURS, XGRID, [Q0]
        ).squeeze(-1)
    )
    return grid


def resample_from_ns_posterior(
    samples, n_posterior_samples=1000, posterior_resampling_seed=123456
):
    """
    TODO
    """

    current_samples = samples.copy()

    rng = jax.random.PRNGKey(posterior_resampling_seed)

    resampled_samples = jax.random.choice(
        rng, current_samples, (n_posterior_samples,), replace=False
    )

    return resampled_samples


def closure_test_central_pdf_grid(closure_test_pdf_grid):
    """
    Returns the central replica of the closure test pdf grid.
    """
    return closure_test_pdf_grid[0]
