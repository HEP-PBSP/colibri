import jax.numpy as jnp
import jax.scipy.linalg as jla


def sqrt_covmat_jax(covariance_matrix):
    """
    Same as `validphys.covmats.sqrt_covmat` but
    for jax.numpy arrays

    Parameters
    ----------
    covariance_matrix : jnp.ndarray
        A positive definite covariance matrix, which is N_dat x N_dat (where
        N_dat is the number of data points after cuts) containing uncertainty
        and correlation information.

    Returns
    -------
    sqrt_mat : jnp.ndarray
        The square root of the input covariance matrix, which is N_dat x N_dat
        (where N_dat is the number of data points after cuts), and which is the
        the lower triangular decomposition. The following should be ``True``:
        ``jnp.allclose(sqrt_covmat @ sqrt_covmat.T, covariance_matrix)``.
    """

    dimensions = covariance_matrix.shape

    if covariance_matrix.size == 0:
        raise ValueError("Attempting the decomposition of an empty matrix.")
    elif dimensions[0] != dimensions[1]:
        raise ValueError(
            "The input covariance matrix should be square but "
            f"instead it has dimensions {dimensions[0]} x "
            f"{dimensions[1]}"
        )

    sqrt_diags = jnp.sqrt(jnp.diag(covariance_matrix))
    correlation_matrix = covariance_matrix / sqrt_diags[:, jnp.newaxis] / sqrt_diags
    decomp = jla.cholesky(correlation_matrix)
    sqrt_matrix = (decomp * sqrt_diags).T
    return sqrt_matrix
