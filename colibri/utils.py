"""
colibri.utils.py

Module containing several utils for PDF fits.

"""

import logging
import os
import pathlib
import sys
from functools import wraps

import dill
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from colibri.loss_functions import chi2
from colibri.constants import LHAPDF_XGRID, EXPORT_LABELS
from colibri.export_results import write_exportgrid

from validphys import convolution

log = logging.getLogger(__name__)


def mask_fktable_array(fktable, flavour_indices=None):
    """
    Takes an FKTableData instance and returns an FK table array with masked flavours.

    Parameters
    ----------
    fktable: validphys.coredata.FKTableData

    flavour_indices: list, default is None
        The indices of the flavours to keep.
        If None, returns the original FKTableData.get_np_fktable() array with no masking.

    Returns
    -------
    jnp.array
        The FK table array with masked flavours.
    """

    if flavour_indices is None:
        return jnp.array(fktable.get_np_fktable())

    if fktable.hadronic:
        lumi_indices = fktable.luminosity_mapping
        mask_even = jnp.isin(lumi_indices[0::2], jnp.array(flavour_indices))
        mask_odd = jnp.isin(lumi_indices[1::2], jnp.array(flavour_indices))

        fk_arr_mask = mask_even * mask_odd

        return jnp.array(fktable.get_np_fktable()[:, fk_arr_mask, :, :])

    else:
        lumi_indices = fktable.luminosity_mapping
        fk_arr_mask = jnp.isin(lumi_indices, jnp.array(flavour_indices))

        return jnp.array(fktable.get_np_fktable()[:, fk_arr_mask, :])


def mask_luminosity_mapping(fktable, flavour_indices=None):
    """
    Takes an FKTableData instance and returns a new instance with masked luminosity mapping.

    Parameters
    ----------
    fktable: validphys.coredata.FKTableData

    flavour_indices: list, default is None
        The indices of the flavours to keep.
        If None, returns the original FKTableData.luminosity_mapping with no masking.

    Returns
    -------
    jnp.array
        The luminosity mapping with masked flavours.
    """

    if flavour_indices is None:
        return fktable.luminosity_mapping

    if fktable.hadronic:
        lumi_indices = fktable.luminosity_mapping
        mask_even = jnp.isin(lumi_indices[0::2], jnp.array(flavour_indices))
        mask_odd = jnp.isin(lumi_indices[1::2], jnp.array(flavour_indices))

        # for hadronic predictions pdfs enter in pair, hence product of two
        # boolean arrays and repeat by 2
        mask = jnp.repeat(mask_even * mask_odd, repeats=2)
        lumi_indices = lumi_indices[mask]

        return lumi_indices

    else:
        lumi_indices = fktable.luminosity_mapping
        mask = jnp.isin(lumi_indices, jnp.array(flavour_indices))
        lumi_indices = lumi_indices[mask]

        return lumi_indices


def t0_pdf_grid(t0pdfset, FIT_XGRID, Q0=1.65):
    """
    Computes the t0 pdf grid in the evolution basis.

    Parameters
    ----------
    t0pdfset: validphys.core.PDF

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    Q0: float, default is 1.65

    Returns
    -------
    t0grid: jnp.array
        t0 grid, is N_rep x N_fl x N_x
    """

    t0grid = jnp.array(
        convolution.evolution.grid_values(
            t0pdfset, convolution.FK_FLAVOURS, FIT_XGRID, [Q0]
        ).squeeze(-1)
    )
    return t0grid


def closure_test_pdf_grid(closure_test_pdf, FIT_XGRID, Q0=1.65):
    """
    Computes the closure_test_pdf grid in the evolution basis.

    Parameters
    ----------
    closure_test_pdf: validphys.core.PDF

    FIT_XGRID: np.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    Q0: float, default is 1.65

    Returns
    -------
    grid: jnp.array
        grid, is N_rep x N_fl x N_x
    """

    grid = jnp.array(
        convolution.evolution.grid_values(
            closure_test_pdf, convolution.FK_FLAVOURS, FIT_XGRID, [Q0]
        ).squeeze(-1)
    )
    return grid


def resample_from_ns_posterior(
    samples, n_posterior_samples=1000, posterior_resampling_seed=123456
):
    """
    Sample uniformly without replacement from a distribution.

    Parameters
    ----------
    samples: array of samples
    n_posterior_samples: int
    posterior_resampling_seed: int

    Returns
    -------
    array of resampled samples
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


def get_fit_path(fit):
    fit_path = pathlib.Path(sys.prefix) / "share/colibri/results" / fit
    if not os.path.exists(fit_path):
        raise FileNotFoundError(
            "Could not find a fit " + fit + " in the colibri/results directory."
        )
    return pathlib.Path(fit_path)


def get_full_posterior(colibri_fit):
    """
    Given a colibri fit, returns the pandas dataframe with the results of the fit
    at the parameterisation scale.

    Parameters
    ----------
    colibri_fit : str
        The name of the fit to read.


    Returns
    -------
    pandas dataframe
    """

    fit_path = get_fit_path(colibri_fit)

    csv_path = fit_path / "full_posterior_sample.csv"
    # check that file exist
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Could not find the full posterior sample for the fit " + colibri_fit
        )

    df = pd.read_csv(csv_path, index_col=0)

    return df


def get_pdf_model(colibri_fit):
    """
    Given a colibri fit, returns the PDF model.

    Parameters
    ----------
    colibri_fit : str
        The name of the fit to read.


    Returns
    -------
    PDFModel
    """

    fit_path = get_fit_path(colibri_fit)

    pdf_model_path = fit_path / "pdf_model.pkl"
    # check that file exist
    if not os.path.exists(pdf_model_path):
        raise FileNotFoundError(
            "Could not find the pdf model for the fit " + colibri_fit
        )

    with open(pdf_model_path, "rb") as file:
        pdf_model = dill.load(file)

    return pdf_model


def pdf_models_equal(pdf_model_1, pdf_model_2):
    """
    Checks if two pdf models are equal.

    Parameters
    ----------
    pdf_model_1 : PDFModel
    pdf_model_2 : PDFModel

    Returns
    -------
    bool
    """

    # Check that the two models have the same attributes
    if vars(pdf_model_1) != vars(pdf_model_2):
        # Check that keys are the same
        if vars(pdf_model_1).keys() != vars(pdf_model_2).keys():
            log.error("The two models do not have the same structure.")
            log.error(
                "The first model has attributes " f"{list(vars(pdf_model_1).keys())} "
            )
            log.error(
                "The second model has attributes " f"{list(vars(pdf_model_2).keys())}."
            )
            return False

        log.error("The two models do not have the same attributes.")
        # Loop over the attributes and check which ones are different
        for key, value in vars(pdf_model_1).items():
            if value != vars(pdf_model_2)[key]:
                log.error("The first model has attribute " f"{key} = {value} ")
                log.error(
                    "The second model has attribute "
                    f"{key} = {vars(pdf_model_2)[key]}."
                )

        return False

    return True


def cast_to_numpy(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return np.array(result)

    return wrapper


def likelihood_float_type(
    _pred_data,
    pdf_model,
    FIT_XGRID,
    bayesian_prior,
    output_path,
    central_inv_covmat_index,
    fast_kernel_arrays,
):
    """
    Writes the dtype of the likelihood function to a file.
    Mainly used for testing purposes.
    """

    loss_function = chi2

    central_values = central_inv_covmat_index.central_values
    inv_covmat = central_inv_covmat_index.inv_covmat

    pred_and_pdf = pdf_model.pred_and_pdf_func(FIT_XGRID, forward_map=_pred_data)

    @jax.jit
    def log_likelihood(params, central_values, inv_covmat, fast_kernel_arrays):
        predictions, _ = pred_and_pdf(params, fast_kernel_arrays)
        return -0.5 * loss_function(central_values, predictions, inv_covmat)

    params = bayesian_prior(
        jax.random.uniform(jax.random.PRNGKey(0), shape=(len(pdf_model.param_names),))
    )

    dtype = log_likelihood(params, central_values, inv_covmat, fast_kernel_arrays).dtype

    # save the dtype to the output path
    with open(output_path / "dtype.txt", "w") as file:
        file.write(str(dtype))


def ns_fit_resampler(
    fit_path,
    n_replicas,
    resampling_seed,
):
    """
    Function uses resample_from_ns_posterior to resample from a nested sampling result posterior.
    """

    # Check that the .txt file with equally weighted posterior samples exists
    if not os.path.exists(fit_path / "ultranest_logs/chains/equal_weighted_post.txt"):
        raise FileNotFoundError(
            f"{fit_path}/ultranest_logs/chains/equal_weighted_post.txt does not exist;"
            "please run the bayesian fit first."
        )

    equal_weight_post_path = fit_path / "ultranest_logs/chains/equal_weighted_post.txt"

    samples = pd.read_csv(equal_weight_post_path, sep="\s+", dtype=float).values

    if n_replicas > samples.shape[0]:
        n_replicas = samples.shape[0]
        log.warning(
            f"The chosen number of posterior samples exceeds the number of posterior"
            "samples computed by ultranest. Setting the number of resampled posterior"
            f"samples to {n_replicas}"
        )

    resampled_posterior = resample_from_ns_posterior(
        samples,
        n_replicas,
        resampling_seed,
    )

    return resampled_posterior


def analytics_fit_resampler(
    fit_path,
    n_replicas,
    resampling_seed,
):
    """
    Function uses resample_from_ns_posterior to resample from analytic posterior.
    """

    # Check that the .txt file with equally weighted posterior samples exists
    if not os.path.exists(fit_path / "full_posterior_sample.csv"):
        raise FileNotFoundError(
            f"{fit_path}/full_posterior_sample.csv does not exist;"
            "please run the analytic fit first."
        )

    equal_weight_post_path = fit_path / "full_posterior_sample.csv"

    samples = (
        pd.read_csv(equal_weight_post_path, index_col=None, dtype=float)
        .iloc[:, 1:]
        .values
    )

    if n_replicas > samples.shape[0]:
        n_replicas = samples.shape[0]
        log.warning(
            f"The chosen number of posterior samples exceeds the number of posterior"
            "samples computed by ultranest. Setting the number of resampled posterior"
            f"samples to {n_replicas}"
        )

    resampled_posterior = resample_from_ns_posterior(
        samples,
        n_replicas,
        resampling_seed,
    )

    return resampled_posterior


def write_resampled_ns_fit(
    resampled_posterior,
    fit_path,
    resampled_fit_path,
    n_replicas,
    resampled_fit_name,
    parametrisation_scale,
    copy_fit_dir=True,
    write_ns_results=True,
    replica_range=None,
):
    """
    Writes the resampled ns fit to `resampled_fit_path`.
    """

    # convert paths to Posix if they are not already
    if (not type(fit_path) == pathlib.PosixPath) or (
        not type(resampled_fit_path) == pathlib.PosixPath
    ):
        fit_path = pathlib.PosixPath(fit_path)
        resampled_fit_path = pathlib.PosixPath(resampled_fit_path)

    log.info(f"Loading pdf model from {fit_path}")

    # load pdf_model from fit using dill
    with open(fit_path / "pdf_model.pkl", "rb") as file:
        pdf_model = dill.load(file)

    if copy_fit_dir:
        # copy old fit to resampled fit
        os.system(f"cp -r {fit_path} {resampled_fit_path}")

        # remove old replicas from resampled fit
        os.system(f"rm -r {resampled_fit_path}/replicas/*")

    if write_ns_results:
        # overwrite old ns_result.csv with resampled posterior
        parameters = pdf_model.param_names
        df = pd.DataFrame(resampled_posterior, columns=parameters)
        df.to_csv(str(resampled_fit_path) + "/ns_result.csv", float_format="%.5e")

    if replica_range:
        indices_per_process = replica_range
    else:
        indices_per_process = range(n_replicas)

    new_rep_path = resampled_fit_path / "replicas"

    if not os.path.exists(new_rep_path):
        os.mkdir(new_rep_path)

    # Finish by writing the replicas to export grids, ready for evolution
    for i in indices_per_process:

        # Get the PDF grid in the evolution basis
        parameters = resampled_posterior[i]
        lhapdf_interpolator = pdf_model.grid_values_func(LHAPDF_XGRID)
        grid_for_writing = np.array(lhapdf_interpolator(parameters))

        replica_index = i + 1

        replica_index_path = new_rep_path / f"replica_{replica_index}"
        if not os.path.exists(replica_index_path):
            os.mkdir(replica_index_path)

        grid_name = replica_index_path / resampled_fit_name

        log.info(f"Writing exportgrid for replica {replica_index}")
        write_exportgrid(
            grid_for_writing=grid_for_writing,
            grid_name=grid_name,
            replica_index=replica_index,
            Q=parametrisation_scale,
            xgrid=LHAPDF_XGRID,
            export_labels=EXPORT_LABELS,
        )

    log.info(f"Resampling completed. Resampled fit stored in {resampled_fit_path}")


def compute_determinants_of_principal_minors(C):
    """
    Computes the determinants of the principal minors of a symmetric, positive semi-definite matrix C.

    Parameters
    ----------
    C (np.ndarray): An nxn covariance matrix (symmetric, positive semi-definite)

    Returns
    -------
    List[float]: A list of determinants of the principal minors from C_n down to C_0
    """
    n = C.shape[0]
    determinants = []

    # Perform the Cholesky decomposition of the full matrix C
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is not positive semi-definite or symmetric.")

    # Compute determinants of principal minors by iteratively removing rows/columns from the Cholesky factor
    for k in range(n, 0, -1):
        # Compute determinant of C_k using the product of diagonal entries of the top-left kxk submatrix of L
        L_k = L[:k, :k]
        det_C_k = np.prod(np.diag(L_k)) ** 2  # Square of product of diagonals
        determinants.append(det_C_k)

    # C_0 is defined to have determinant 1
    determinants.append(1.0)

    return np.array(determinants)[::-1]
