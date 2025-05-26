"""
colibri.utils.py

Module containing several utils for PDF fits.

"""

import logging
import os
import pathlib
import sys
from functools import wraps
from typing import Union

import dill
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from colibri.loss_functions import chi2
from colibri.constants import LHAPDF_XGRID, EXPORT_LABELS
from colibri.export_results import write_exportgrid

from validphys import convolution

import importlib
import inspect


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


def resample_from_ns_posterior(
    samples, n_posterior_samples=1000, posterior_resampling_seed=123456
):
    """
    Resamples a subset of data points from a given set of samples without replacement.

    Parameters
    ----------
    samples: jnp.ndarray
        The input dataset to be resampled.

    n_posterior_samples: int, default is 1000
        The number of samples to draw from the input dataset.

    posterior_resampling_seed: int, default is 123456
        The random seed to ensure reproducibility of the resampling process.

    Returns
    -------
    resampled_samples: jax.Array
        The resampled subset of the input dataset, containing n_posterior_samples without selected replacement.

    """

    current_samples = samples.copy()

    rng = jax.random.PRNGKey(posterior_resampling_seed)

    resampled_samples = jax.random.choice(
        rng, current_samples, (n_posterior_samples,), replace=False
    )

    return resampled_samples


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


def resample_posterior_from_file(
    fit_path: pathlib.Path,
    file_path: str,
    n_replicas: int,
    resampling_seed: int,
    use_all_columns: bool = False,
    read_csv_args: dict = None,
):
    """
    Generic function to resample from a posterior using a specified file path.

    Parameters
    ----------
    fit_path: pathlib.Path
        The path to the fit folder.

    file_path: str
        The name of the file containing the posterior samples inside the fit folder.

    n_replicas: int
        The number of posterior samples to resample from the file.

    resampling_seed: int
        The random seed to use for resampling.

    use_all_columns: bool, default is False
        If True, all columns of the file are used. If False, the first column is ignored.

    read_csv_args: dict, default is None
        Additional arguments to pass to pd.read_csv when loading the file.

    Returns
    -------
    resampled_posterior: np.ndarray
        The resampled posterior samples.
    """
    # Check that the file exists
    full_path = fit_path / file_path
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"{full_path} does not exist; please run the appropriate fit first."
        )

    # Load the samples
    samples = pd.read_csv(full_path, **read_csv_args)
    if not use_all_columns:
        samples = samples.iloc[:, 1:]

    samples = samples.values

    # Adjust number of replicas if necessary
    if n_replicas > samples.shape[0]:
        n_replicas = samples.shape[0]
        log.warning(
            f"The chosen number of posterior samples exceeds the available posterior samples."
            f" Setting the number of resampled posterior samples to {n_replicas}."
        )

    # Resample from posterior
    resampled_posterior = resample_from_ns_posterior(
        samples,
        n_replicas,
        resampling_seed,
    )
    return resampled_posterior


def full_posterior_sample_fit_resampler(
    fit_path: pathlib.Path, n_replicas: int, resampling_seed: int
):
    """
    Wrapper for resampling from a fit with a full_posterior_sample.csv like file
    storing the posterior samples in the root of the folder.
    """
    return resample_posterior_from_file(
        fit_path,
        "full_posterior_sample.csv",
        n_replicas,
        resampling_seed,
        use_all_columns=False,
        read_csv_args={"index_col": None, "dtype": float},
    )


def write_resampled_bayesian_fit(
    resampled_posterior: np.ndarray,
    fit_path: pathlib.Path,
    resampled_fit_path: pathlib.Path,
    resampled_fit_name: Union[str, pathlib.Path],
    parametrisation_scale: float,
    csv_results_name: str,
):
    """
    Writes the resampled ns fit to `resampled_fit_path`.

    Parameters
    ----------
    resampled_posterior: np.ndarray
        The resampled posterior.

    fit_path: pathlib.Path
        The path to the original fit.

    resampled_fit_path: pathlib.Path
        The path to the resampled fit.

    resampled_fit_name: Union[str, pathlib.Path]
        The name of the resampled fit.

    parametrisation_scale: float

    csv_results_name: str
        The name of the csv file to store the resampled posterior.
    """
    log.info(f"Loading pdf model from {fit_path}")

    # load pdf_model from fit using dill
    with open(fit_path / "pdf_model.pkl", "rb") as file:
        pdf_model = dill.load(file)

    # copy old fit to resampled fit
    os.system(f"cp -r {fit_path} {resampled_fit_path}")

    # remove old replicas from resampled fit
    os.system(f"rm -r {resampled_fit_path}/replicas/*")

    # overwrite old ns_result.csv with resampled posterior
    parameters = pdf_model.param_names
    df = pd.DataFrame(resampled_posterior, columns=parameters)
    df.to_csv(str(resampled_fit_path) + f"/{csv_results_name}.csv", float_format="%.5e")

    new_rep_path = resampled_fit_path / "replicas"

    if not os.path.exists(new_rep_path):
        os.mkdir(new_rep_path)

    # Finish by writing the replicas to export grids, ready for evolution
    for i, parameters in enumerate(resampled_posterior):
        # Get the PDF grid in the evolution basis
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


def pdf_model_from_colibri_model(model_settings):
    """
    Produce a PDF model from a colibri model.

    Parameters
    ----------
    model_settings: dict
        The settings to produce the PDF model.

    Returns
    -------
    PDFModel
    """
    model_name = model_settings["model"]
    # Dynamically import the module
    try:
        module = importlib.import_module(model_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Colibri model '{model_name}' is not installed.")

    log.info(f"Successfully imported '{model_name}' model for pdf_model production.")

    if hasattr(module, "config"):
        from colibri.config import colibriConfig

        config = getattr(module, "config")
        classes = inspect.getmembers(config, inspect.isclass)

        # Loop through the classes in the module
        # and find the class that is a subclass of colibriConfig
        for _, cls in classes:
            if issubclass(cls, colibriConfig) and cls is not colibriConfig:
                # Get the signature of the produce_pdf_model method
                signature = inspect.signature(cls(input_params={}).produce_pdf_model)

                # Get the required arguments for the produce_pdf_model method
                required_args = []
                # Loop through the parameters in the function's signature
                for name, param in signature.parameters.items():
                    # Check if the parameter has no default value
                    if param.default == inspect.Parameter.empty and param.kind in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    ):
                        if name == "output_path" or name == "dump_model":
                            continue
                        required_args.append(name)

                # Create a dictionary with the required arguments
                # and their values from closure_test_model_settings
                inputs = {}
                for arg in signature.parameters:
                    if arg in model_settings:
                        inputs[arg] = model_settings[arg]

                # Check that keys in inputs are the same as required_args
                if set(inputs.keys()) != set(required_args):
                    raise ValueError(
                        f"Required arguments for the model '{model_name}' are "
                        f"{required_args}, but got {list(inputs.keys())}."
                    )

                # Produce the pdf model
                pdf_model = cls(input_params={}).produce_pdf_model(
                    **inputs, output_path=None, dump_model=False
                )

                return pdf_model
    else:
        raise AttributeError(f"The model '{model_name}' has no 'config' module.")


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


def closest_indices(a, v, atol=1e-8):
    """
    Finds the indices of values in `a` that are closest to the given value(s) `v`.

    Unlike `np.searchsorted`, this function identifies indices where the values in `v`
    are approximately equal to those in `a` within the specified tolerance.
    The main difference is that np.searchsorted returns the index where each
    element of v should be inserted in a in order to preserve the order (see example below).

    Parameters
    ----------
    a : array-like

    v : array-like or float

    atol : float, default is 1e-8
        absolute tolerance used to find closest indices.

    Returns
    -------
    array-like

    Examples
    --------
    >>> a = np.array([1, 2, 3])
    >>> v = np.array([1.1, 3.0])
    >>> closest_indices(array, value, atol=0.1)
    array([0, 2])

    >>> np.searchsorted(a, v)
    array([1, 2])

    """
    # Handle scalar input for v
    if v.ndim == 0:
        return jnp.where(jnp.isclose(a, v, atol=atol) == True)[0]

    return jnp.where(jnp.isclose(a, v[:, None], atol=atol) == True)[1]
