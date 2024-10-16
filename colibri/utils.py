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


def closure_test_pdf_grid(
    closure_test_pdf, FIT_XGRID, Q0=1.65, closure_test_model_settings={}
):
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

    if closure_test_pdf == "colibri_model":
        pdf = closure_test_colibri_model_pdf(closure_test_model_settings, FIT_XGRID)
        return [pdf]
    else:
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


def closure_test_colibri_model_pdf(closure_test_model_settings, FIT_XGRID):
    try:
        model = closure_test_model_settings["model"]
        # Dynamically import the module
        module = importlib.import_module(model)
        log.info(f"Successfully imported '{model}' model for closure test.")

        if hasattr(module, "config"):
            from colibri.config import colibriConfig

            config = getattr(module, "config")
            classes = inspect.getmembers(config, inspect.isclass)

            for _, cls in classes:
                if issubclass(cls, colibriConfig) and cls is not colibriConfig:
                    signature = inspect.signature(
                        cls(input_params={}).produce_pdf_model
                    )
                    inputs = {}
                    for arg in signature.parameters:
                        if arg in closure_test_model_settings:
                            inputs[arg] = closure_test_model_settings[arg]

                    pdf_model = cls(input_params={}).produce_pdf_model(
                        **inputs, output_path=None, dump_model=False
                    )

            pdf_grid_func = pdf_model.grid_values_func(FIT_XGRID)
            params = jnp.array(closure_test_model_settings["parameters"])
            pdf_grid = pdf_grid_func(params)

            return pdf_grid

        else:
            raise AttributeError(f"The model '{model}' has no 'config' module.")

    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Colibri model '{model}' is not installed.")
