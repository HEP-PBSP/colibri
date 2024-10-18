import jax.numpy as jnp
import logging

from validphys import convolution

import importlib
import inspect

log = logging.getLogger(__name__)


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
    if isinstance(closure_test_pdf, str):
        if closure_test_pdf == "colibri_model":
            pdf = closure_test_colibri_model_pdf(closure_test_model_settings, FIT_XGRID)
            return [pdf]
        else:
            raise ValueError(
                f"Unknown closure test pdf '{closure_test_pdf}'. "
                "Supported values are 'colibri_model' or LHAPDF sets."
            )
    else:
        grid = jnp.array(
            convolution.evolution.grid_values(
                closure_test_pdf, convolution.FK_FLAVOURS, FIT_XGRID, [Q0]
            ).squeeze(-1)
        )
    return grid


def closure_test_central_pdf_grid(closure_test_pdf_grid):
    """
    Returns the central replica of the closure test pdf grid.
    """
    return closure_test_pdf_grid[0]


def closure_test_colibri_model_pdf(closure_test_model_settings, FIT_XGRID):
    """
    Computes the closure test pdf grid from a colibri model.

    Parameters
    ----------
    closure_test_model_settings: dict
        Settings for the closure test model.

    FIT_XGRID: jnp.ndarray
        xgrid of the theory, computed by a production rule by taking
        the sorted union of the xgrids of the datasets entering the fit.

    Returns
    -------
    jnp.array
        The closure test pdf grid.
    """
    try:
        model = closure_test_model_settings["model"]
        # Dynamically import the module
        module = importlib.import_module(model)
        log.info(f"Successfully imported '{model}' model for closure test.")

        if hasattr(module, "config"):
            from colibri.config import colibriConfig

            config = getattr(module, "config")
            classes = inspect.getmembers(config, inspect.isclass)

            # Loop through the classes in the module
            # and find the class that is a subclass of colibriConfig
            for _, cls in classes:
                if issubclass(cls, colibriConfig) and cls is not colibriConfig:
                    # Get the signature of the produce_pdf_model method
                    signature = inspect.signature(
                        cls(input_params={}).produce_pdf_model
                    )

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
                        if arg in closure_test_model_settings:
                            inputs[arg] = closure_test_model_settings[arg]

                    # Check that keys in inputs are the same as required_args
                    if set(inputs.keys()) != set(required_args):
                        raise ValueError(
                            f"Required arguments for the model '{model}' are "
                            f"{required_args}, but got {list(inputs.keys())}."
                        )

                    # Produce the pdf model
                    pdf_model = cls(input_params={}).produce_pdf_model(
                        **inputs, output_path=None, dump_model=False
                    )

            # Compute the pdf grid
            pdf_grid_func = pdf_model.grid_values_func(FIT_XGRID)
            params = jnp.array(closure_test_model_settings["parameters"])
            pdf_grid = pdf_grid_func(params)

            return pdf_grid

        else:
            raise AttributeError(f"The model '{model}' has no 'config' module.")

    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Colibri model '{model}' is not installed.")
