"""
colibri.pdf_model.py

This module implements an abstract class PDFModel which is filled by the various models.

"""

from abc import ABC, abstractmethod
from typing import Callable, Tuple

import jax.numpy as jnp
from jax.typing import ArrayLike


class PDFModel(ABC):
    """An abstract class describing the key features of a PDF model."""

    name = "Abstract PDFModel"

    @property
    @abstractmethod
    def param_names(self) -> list:
        """This should return a list of names for the fitted parameters of the PDF model.
        The order of the names is important as it will be assumed to be the order of the parameters
        fed to the model.
        """
        pass
    
    def extra_params(self) -> list:
        """
        This should return a list of names for parameters that are not used to parametrize the PDF model,
        but are still relevant for the model (e.g. heavy quark masses, SMEFT parameters, etc.).
        
        Default is an empty list.
        """
        return []
    
    def all_param_names(self) -> list:
        """Returns a list of all parameter names, including both PDF model parameters and extra parameters."""
        return self.param_names + self.extra_params

    @abstractmethod
    def grid_values_func(self, xgrid: ArrayLike) -> Callable[[jnp.array], jnp.ndarray]:
        """This function should produce a grid values function, which takes
        in the model parameters, and produces the PDF values on the grid xgrid.
        The grid values function should be a function of the parameters and return
        an array of shape (N_fl, Nx). The first dimension is the number of flavours expected
        by the FK tables belonging to the chosen theoryID.
        The second dimension is the number of points in the xgrid, i.e. Nx = len(xgrid).

        Example
        -------
        ::

            def grid_values_func(xgrid):
                def func(params):
                    # Define expression for each flavour
                    fl_1 = params[0] + params[1] * xgrid
                    fl_2 = params[2] + params[3] * xgrid

                    # Combine the flavours into a single array
                    # This is just an example, the actual implementation will depend on the model
                    # and the number of flavours

                    return jnp.array([fl_1, fl_2])
                return func
        """
        pass

    def pred_and_pdf_func(
        self,
        xgrid: ArrayLike,
        forward_map: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    ) -> Callable[[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        """Creates a function that returns a tuple of two arrays, given the model parameters and the fast kernel arrays as input.

        The returned function produces:
        - The first array: 1D vector of theory predictions for the data.
        - The second array: PDF values evaluated on the x-grid, using `self.grid_values_func`, with shape (Nfl, Nx).

        The `forward_map` is used to map the PDF values defined on the x-grid and the fast kernel arrays into the corresponding theory prediction vector.
        """
        pdf_func = self.grid_values_func(xgrid)

        def pred_and_pdf(params, fast_kernel_arrays):
            """
            Parameters
            ----------
            params: jnp.array
                The model parameters.

            fast_kernel_arrays: tuple
                tuple of tuples of jnp.arrays
                The FK tables to use.

            Returns
            -------
            tuple
                The predictions and the PDF values.
            """
            pdf = pdf_func(params)
            predictions = forward_map(pdf, fast_kernel_arrays)
            return predictions, pdf

        return pred_and_pdf

    def extra_forward_map(self, predictions, extra_params):
        """
        NOTE: 
        this function here is user specified and can potentially be anything.
        The default forward_map always computes theory predictions from PDFs and FK tables.

        This function can be completed to modify the predictions computed from the PDFs,
        using extra parameters that are not part of the PDF parametrization.

        e.g. (how the function could look like for SMEFT):
        def extra_forward_map(self, predictions, extra_params):
            # extra_params could contain SMEFT Wilson coefficients
            # Modify predictions based on SMEFT contributions
            modified_predictions = predictions + compute_smeft_contributions(predictions, extra_params)
            return modified_predictions
        
        Parameters
        ----------
        predictions: jnp.ndarray
            The theory predictions computed from the PDFs.
        extra_params: list
            The extra parameters to modify the predictions.    
        """
        raise NotImplementedError(
            "extra_forward_map is not implemented for this PDFModel."
        )
    
    def predictions(self, xgrid, forward_map):
        """
        The default simply returns self.pred_and_pdf_func when the extra_forward_map is NotImplemented.

        TODO: ...
        """
        pred_and_pdf = self.pred_and_pdf_func(xgrid, forward_map)

        # Check if extra_forward_map is implemented
        if self.extra_forward_map.__func__ is PDFModel.extra_forward_map:
            return pred_and_pdf
        else:
            def modified_pred_and_pdf(params, fast_kernel_arrays):
                predictions, pdf = pred_and_pdf(params.pdf, fast_kernel_arrays)
                modified_predictions = self.extra_forward_map(predictions, params.extra)
                return modified_predictions, pdf
            return modified_pred_and_pdf
