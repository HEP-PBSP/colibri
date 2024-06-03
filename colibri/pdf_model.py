"""
colibri.pdf_model.py

This module implements an abstract class PDFModel which is filled by the various models.

"""

from abc import ABC, abstractmethod
from typing import Callable, Tuple
import jax.numpy as jnp
import jax


class PDFModel(ABC):
    """An abstract class describing the key features of a PDF."""

    name = "Abstract PDFModel"

    @property
    @abstractmethod
    def param_names(self):
        """This should return a list of names for the fitted parameters of the model."""
        pass

    @abstractmethod
    def grid_values_func(self, xgrid):
        """This function should produce a grid values function, which takes
        in the model parameters, and produces the PDF values on the grid xgrid.
        """
        pass

    def pred_and_pdf_func(
        self, xgrid, forward_map
    ) -> Callable[[jnp.array], Tuple[jnp.ndarray, jnp.ndarray]]:
        """This method produces a function that returns a tuple of 2 arrays,
        taking the model parameters as input.
        The first array are the predictions for the data,
        the second are the PDF values on the xgrid.

        The forward_map is a function that takes in the PDF defined on the
        xgrid grid. They must therefore be compatible.
        """
        pdf_func = self.grid_values_func(xgrid)

        def pred_and_pdf(params, fk_tables):
            pdf = pdf_func(params)
            predictions = forward_map(pdf, fk_tables)
            return predictions, pdf

        return pred_and_pdf
