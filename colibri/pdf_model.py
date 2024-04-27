"""
colibri.pdf_model.py

This module implements an abstract class PDFModel which is filled by the various models.

"""

from abc import ABC, abstractmethod, abstractproperty
from typing import Callable, Tuple
import jax.numpy as jnp


class PDFModel(ABC):
    """An abstract class describing the key features of a PDF."""

    @abstractproperty
    def param_names(self):
        """This should return a list of names for the fitted parameters of the model."""
        pass

    @abstractmethod
    def grid_values_func(self, xgrid):
        """This function should produce a grid values function, which takes
        in the model parameters, and produces the PDF values on the grid xgrid.
        """
        pass

    @abstractmethod
    def pred_and_pdf_func(
        self, xgrid, forward_map
    ) -> Callable[[jnp.array], Tuple[jnp.ndarray, jnp.ndarray]]:
        """This method should produce a function that returns a tuple of 2 arrays,
        taking the model parameters as input.
        The first array are the predictions for the data,
        the second are the PDF values on the xgrid.
        """
        pass
