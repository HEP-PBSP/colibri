"""
colibri.pdf_model.py

This module implements an abstract class PDFModel which is filled by the various models.

"""

from abc import ABC, abstractmethod
from typing import Callable

import jax.numpy as jnp
from jax.typing import ArrayLike


class PDFModel(ABC):
    """An abstract class describing the key features of a PDF model."""

    name = "Abstract PDFModel"

    @property
    @abstractmethod
    def param_names(self) -> list:
        """This should return a list of names for the fitted parameters of the model.
        The order of the names is important as it will be assumed to be the order of the parameters
        fed to the model.
        """
        pass

    @property
    def n_parameters(self):
        """
        Returns the number of parameters of the pdf model.
        """
        return len(self.param_names)

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
