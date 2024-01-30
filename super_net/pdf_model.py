"""
super_net.pdf_model.py

This module implements an abstract class PDFModel which is filled by the various models.

"""

from abc import ABC, abstractmethod, abstractproperty

class PDFModel(ABC):
    """An abstract class describing the key features of a PDF.
    """

    @abstractproperty
    def param_names(self):
        """This should return a list of names for the fitted parameters of the model.
        """
        pass

    @abstractproperty
    def init_params(self):
        """This should return a valid list of initial model parameters, for MC fits.
        """
        pass

    @abstractmethod
    def grid_values_func(self, xgrid):
        """This function should produce a grid values function, which takes
        in the model parameters, and produces the PDF values on the grid xgrid.
        """
        pass

    @abstractproperty
    def bayesian_prior(self):
        """This should return the Bayesian prior for a Nested Sampling fit.
        """
        pass
