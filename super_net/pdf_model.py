"""
super_net.pdf_model.py

This module implements an abstract class PDFModel which is filled by the various models.

Author: James Moore
Date: 5.1.2023
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
    def grid_values(self, params):
        """This should return a PDF grid given some model parameters. 
        """
        pass

    @abstractproperty
    def bayesian_prior(self):
        """This should return the Bayesian prior for a Nested Sampling fit.
        """
        pass
