"""
nnpdf_colibri.model.py

The nnpdf4.0 model.
"""

import jax
import jax.numpy as jnp

from flax import linen as nn

from validphys import convolution
from validphys.core import PDF

from colibri.pdf_model import PDFModel


def pdf_model():
    return NNPDFColibriModel()


class NNDPF40DenseNN(nn.Module):
    hidden_size1: int = 25
    hidden_size2: int = 20
    output_size: int = 14

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size1, kernel_init=nn.initializers.glorot_normal())(x)
        x = jnp.tanh(x)
        x = nn.Dense(self.hidden_size2, kernel_init=nn.initializers.glorot_normal())(x)
        x = jnp.tanh(x)
        x = nn.Dense(self.output_size, kernel_init=nn.initializers.glorot_normal())(x)
        return x



class NNPDFColibriModel(PDFModel):
    """The NNPDF4.0 PDF model reimplemented in colibri."""

    @property
    def param_names(self):
        """The fitted parameters of the model."""
        return [f"{fl}({x})" for fl in self.fitted_flavours for x in self.xgrids[fl]]

    def grid_values_func(self, interpolation_grid):
        """
        Takes a grid of x values and returns the NNPDF4.0 NN parameterisation of PDFs in the
        evolution basis.

        The parameterisation is given by a dense neural network with two input nodes 
        (X and log(X)), 2 hidden layers with 25 and 20 nodes respectively, and 14 output nodes.
        The NN has hyperbolic tangent activation functions and glorot normal initialisation.

        Parameters
        ----------
        interpolation_grid: jnp.array, default has shape (50,)
            The grid of x values to evaluate the PDFs at.

        """
        # Dense flax NN with architecture 
        # 2 -> 25 -> 20 -> 14
        
        interpolation_grid = jnp.concatenate([[interpolation_grid], [jnp.log(interpolation_grid)]], axis=0)

        pdf_model = NNDPF40DenseNN()

        @jax.jit
        def nn_model(params):
            return pdf_model.apply(params, interpolation_grid)
        
        return nn_model
            

                