"""
nnpdf_colibri.model.py

The nnpdf4.0 model.
"""

import jax
import jax.numpy as jnp

from flax import linen as nn

from colibri.pdf_model import PDFModel
from colibri.constants import XGRID


def pdf_model():
    return NNPDFColibriModel()


class NNDPF40DenseNN(nn.Module):
    hidden_size1: int = 25
    hidden_size2: int = 20
    output_size: int = 700 # 14 flavours * 50 x values

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
        
        interpolation_grid = jnp.array(interpolation_grid)
        interpolation_grid = jnp.concatenate((interpolation_grid, jnp.log(interpolation_grid) ))
        

        pdf_model = NNDPF40DenseNN()

        @jax.jit
        def nn_model(params):
            return pdf_model.apply(params, interpolation_grid).reshape(14,50)
        
        return nn_model
            

def mc_initial_parameters(replica_index):
    """
    This function initialises the parameters for the weight minimisation
    in a Monte Carlo fit.

    Parameters
    ----------
    replica_index: int
        The index of the replica.

    Returns
    -------
    initial_values: flax.core.frozen_dict.FrozenDict
        The initial values for the NN parameters.
    """
    # Getting the initial parameters of the model
    rng = jax.random.PRNGKey(replica_index)

    input_grid = jnp.array(XGRID)
    input_grid = jnp.concatenate((input_grid, jnp.log(input_grid) ))
    
    model = NNDPF40DenseNN()
    init_params = model.init(rng, input_grid)
    
    return init_params