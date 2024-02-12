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
    output_size: int = 700  # 14 flavours * 50 x values

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size1, kernel_init=nn.initializers.glorot_normal())(x)
        x = jnp.tanh(x)
        x = nn.Dense(self.hidden_size2, kernel_init=nn.initializers.glorot_normal())(x)
        x = jnp.tanh(x)
        x = nn.Dense(self.output_size, kernel_init=nn.initializers.glorot_normal())(x)
        x = self.sum_rules(x)
        return x
    
    
    def sum_rules(self, x):
        """
        Apply the sum rule to: g+Sigma, V, V8, V3
        This is a special normalization layer.

        """
        
        # g+Sigma
        g_sigma_norm = jnp.trapz(x[50:100] + x[100:150], x=jnp.array(XGRID))
        x_g_sigma = jax.tree_map(lambda x: x[50:150] / g_sigma_norm, x)

        # V
        V_norm = jnp.trapz(x[150:200], x=jnp.array(XGRID))
        x_v = jax.tree_map(lambda x: x[150:200] / V_norm * 3, x)


        # V3
        V3_norm = jnp.trapz(x[200:250], x=jnp.array(XGRID))
        x_v3 = jax.tree_map(lambda x: x[200:250] / V3_norm, x)

        # V8
        V8_norm = jnp.trapz(x[250:300], x=jnp.array(XGRID))
        x_v8 = jax.tree_map(lambda x: x[250:300] / V8_norm * 3, x)

        new_x = jnp.concatenate((x[0:50], x_g_sigma, x_v, x_v3, x_v8, x[300:700]))
        return new_x
        


class NNPDFColibriModel(PDFModel):
    """The NNPDF4.0 PDF model reimplemented in colibri."""

    @property
    def param_names(self):
        """The fitted parameters of the model."""
        pass

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
        interpolation_grid = jnp.concatenate(
            (interpolation_grid, jnp.log(interpolation_grid))
        )

        pdf_model = NNDPF40DenseNN()

        @jax.jit
        def nn_model(params):
            pdf = pdf_model.apply(params, interpolation_grid).reshape(14, 50)
            return pdf

        return nn_model


def mc_initial_parameters(replica_index, parameterisation_grid):
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

    input_grid = jnp.array(parameterisation_grid)
    input_grid = jnp.concatenate((input_grid, jnp.log(input_grid)))

    model = NNDPF40DenseNN()
    init_params = model.init(rng, input_grid)

    return init_params
