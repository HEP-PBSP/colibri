"""
example_pdf_model.model.py

"""

import jax.numpy as jnp
import jax.scipy.special as jsp # import this module to compute gamma functions
from colibri.pdf_model import PDFModel
from validphys import convolution


class ExamplePDFModel(PDFModel):
    """ """

    def __init__(self, fitted_flavours):
        self.fitted_flavours = fitted_flavours

    @property
    def param_names(self):
        """The fitted parameters of the model."""
        return [
            "alpha_gluon",
            "beta_gluon",
            "alpha_up",
            "beta_up",
            "epsilon_up",
            "gamma_up",
            "alpha_down",
            "beta_down",
            "epsilon_down",
            "gamma_down",
            "norm_sigma",
            "alpha_sigma",
            "beta_sigma",
        ]

    @property
    def n_parameters(self):
        """The number of parameters of the model."""
        return len(self.param_names)

    def _pdf_gluon(self, x, alpha_gluon, beta_gluon, norm_sigma, alpha_sigma, beta_sigma):
        """ Computes normalisation factor A_g in terms of free parameters and computes the gluon PDF."""
        A_g = (
        jsp.gamma(alpha_gluon + beta_gluon + 2)
        / (jsp.gamma(alpha_gluon + 1) * jsp.gamma(beta_gluon + 1))
         ) * (
        1 - norm_sigma
        * (jsp.gamma(alpha_sigma + 1) * jsp.gamma(beta_sigma + 1))
        / jsp.gamma(alpha_sigma + beta_sigma + 2)
          )
        return A_g * x **alpha_gluon * (1 - x) ** beta_gluon

    def _pdf_sigma(self, x, norm_sigma, alpha_sigma, beta_sigma):
        """ """
        return norm_sigma * x ** alpha_sigma * (1 - x) ** beta_sigma

    def _A_uv(self, alpha_up, beta_up, epsilon_up, gamma_up):
        """Compute the normalization factor A_{u_v} in terms of free parameters."""
        return (2 / jsp.gamma(beta_up + 1)) / (
        (jsp.gamma(alpha_up) / jsp.gamma(alpha_up + beta_up + 1))
        + epsilon_up * (jsp.gamma(alpha_up + 0.5) / jsp.gamma(alpha_up + beta_up + 1.5))
        + gamma_up * (jsp.gamma(alpha_up + 1) / jsp.gamma(alpha_up + beta_up + 2))
        )

    def _A_dv(self, alpha_down, beta_down, epsilon_down, gamma_down):
        """Compute the normalization factor A_{d_v} in terms of free parameters."""
        return (1 / jsp.gamma(beta_down + 1)) / (
        (jsp.gamma(alpha_down) / jsp.gamma(alpha_down + beta_down + 1))
        + epsilon_down * (jsp.gamma(alpha_down + 0.5) / jsp.gamma(alpha_down + beta_down + 1.5))
        + gamma_down * (jsp.gamma(alpha_down + 1) / jsp.gamma(alpha_down + beta_down + 2))
        )

    def _pdf_valence(self, x, alpha_up, beta_up, epsilon_up, gamma_up, alpha_down, beta_down, epsilon_down, gamma_down):
        """ """
        A_uv = self._A_uv(alpha_up, beta_up, epsilon_up, gamma_up)
        A_dv = self._A_dv(alpha_down, beta_down, epsilon_down, gamma_down)
        up_valence = x ** alpha_up * (1 - x) ** beta_up * (1 + epsilon_up * jnp.sqrt(x) + gamma_up * x)
        down_valence = x ** alpha_down * (1 - x) ** beta_down * (1 + epsilon_down * jnp.sqrt(x) + gamma_down * x)
        return A_uv * up_valence + A_dv * down_valence

    def _pdf_valence3(self, x, alpha_up, beta_up, epsilon_up, gamma_up, alpha_down, beta_down, epsilon_down, gamma_down):
        """ """
        A_uv = self._A_uv(alpha_up, beta_up, epsilon_up, gamma_up)
        A_dv = self._A_dv(alpha_down, beta_down, epsilon_down, gamma_down)
        up_valence = x ** alpha_up * (1 - x) ** beta_up * (1 + epsilon_up * jnp.sqrt(x) + gamma_up * x)
        down_valence = x ** alpha_down * (1 - x) ** beta_down * (1 + epsilon_down * jnp.sqrt(x) + gamma_down * x)
        return  A_uv * up_valence - A_dv * down_valence

    def grid_values_func(self, xgrid):
        """This function should produce a grid values function, which takes
        in the model parameters, and produces the PDF values on the grid xgrid.
        """
        xgrid = jnp.array(xgrid)

        def pdf_func(params):
            """ """
            alpha_gluon = params[0]
            beta_gluon = params[1]
            alpha_up = params[2]
            beta_up = params[3]
            epsilon_up = params[4]
            gamma_up = params[5]
            alpha_down = params[6]
            beta_down = params[7]
            epsilon_down = params[8]
            gamma_down = params[9]
            norm_sigma = params[10]
            alpha_sigma = params[11]   
            beta_sigma = params[12]
            pdf_grid = []
            for fl in convolution.FK_FLAVOURS:
                if fl in self.fitted_flavours:
                    gluon_pdf = self._pdf_gluon(xgrid, alpha_gluon, beta_gluon, norm_sigma, alpha_sigma, beta_sigma)
                    sigma_pdf = self._pdf_sigma(xgrid, norm_sigma, alpha_sigma, beta_sigma)
                    valence_pdf = self._pdf_valence(xgrid, alpha_up, beta_up, epsilon_up, gamma_up, alpha_down, beta_down, epsilon_down, gamma_down)
                    valence3_pdf = self._pdf_valence3(xgrid, alpha_up, beta_up, epsilon_up, gamma_up, alpha_down, beta_down, epsilon_down, gamma_down)
                    if fl == "g":
                        pdf_grid.append(gluon_pdf)
                    elif fl == "\\Sigma":                        
                        pdf_grid.append(sigma_pdf)
                    elif fl == "V":                        
                        pdf_grid.append(valence_pdf)
                    elif fl == "V3":                        
                        pdf_grid.append(valence3_pdf)
                    elif fl == "T3":
                        pdf_grid.append(valence3_pdf)
                    elif fl == "T8":
                        pdf_grid.append(0.4 * sigma_pdf)
                    elif fl == "V8":
                        pdf_grid.append(valence_pdf)                    
                    else:
                        pdf_grid.append(jnp.zeros_like(xgrid))
                else:
                    pdf_grid.append(jnp.zeros_like(xgrid))
            return jnp.array(pdf_grid)

        return pdf_func
