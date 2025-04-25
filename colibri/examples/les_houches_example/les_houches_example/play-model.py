"""
les_houches_example.model.py
"""

import jax.numpy as jnp
import jax.scipy.special as jsp  
from colibri.pdf_model import PDFModel
from colibri.export_results import write_exportgrid
from colibri.constants import LHAPDF_XGRID, EXPORT_LABELS
import os
import pathlib
import yaml

class LesHouchesPDF(PDFModel):
    """
    A PDFModel implementation for the Les Houches parametrisation.
    """

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

    def _pdf_gluon(
        self, x, alpha_gluon, beta_gluon, norm_sigma, alpha_sigma, beta_sigma
    ):
        """Computes normalisation factor A_g in terms of free parameters and computes the gluon PDF."""
        A_g = (
            jsp.gamma(alpha_gluon + beta_gluon + 2)
            / (jsp.gamma(alpha_gluon + 1) * jsp.gamma(beta_gluon + 1))
        ) * (
            1
            - norm_sigma
            * (jsp.gamma(alpha_sigma + 1) * jsp.gamma(beta_sigma + 1))
            / jsp.gamma(alpha_sigma + beta_sigma + 2)
        )
        return A_g * x**alpha_gluon * (1 - x) ** beta_gluon

    def _pdf_sigma(self, x, norm_sigma, alpha_sigma, beta_sigma):
        """Compute the Sigma pdf."""
        return norm_sigma * x**alpha_sigma * (1 - x) ** beta_sigma

    def _A_uv(self, alpha_up, beta_up, epsilon_up, gamma_up):
        """Compute the normalization factor A_{u_v} in terms of free parameters."""
        return (2 / jsp.gamma(beta_up + 1)) / (
            (jsp.gamma(alpha_up) / jsp.gamma(alpha_up + beta_up + 1))
            + epsilon_up
            * (jsp.gamma(alpha_up + 0.5) / jsp.gamma(alpha_up + beta_up + 1.5))
            + gamma_up * (jsp.gamma(alpha_up + 1) / jsp.gamma(alpha_up + beta_up + 2))
        )

    def _A_dv(self, alpha_down, beta_down, epsilon_down, gamma_down):
        """Compute the normalization factor A_{d_v} in terms of free parameters."""
        return (1 / jsp.gamma(beta_down + 1)) / (
            (jsp.gamma(alpha_down) / jsp.gamma(alpha_down + beta_down + 1))
            + epsilon_down
            * (jsp.gamma(alpha_down + 0.5) / jsp.gamma(alpha_down + beta_down + 1.5))
            + gamma_down
            * (jsp.gamma(alpha_down + 1) / jsp.gamma(alpha_down + beta_down + 2))
        )

    def _pdf_valence(
        self,
        x,
        alpha_up,
        beta_up,
        epsilon_up,
        gamma_up,
        alpha_down,
        beta_down,
        epsilon_down,
        gamma_down,
    ):
        """ """
        A_uv = self._A_uv(alpha_up, beta_up, epsilon_up, gamma_up)
        A_dv = self._A_dv(alpha_down, beta_down, epsilon_down, gamma_down)
        up_valence = (
            x**alpha_up
            * (1 - x) ** beta_up
            * (1 + epsilon_up * jnp.sqrt(x) + gamma_up * x)
        )
        down_valence = (
            x**alpha_down
            * (1 - x) ** beta_down
            * (1 + epsilon_down * jnp.sqrt(x) + gamma_down * x)
        )
        return A_uv * up_valence + A_dv * down_valence

    def _pdf_valence3(
        self,
        x,
        alpha_up,
        beta_up,
        epsilon_up,
        gamma_up,
        alpha_down,
        beta_down,
        epsilon_down,
        gamma_down,
    ):
        """ """
        A_uv = self._A_uv(alpha_up, beta_up, epsilon_up, gamma_up)
        A_dv = self._A_dv(alpha_down, beta_down, epsilon_down, gamma_down)
        up_valence = (
            x**alpha_up
            * (1 - x) ** beta_up
            * (1 + epsilon_up * jnp.sqrt(x) + gamma_up * x)
        )
        down_valence = (
            x**alpha_down
            * (1 - x) ** beta_down
            * (1 + epsilon_down * jnp.sqrt(x) + gamma_down * x)
        )
        return A_uv * up_valence - A_dv * down_valence

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

            # Compute the PDFs for each flavour
            gluon_pdf = self._pdf_gluon(
                xgrid, alpha_gluon, beta_gluon, norm_sigma, alpha_sigma, beta_sigma
            )
            sigma_pdf = self._pdf_sigma(xgrid, norm_sigma, alpha_sigma, beta_sigma)
            valence_pdf = self._pdf_valence(
                xgrid,
                alpha_up,
                beta_up,
                epsilon_up,
                gamma_up,
                alpha_down,
                beta_down,
                epsilon_down,
                gamma_down,
            )
            valence3_pdf = self._pdf_valence3(
                xgrid,
                alpha_up,
                beta_up,
                epsilon_up,
                gamma_up,
                alpha_down,
                beta_down,
                epsilon_down,
                gamma_down,
            )
            t8_pdf = 0.4 * sigma_pdf  # T8 is a scaled version of Sigma

            # Build the PDF grid
            pdf_grid = jnp.array(
                [
                    jnp.zeros_like(xgrid),  # Photon
                    sigma_pdf,  # Σ
                    gluon_pdf,  # g
                    valence_pdf,  # V
                    valence3_pdf,  # V3
                    valence_pdf,  # V8 = V
                    jnp.zeros_like(xgrid),  # V15
                    jnp.zeros_like(xgrid),  # V24
                    jnp.zeros_like(xgrid),  # V35
                    valence3_pdf,  # T3 = V3
                    t8_pdf,  # T8
                    jnp.zeros_like(xgrid),  # T15
                    jnp.zeros_like(xgrid),  # T24
                    jnp.zeros_like(xgrid),  # T35
                ]
            )
            return pdf_grid

        return pdf_func


# ----------- Generate a grid for the Les Houches parametrisation ----------- #

# Define the xgrid
FIT_XGRID = jnp.array(LHAPDF_XGRID)

fitted_flavours = ["Sigma", "g", "V", "V3"]
lh_pdf_model = LesHouchesPDF(fitted_flavours=fitted_flavours)

pdf_grid_func = lh_pdf_model.grid_values_func(FIT_XGRID)

params =  [
    0.356,  # alpha_gluon
    10.9,   # beta_gluon
    0.718,  # alpha_up
    3.81,   # beta_up
    -1.56,  # epsilon_up
    3.30,   # gamma_up
    1.71,   # alpha_down
    10.0,   # beta_down
    -3.83,  # epsilon_down
    4.64,   # gamma_down
    0.211,  # norm_sigma
    -0.048, # alpha_sigma
    2.20,   # beta_sigma
]  
pdf_grid = pdf_grid_func(params)

WRITE_GRID = jnp.array(pdf_grid)[jnp.newaxis, :, :]
fit_name = "les_houches_parametrisation"

# Create directories if they do not exist
if not os.path.exists(fit_name):
    os.makedirs(fit_name)
    os.makedirs(fit_name + "/replicas")
    os.makedirs(fit_name + "/input")
    # create runcard.yaml firl in input folder
    runc = {"theoryid": 40000000}
    with open(fit_name + "/input/runcard.yaml", "w") as f:
        f.write(yaml.dump(runc))

replicas_path = fit_name / pathlib.PosixPath("replicas")

if not os.path.exists(replicas_path):
    os.makedirs(replicas_path)

# Loop to create 3 identical replicas. We create 3 so that evolve_fit runs with no errors. 
for replica_index in range(1, 3 + 1):
    
    grid_for_writing = WRITE_GRID[0]

    rep_path = replicas_path / f"replica_{replica_index}"
    rep_path.mkdir(exist_ok=True)
    grid_name = rep_path / fit_name

    
    write_exportgrid(
        grid_for_writing=grid_for_writing,
        grid_name=str(grid_name),
        replica_index=replica_index,
        Q=1.65,
        xgrid=LHAPDF_XGRID,
        export_labels=EXPORT_LABELS,
    )
